"""
generate_csv_to_gif.py

Generate decoded skeletons (x_hat) + GIFs from two IMU CSV folders.

Uses your existing checkpoints:
  - smoke_stage1.pt  (Stage1: decoder)
  - smoke_stage2*.pt (Stage2: sensor encoder)
  - smoke_stage3*.pt (Stage3: denoiser)

Output:
  - outputs/generated_decoded.pt   (small, only a few samples)
  - outputs/gifs/*.gif

IMPORTANT:
- This script assumes your repo has these classes:
    from diffusion_model.sensor_model import TwoSensorIMUEncoder
    from diffusion_model.skeleton_model import SkeletonStage1Model
    from diffusion_model.model import GraphDenoiserMasked
  If class names differ in your repo, change the imports only.
"""

from __future__ import annotations

import os
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------
# EDIT ONLY IF YOUR CLASS NAMES DIFFER
# -------------------------
from diffusion_model.sensor_model import TwoSensorIMUEncoder
from diffusion_model.skeleton_model import SkeletonStage1Model
from diffusion_model.model import GraphDenoiserMasked


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_load(path: str, map_location="cpu"):
    # Silences the "weights_only" warning on newer torch, while staying compatible with older torch.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_read_csv_last3(path: str) -> np.ndarray:
    """Robust CSV -> float32 [T,3] (uses last 3 columns)."""
    try:
        df = pd.read_csv(path, header=None)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty CSV: {path}")
    except Exception as e:
        raise ValueError(f"Failed to read {path}: {e}")

    arr = df.to_numpy()
    if arr.size == 0:
        raise ValueError(f"No data in CSV: {path}")

    arr = pd.DataFrame(arr).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected >=3 cols in {path}, got {arr.shape}")

    arr = arr[:, -3:]  # last 3 accel cols
    arr[np.isinf(arr)] = np.nan

    col_mean = np.nanmean(arr, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    return arr.astype(np.float32)


def zscore_per_channel(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def build_chain_adjacency(num_joints: int, device: torch.device, include_self: bool = True) -> torch.Tensor:
    """Simple chain adjacency [J,J]. Good enough for forward() signature requirements."""
    A = torch.zeros((num_joints, num_joints), device=device, dtype=torch.float32)
    for j in range(num_joints - 1):
        A[j, j + 1] = 1.0
        A[j + 1, j] = 1.0
    if include_self:
        A.fill_diagonal_(1.0)
    return A


# -------------------------
# Dataset: paired windows
# -------------------------
class PairedIMUWindowDataset(Dataset):
    def __init__(
        self,
        imu_dir1: str,
        imu_dir2: str,
        window: int = 90,
        stride: int = 30,
        normalize: str = "zscore",
        max_files: Optional[int] = None,
    ):
        self.window = window
        self.stride = stride
        self.normalize = normalize

        f1 = sorted(glob.glob(os.path.join(imu_dir1, "*.csv")))
        f2 = sorted(glob.glob(os.path.join(imu_dir2, "*.csv")))
        if not f1:
            raise FileNotFoundError(f"No CSVs in imu_dir1: {imu_dir1}")
        if not f2:
            raise FileNotFoundError(f"No CSVs in imu_dir2: {imu_dir2}")

        m2 = {os.path.basename(p): p for p in f2}
        pairs = [(p, m2[os.path.basename(p)]) for p in f1 if os.path.basename(p) in m2]
        if not pairs:
            raise ValueError("No matching basenames between imu_dir1 and imu_dir2")

        if max_files is not None:
            pairs = pairs[:max_files]
        self.pairs = pairs

        self.cache_A1: List[np.ndarray] = []
        self.cache_A2: List[np.ndarray] = []
        self.file_names: List[str] = []
        self.index: List[Tuple[int, int]] = []  # (fi, start)

        for (p1, p2) in tqdm(self.pairs, desc="Loading IMU CSVs"):
            fname = os.path.basename(p1)
            try:
                A1 = safe_read_csv_last3(p1)
                A2 = safe_read_csv_last3(p2)
            except Exception as e:
                print(f"[skip] {fname}: {e}")
                continue

            T = min(len(A1), len(A2))
            if T < window:
                print(f"[skip] {fname}: too short T={T} < window={window}")
                continue

            A1 = A1[:T]
            A2 = A2[:T]

            if normalize == "zscore":
                A1 = zscore_per_channel(A1)
                A2 = zscore_per_channel(A2)
            elif normalize == "none":
                pass
            else:
                raise ValueError("normalize must be zscore or none")

            self.cache_A1.append(A1)
            self.cache_A2.append(A2)
            self.file_names.append(fname)

            fi = len(self.cache_A1) - 1
            for start in range(0, T - window + 1, stride):
                self.index.append((fi, start))

        if not self.index:
            raise ValueError("No windows created. Check window/stride and data lengths.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fi, start = self.index[idx]
        A1 = self.cache_A1[fi][start : start + self.window]
        A2 = self.cache_A2[fi][start : start + self.window]
        return {
            "A1": torch.from_numpy(A1).float(),  # [W,3]
            "A2": torch.from_numpy(A2).float(),  # [W,3]
            "file": self.file_names[fi],
            "start": torch.tensor(start, dtype=torch.long),
        }


# -------------------------
# DDPM sampler in latent space
# -------------------------
class DDPM:
    def __init__(self, T: int, beta_start: float, beta_end: float, device: torch.device):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def sample(self, eps_fn, shape: Tuple[int, ...], steps: int) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)

        # uniform subsampling of timesteps
        idx = torch.linspace(0, self.T - 1, steps).long().tolist()
        timesteps = list(reversed(idx))

        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps = eps_fn(x, t_batch)

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            abar_t = self.abar[t]

            mu = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - abar_t)) * eps)

            if t > 0:
                x = mu + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mu

        return x


# -------------------------
# GIF rendering (simple XY projection)
# -------------------------
def save_gif_xy(points_wj3: np.ndarray, out_gif: Path, fps: int = 15) -> None:
    """
    points_wj3: [W,J,3]
    Saves a 2D XY scatter GIF.
    """
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    W, J, _ = points_wj3.shape
    frames = []

    # autoscale
    xs = points_wj3[..., 0]
    ys = points_wj3[..., 1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    for t in range(W):
        fig = plt.figure(figsize=(4, 4), dpi=120)
        ax = fig.add_subplot(111)
        ax.scatter(points_wj3[t, :, 0], points_wj3[t, :, 1], s=12)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"t={t}")
        ax.axis("off")

        fig.canvas.draw()
        # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # draw the canvas first

        # Robust: works on newer Matplotlib (preferred)
        if hasattr(fig.canvas, "buffer_rgba"):
            img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # (H,W,4)
            img = img[..., :3]  # drop alpha -> (H,W,3)

        # Fallback: older Matplotlib using ARGB bytes
        else:
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
            img = buf[..., [1, 2, 3]]  # ARGB -> RGB
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_gif), frames, fps=fps)


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imu_dir1", type=str, required=True)
    p.add_argument("--imu_dir2", type=str, required=True)

    p.add_argument("--stage1_ckpt", type=str, default="checkpoints/smoke_stage1.pt")
    p.add_argument("--stage2_ckpt", type=str, default="checkpoints/smoke_stage2_accel_only.pt")
    p.add_argument("--stage3_ckpt", type=str, default="checkpoints/smoke_stage3_accel_only.pt")

    p.add_argument("--window", type=int, default=90)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "none"])
    p.add_argument("--max_files", type=int, default=30, help="limit files for generation (prevents huge outputs)")

    p.add_argument("--num_joints", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)

    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--sampling_steps", type=int, default=200)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)

    p.add_argument("--out_pt", type=str, default="outputs/generated_decoded.pt")
    p.add_argument("--gif_dir", type=str, default="outputs/gifs")
    p.add_argument("--num_gifs", type=int, default=10, help="how many windows to export as GIF")
    p.add_argument("--gif_fps", type=int, default=15)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    ds = PairedIMUWindowDataset(
        imu_dir1=args.imu_dir1,
        imu_dir2=args.imu_dir2,
        window=args.window,
        stride=args.stride,
        normalize=args.normalize,
        max_files=args.max_files,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ---- Load Stage1 (decoder) ----
    stage1 = SkeletonStage1Model(
        joint_dim=3,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=3,
        num_heads=8,
        diffusion_steps=args.T,
    ).to(device)

    ckpt1 = safe_torch_load(args.stage1_ckpt, map_location="cpu")
    # Accept common checkpoint formats
    if isinstance(ckpt1, dict) and "decoder" in ckpt1:
        stage1.decoder.load_state_dict(ckpt1["decoder"], strict=False)
    else:
        # fallback: try loading entire dict into stage1 (strict=False)
        stage1.load_state_dict(ckpt1 if isinstance(ckpt1, dict) else ckpt1, strict=False)

    stage1.eval()

    # ---- Load Stage2 (sensor encoder) ----
    sensor = TwoSensorIMUEncoder(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    ckpt2 = safe_torch_load(args.stage2_ckpt, map_location="cpu")
    sensor.load_state_dict(ckpt2 if isinstance(ckpt2, dict) else ckpt2, strict=False)
    sensor.eval()

    # ---- Load Stage3 (denoiser) ----
    denoiser = GraphDenoiserMasked(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    ckpt3 = safe_torch_load(args.stage3_ckpt, map_location="cpu")
    denoiser.load_state_dict(ckpt3 if isinstance(ckpt3, dict) else ckpt3, strict=False)
    denoiser.eval()

    adjacency = build_chain_adjacency(args.num_joints, device=device, include_self=True)

    ddpm = DDPM(T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, device=device)

    out_items: List[Dict[str, Any]] = []
    gif_dir = Path(args.gif_dir)
    gif_dir.mkdir(parents=True, exist_ok=True)

    made_gifs = 0

    for batch in tqdm(loader, desc="Generating+Decoding"):
        A1 = batch["A1"].to(device)  # [B,W,3]
        A2 = batch["A2"].to(device)
        B = A1.shape[0]

        with torch.no_grad():
            cond_out = sensor(A1, A2)
            # handle sensor returning (h_joint, h_seq)
            if isinstance(cond_out, tuple):
                h_joint, h_seq = cond_out
                cond = h_joint if (isinstance(h_joint, torch.Tensor) and h_joint.ndim >= 3) else h_seq
            else:
                cond = cond_out

            def eps_fn(z_t: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
                # GraphDenoiserMasked expects adjacency as 3rd arg.
                # conditioning should be passed after that (usually named "context").
                try:
                    eps = denoiser(z_t, t_batch, adjacency, cond)     # <-- SWAP HERE
                except TypeError:
                    eps = denoiser(z_t, t_batch, adjacency, context=cond)  # <-- common keyword
                if isinstance(eps, tuple):
                    eps = eps[0]
                return eps

            # Sample latent z0 [B,W,J,D]
            z0 = ddpm.sample(
                eps_fn=eps_fn,
                shape=(B, args.window, args.num_joints, args.latent_dim),
                steps=args.sampling_steps,
            )

            # Decode to coords x_hat [B,W,J,3]
            # Stage1 decoder signature differs across repos; try common patterns:
            try:
                x_hat = stage1.decoder(z0, adjacency)
            except Exception:
                x_hat = stage1.decoder(z0)

        # save a SMALL pt: store x_hat only (float16) + minimal metadata
        for i in range(B):
            item = {
                "file": batch["file"][i],
                "start": int(batch["start"][i].item()),
                "x_hat": x_hat[i].detach().cpu().to(torch.float16),  # [W,J,3]
            }
            out_items.append(item)

            # make gifs for first N windows
            if made_gifs < args.num_gifs:
                pts = item["x_hat"].to(torch.float32).numpy()  # [W,J,3]
                gif_path = gif_dir / (batch["file"][i].replace(".csv", f"_start{item['start']}.gif"))
                save_gif_xy(pts, gif_path, fps=args.gif_fps)
                made_gifs += 1

    out_path = Path(args.out_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_items, out_path)

    print(f"\nSaved decoded windows (x_hat) to: {out_path}")
    print(f"Saved {made_gifs} GIFs to: {gif_dir}")


if __name__ == "__main__":
    main()