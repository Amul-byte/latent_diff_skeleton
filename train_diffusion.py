import os
import re
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -------------------------
# Utils: robust CSV loading
# -------------------------
def safe_load_csv_float32(fpath: str) -> np.ndarray | None:
    try:
        arr = np.genfromtxt(
            fpath,
            delimiter=",",
            dtype=np.float32,
            invalid_raise=False,
        )
    except Exception as e:
        print(f"[SKIP] Failed reading {fpath}: {e}")
        return None

    if arr is None or not hasattr(arr, "size") or arr.size == 0:
        print(f"[SKIP] {fpath}: empty/bad read")
        return None

    if arr.ndim == 0:
        print(f"[SKIP] {fpath}: scalar read (likely empty)")
        return None

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr


def coerce_skel_to_Tx96(arr: np.ndarray, fpath: str) -> np.ndarray | None:
    """
    Accept:
      - [T, 96]
      - [T, 97] -> drop first col (common if index col exists)
      - [T, 32, 3] -> reshape
      - [T*96] -> reshape
    """
    if arr.ndim == 2:
        if arr.shape[1] == 96:
            return arr
        if arr.shape[1] == 97:
            return arr[:, 1:]
        print(f"[SKIP] {fpath}: unexpected skeleton cols {arr.shape}")
        return None

    if arr.ndim == 3 and arr.shape[1:] == (32, 3):
        T = arr.shape[0]
        return arr.reshape(T, 96)

    if arr.ndim == 1 and (arr.size % 96 == 0):
        T = arr.size // 96
        return arr.reshape(T, 96)

    print(f"[SKIP] {fpath}: unexpected skeleton shape {arr.shape}")
    return None


def coerce_sensor_to_Tx3(arr: np.ndarray, fpath: str, use_last_3_cols: bool = True) -> np.ndarray | None:
    """
    Accept:
      - [T, 3]
      - [T, >=3] -> use last 3 or first 3
      - [T*3] -> reshape
    """
    if arr.ndim == 2:
        if arr.shape[1] < 3:
            print(f"[SKIP] {fpath}: need >=3 cols, got {arr.shape}")
            return None
        if arr.shape[1] == 3:
            return arr
        return arr[:, -3:] if use_last_3_cols else arr[:, :3]

    if arr.ndim == 1 and (arr.size % 3 == 0):
        T = arr.size // 3
        return arr.reshape(T, 3)

    print(f"[SKIP] {fpath}: unexpected sensor shape {arr.shape}")
    return None


def parse_label_from_filename(fname: str, num_classes: int) -> int:
    m = re.search(r"A(\d+)", fname)
    if not m:
        return 0
    a = int(m.group(1))
    y = max(0, a - 1)
    if y >= num_classes:
        y = num_classes - 1
    return y


# -------------------------
# Dataset: (skeleton, sensor1, sensor2, label) windows
# -------------------------
class MultiModalWindowDataset(Dataset):
    def __init__(
        self,
        skeleton_dir: str,
        sensor_dir1: str,
        sensor_dir2: str,
        window: int = 90,
        stride: int = 30,
        num_classes: int = 12,
        use_last_3_cols: bool = True,
        label_mode: str = "filename",  # "filename" or "zero"
    ):
        self.window = window
        self.stride = stride
        self.num_classes = num_classes
        self.use_last_3_cols = use_last_3_cols
        self.label_mode = label_mode

        skel_files = sorted(glob.glob(os.path.join(skeleton_dir, "*.csv")))
        s1_files = sorted(glob.glob(os.path.join(sensor_dir1, "*.csv")))
        s2_files = sorted(glob.glob(os.path.join(sensor_dir2, "*.csv")))
        if not skel_files:
            raise FileNotFoundError(f"No skeleton CSVs in {skeleton_dir}")
        if not s1_files:
            raise FileNotFoundError(f"No sensor1 CSVs in {sensor_dir1}")
        if not s2_files:
            raise FileNotFoundError(f"No sensor2 CSVs in {sensor_dir2}")

        skel_map = {os.path.basename(p): p for p in skel_files}
        s1_map = {os.path.basename(p): p for p in s1_files}
        s2_map = {os.path.basename(p): p for p in s2_files}
        common = sorted(set(skel_map.keys()) & set(s1_map.keys()) & set(s2_map.keys()))
        if not common:
            raise ValueError("No matching filenames across skeleton/sensor1/sensor2 folders.")

        # Cache per-file arrays AFTER filtering bad files
        self.skel_cache: list[np.ndarray] = []
        self.s1_cache: list[np.ndarray] = []
        self.s2_cache: list[np.ndarray] = []
        self.names: list[str] = []
        self.index: list[tuple[int, int]] = []  # (cache_fi, start)

        skipped = 0
        for fname in common:
            fp_skel = skel_map[fname]
            fp_s1 = s1_map[fname]
            fp_s2 = s2_map[fname]

            sk = safe_load_csv_float32(fp_skel)
            a1 = safe_load_csv_float32(fp_s1)
            a2 = safe_load_csv_float32(fp_s2)
            if sk is None or a1 is None or a2 is None:
                skipped += 1
                continue

            sk = coerce_skel_to_Tx96(sk, fp_skel)
            a1 = coerce_sensor_to_Tx3(a1, fp_s1, use_last_3_cols=self.use_last_3_cols)
            a2 = coerce_sensor_to_Tx3(a2, fp_s2, use_last_3_cols=self.use_last_3_cols)
            if sk is None or a1 is None or a2 is None:
                skipped += 1
                continue

            T = min(sk.shape[0], a1.shape[0], a2.shape[0])
            if T < self.window:
                skipped += 1
                continue

            sk = sk[:T]
            a1 = a1[:T]
            a2 = a2[:T]

            cache_fi = len(self.skel_cache)
            self.skel_cache.append(sk)
            self.s1_cache.append(a1)
            self.s2_cache.append(a2)
            self.names.append(fname)

            for start in range(0, T - self.window + 1, self.stride):
                self.index.append((cache_fi, start))

        if not self.index:
            raise ValueError("No windows created. Check window/stride or file lengths.")

        print(
            f"[Dataset] common_files={len(common)} valid_files={len(self.skel_cache)} "
            f"skipped={skipped} total_windows={len(self.index)}"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, start = self.index[idx]
        sk = self.skel_cache[fi][start : start + self.window]  # [W,96]
        s1 = self.s1_cache[fi][start : start + self.window]    # [W,3]
        s2 = self.s2_cache[fi][start : start + self.window]    # [W,3]

        if self.label_mode == "filename":
            y = parse_label_from_filename(self.names[fi], self.num_classes)
        else:
            y = 0

        return (
            torch.from_numpy(sk).float(),
            torch.from_numpy(s1).float(),
            torch.from_numpy(s2).float(),
            torch.tensor(y, dtype=torch.long),
        )


# -------------------------
# Diffusion schedule (DDPM)
# -------------------------
class DDPMScheduler:
    def __init__(self, T: int, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.T = T
        self.device = device
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = abar
        self.sqrt_ab = torch.sqrt(abar)
        self.sqrt_1mab = torch.sqrt(1.0 - abar)

    def q_sample(self, x0, t, noise):
        # x0: [B,W,D], t: [B]
        sqrt_ab = self.sqrt_ab[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_1mab[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise


# -------------------------
# Conditioning: two-sensor encoder -> context vector
# -------------------------
class SensorEncoder(nn.Module):
    def __init__(self, hidden=128, out_dim=256):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=3, hidden_size=hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=3, hidden_size=hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.proj = nn.Linear(2 * hidden, out_dim)

    def forward(self, s1, s2):  # [B,W,3],[B,W,3]
        o1, _ = self.lstm1(s1)
        o2, _ = self.lstm2(s2)
        h = torch.cat([o1[:, -1, :], o2[:, -1, :]], dim=1)
        return self.proj(h)  # [B,out_dim]


# -------------------------
# Time embedding
# -------------------------
def timestep_embedding(t, dim: int):
    # t: [B] int64
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half, device=t.device).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # [B,dim]


# -------------------------
# Denoiser: predicts noise epsilon
# Uses per-frame conditioning (context + time + label)
# -------------------------
class NoisePredictor(nn.Module):
    def __init__(self, x_dim=96, ctx_dim=256, t_dim=128, num_classes=12, width=512):
        super().__init__()
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, 64)

        in_dim = x_dim + ctx_dim + t_dim + 64
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, x_dim),
        )

        self.t_dim = t_dim

    def forward(self, xt, t, ctx, y):
        # xt: [B,W,96], t:[B], ctx:[B,ctx_dim], y:[B]
        B, W, D = xt.shape
        te = timestep_embedding(t, self.t_dim)  # [B,t_dim]
        ye = self.label_emb(y)                  # [B,64]

        # expand to per-frame conditioning
        ctx_e = ctx.unsqueeze(1).expand(B, W, ctx.shape[-1])
        te_e = te.unsqueeze(1).expand(B, W, te.shape[-1])
        ye_e = ye.unsqueeze(1).expand(B, W, ye.shape[-1])

        inp = torch.cat([xt, ctx_e, te_e, ye_e], dim=-1)  # [B,W,in_dim]
        eps_hat = self.net(inp)                           # [B,W,96]
        return eps_hat


# -------------------------
# Train
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skeleton_dir", type=str, required=True)
    p.add_argument("--sensor_dir1", type=str, required=True)
    p.add_argument("--sensor_dir2", type=str, required=True)

    p.add_argument("--window", type=int, default=90)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--num_classes", type=int, default=12)
    p.add_argument("--label_mode", type=str, default="filename", choices=["filename", "zero"])
    p.add_argument("--use_last_3_cols", action="store_true")

    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="best_diffusion_model.pth")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = MultiModalWindowDataset(
        skeleton_dir=args.skeleton_dir,
        sensor_dir1=args.sensor_dir1,
        sensor_dir2=args.sensor_dir2,
        window=args.window,
        stride=args.stride,
        num_classes=args.num_classes,
        use_last_3_cols=args.use_last_3_cols,
        label_mode=args.label_mode,
    )

    # quick split
    n = len(ds)
    n_train = int(0.8 * n)
    train_ds, val_ds = torch.utils.data.random_split(
        ds,
        [n_train, n - n_train],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    device = torch.device(args.device)
    sched = DDPMScheduler(T=args.timesteps, device=device)

    sensor_enc = SensorEncoder(hidden=128, out_dim=256).to(device)
    denoiser = NoisePredictor(
        x_dim=96, ctx_dim=256, t_dim=128,
        num_classes=args.num_classes, width=512
    ).to(device)

    params = list(sensor_enc.parameters()) + list(denoiser.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    best_val = 1e9

    for epoch in range(1, args.epochs + 1):
        sensor_enc.train()
        denoiser.train()
        train_loss = 0.0

        for x0, s1, s2, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            x0 = x0.to(device, non_blocking=True)  # [B,W,96]
            s1 = s1.to(device, non_blocking=True)  # [B,W,3]
            s2 = s2.to(device, non_blocking=True)  # [B,W,3]
            y = y.to(device, non_blocking=True)    # [B]

            B = x0.shape[0]
            t = torch.randint(0, args.timesteps, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt = sched.q_sample(x0, t, noise)

            ctx = sensor_enc(s1, s2)               # [B,256]
            eps_hat = denoiser(xt, t, ctx, y)      # [B,W,96]

            loss = F.mse_loss(eps_hat, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # val
        sensor_enc.eval()
        denoiser.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x0, s1, s2, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                x0 = x0.to(device, non_blocking=True)
                s1 = s1.to(device, non_blocking=True)
                s2 = s2.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                B = x0.shape[0]
                t = torch.randint(0, args.timesteps, (B,), device=device, dtype=torch.long)
                noise = torch.randn_like(x0)
                xt = sched.q_sample(x0, t, noise)

                ctx = sensor_enc(s1, s2)
                eps_hat = denoiser(xt, t, ctx, y)
                loss = F.mse_loss(eps_hat, noise)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch}: train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "sensor_encoder": sensor_enc.state_dict(),
                "denoiser": denoiser.state_dict(),
                "timesteps": args.timesteps,
                "window": args.window,
                "stride": args.stride,
                "num_classes": args.num_classes,
            }
            torch.save(ckpt, args.out)
            print(f"Saved best checkpoint -> {args.out} (val_mse={best_val:.6f})")


if __name__ == "__main__":
    main()