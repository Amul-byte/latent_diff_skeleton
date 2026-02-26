import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# =========================
# EDIT THESE (repo-specific)
# =========================
# 1) Import your decoder class
# Example candidates in your repo: GraphDecoder, SkeletonStage1Model (with .decoder), etc.
#
# OPTION A (if you have GraphDecoder):
# from diffusion_model.graph_modules import GraphDecoder
#
# OPTION B (if you have SkeletonStage1Model and want to use model.decoder):
# from diffusion_model.skeleton_model import SkeletonStage1Model
#
from diffusion_model.graph_modules import GraphDecoder  # <-- EDIT if needed


def build_decoder(latent_dim: int, hidden_dim: int, num_layers: int, num_heads: int):
    """
    Construct your decoder module.
    EDIT constructor args to match your GraphDecoder signature.
    """
    # Example: GraphDecoder(d_model=256, heads=8, depth=3, dropout=0.1, hops=1)
    # If your GraphDecoder signature differs, edit this.
    return GraphDecoder(d_model=hidden_dim, heads=num_heads, depth=num_layers, dropout=0.1, hops=1)
# =========================


def safe_torch_load(path: str, map_location="cpu"):
    # Avoid FutureWarning spam on newer PyTorch
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_chain_adjacency(num_joints: int, device: torch.device, include_self: bool = True) -> torch.Tensor:
    A = torch.zeros((num_joints, num_joints), device=device, dtype=torch.float32)
    for j in range(num_joints - 1):
        A[j, j + 1] = 1.0
        A[j + 1, j] = 1.0
    if include_self:
        A.fill_diagonal_(1.0)
    return A


def default_edges(num_joints: int = 32):
    return [(i, i + 1) for i in range(num_joints - 1)]


def save_skeleton_gif(x_tj3, out_gif, edges=None, fps=30, stride=1, title=None):
    x_tj3 = np.asarray(x_tj3)
    T, J, _ = x_tj3.shape
    edges = edges or default_edges(J)

    x_tj3 = x_tj3[::stride]
    T = x_tj3.shape[0]

    xy = x_tj3[:, :, :2]  # [T,J,2]

    xmin, ymin = xy.min(axis=(0, 1))
    xmax, ymax = xy.max(axis=(0, 1))
    pad = 0.10 * max(xmax - xmin, ymax - ymin, 1e-6)
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad

    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    if title:
        ax.set_title(title)

    scat = ax.scatter([], [], s=20)
    lines = []
    for _ in edges:
        (ln,) = ax.plot([], [], linewidth=2)
        lines.append(ln)

    def init():
        scat.set_offsets(np.zeros((J, 2)))
        for ln in lines:
            ln.set_data([], [])
        return [scat] + lines

    def update(frame):
        pts = xy[frame]
        scat.set_offsets(pts)
        for k, (a, b) in enumerate(edges):
            xa, ya = pts[a]
            xb, yb = pts[b]
            lines[k].set_data([xa, xb], [ya, yb])
        return [scat] + lines

    anim = FuncAnimation(fig, update, frames=T, init_func=init, blit=True)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


def decode_one(decoder, z0_wjd, adjacency, device):
    """
    z0_wjd: [W,J,D]
    returns x_hat: [W,J,3] (or whatever decoder outputs)
    """
    z = torch.tensor(z0_wjd, dtype=torch.float32, device=device).unsqueeze(0)  # [1,W,J,D]
    with torch.no_grad():
        # Try common signatures: decoder(z, adjacency) or decoder(z)
        try:
            out = decoder(z, adjacency)
        except TypeError:
            out = decoder(z)

    # Handle tuple outputs
    if isinstance(out, tuple):
        out = out[0]

    out = out.squeeze(0).detach().cpu().numpy()  # [W,J,3] expected
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_in", type=str, required=True, help="generated.pt with z0")
    ap.add_argument("--decoder_ckpt", type=str, required=True, help="decoder checkpoint (.pth/.pt)")
    ap.add_argument("--out_dir", type=str, default="outputs/gifs")
    ap.add_argument("--max_items", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--num_joints", type=int, default=32)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)   # EDIT if your decoder uses a different d_model
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--num_heads", type=int, default=8)

    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device)

    items = safe_torch_load(args.pt_in, map_location="cpu")
    if not isinstance(items, list):
        raise ValueError("Expected pt_in to be a list of dicts")

    # Build + load decoder
    decoder = build_decoder(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    ckpt = safe_torch_load(args.decoder_ckpt, map_location="cpu")

    # Try common checkpoint formats
    if isinstance(ckpt, dict) and "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"], strict=False)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        decoder.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        decoder.load_state_dict(ckpt, strict=False)

    decoder.eval()

    adjacency = build_chain_adjacency(args.num_joints, device=device, include_self=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(items), args.max_items)
    print(f"Loaded {len(items)} items; decoding + converting {n} to GIF...")

    for i in range(n):
        it = items[i]
        if "z0" not in it:
            raise ValueError(f"Item {i} missing z0. Keys={list(it.keys())}")

        z0 = it["z0"]
        z0 = z0.detach().cpu().numpy() if torch.is_tensor(z0) else np.asarray(z0)

        # Decode
        x_hat = decode_one(decoder, z0, adjacency, device)

        # Sanity check
        if x_hat.ndim != 3 or x_hat.shape[-1] != 3:
            raise ValueError(f"Decoded x_hat has unexpected shape: {x_hat.shape}")

        fname = it.get("file", f"sample{i}")
        start = it.get("start", 0)
        gif_name = f"{Path(fname).stem}_start{start}_idx{i}.gif"
        out_gif = out_dir / gif_name

        save_skeleton_gif(
            x_tj3=x_hat,
            out_gif=out_gif,
            edges=None,
            fps=args.fps,
            stride=args.stride,
            title=f"{fname} start={start}",
        )
        print("Saved:", out_gif)

    print("Done.")


if __name__ == "__main__":
    main()