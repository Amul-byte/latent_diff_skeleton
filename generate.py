#!/usr/bin/env python3
import os
import torch
import argparse
import imageio
import random
import inspect
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from diffusion_model.model import Stage3Model
from diffusion_model.sensor_model import SensorTGNNEncoder
from diffusion_model.model_loader import load_checkpoint
from diffusion_model.util import build_smartfall_bone_adjacency, assert_smartfall_bone_adjacency, resolve_device


def set_seed_local(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resample_to_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    x: [B,T,3] -> [B,target_len,3] (linear interpolation)
    Needed because IMU is 50Hz and skeleton is 30Hz.
    """
    if x.shape[1] == target_len:
        return x

    # interpolate expects [B,C,T]
    x_bct = x.transpose(1, 2)  # [B,3,T]
    x_rs = torch.nn.functional.interpolate(x_bct, size=target_len, mode="linear", align_corners=False)
    return x_rs.transpose(1, 2)  # [B,target_len,3]


def _bone_connections(num_joints: int) -> list[tuple[int, int]]:
    adj = build_smartfall_bone_adjacency(num_joints=num_joints, include_self=False, device=torch.device("cpu"))
    assert_smartfall_bone_adjacency(adj, include_self=False)
    idx = torch.nonzero(torch.triu(adj, diagonal=1), as_tuple=False)
    return [(int(i), int(j)) for i, j in idx.tolist()]


def warn_if_generation_looks_bad(samples: torch.Tensor, connections: list[tuple[int, int]]) -> None:
    """
    Prints warnings when output statistics look degenerate.
    This cannot prove correctness, but it helps catch bad checkpoints/training mismatch.
    """
    if not torch.isfinite(samples).all():
        print("[WARNING] Non-finite values found in generated skeletons. Check training/checkpoints.")
        return

    std_all = float(samples.std().item())
    if std_all < 1e-4:
        print("[WARNING] Generated skeletons are almost constant. Training may have collapsed.")

    if not connections:
        return

    edge_idx = torch.tensor(connections, dtype=torch.long)
    p1 = samples[:, :, edge_idx[:, 0], :]
    p2 = samples[:, :, edge_idx[:, 1], :]
    bone_len = (p1 - p2).norm(dim=-1)  # [B,T,E]

    med = float(bone_len.median().item())
    p95 = float(torch.quantile(bone_len.flatten(), 0.95).item())
    if med < 1e-4 or p95 > 20.0 * max(med, 1e-6):
        print(
            "[WARNING] Bone lengths are highly unstable. This usually means "
            "training/checkpoint mismatch or poor convergence."
        )


def visualize_skeleton(positions, save_path='skeleton_animation.gif', connections=None):
    # Expects positions [B,T,J,3]
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError(f"Expected positions [B,T,J,3], got {tuple(positions.shape)}")

    frames = []
    sample_idx = 0
    _, num_frames, num_joints, _ = positions.shape
    if connections is None:
        connections = _bone_connections(num_joints)

    sample = positions[sample_idx]  # [T,J,3]
    mins = sample.reshape(-1, 3).min(axis=0)
    maxs = sample.reshape(-1, 3).max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.1 * span
    mins -= pad
    maxs += pad

    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_facecolor('white')
        ax.grid(False)
        ax.set_axis_off()

        frame_xyz = positions[sample_idx, frame_idx]  # [J,3]
        for joint1, joint2 in connections:
            if joint1 >= num_joints or joint2 >= num_joints:
                continue
            joint1_coords = frame_xyz[joint1]
            joint2_coords = frame_xyz[joint2]

            xs = [joint1_coords[0], joint2_coords[0]]
            ys = [joint1_coords[1], joint2_coords[1]]
            zs = [joint1_coords[2], joint2_coords[2]]

            ax.plot(xs, ys, zs, color='darkblue', linewidth=2.0)

        ax.scatter(frame_xyz[:, 0], frame_xyz[:, 1], frame_xyz[:, 2], color='red', s=25)

        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.view_init(elev=-90, azim=-90)

        plt.tight_layout()
        fig.canvas.draw()

        # robust replacement for tostring_rgb (your env sometimes errors)
        if hasattr(fig.canvas, "buffer_rgba"):
            buf = np.asarray(fig.canvas.buffer_rgba())
            image = buf[..., :3].copy()
        else:
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        frames.append(image)
        plt.close(fig)

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    imageio.mimsave(save_path, frames, duration=0.2)
    print(f'GIF saved as {save_path}')


def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaNs detected in {name}")
    else:
        print(f"No NaNs in {name}")


def generate_samples(args, sensor_model, diffusion_model, device):
    # === SAME PATTERN ===
    # dataset -> dataloader -> take one batch -> context -> generate -> return samples

    from diffusion_model.dataset import SmartFallPairedSlidingWindowDataset, read_csv_files

    skeleton_data = read_csv_files(args.skeleton_folder)
    sensor1_data = read_csv_files(args.right_hip_folder)
    sensor2_data = read_csv_files(args.left_wrist_folder)

    ds_kwargs = dict(
        skeleton_data=skeleton_data,
        sensor1_data=sensor1_data,
        sensor2_data=sensor2_data,
        window_size=args.window_size,
        stride=args.window_stride,
        fall_activities=tuple(args.fall_activities),
        drop_misaligned=args.drop_misaligned,
        imu_normalization=args.imu_norm,
        imu_stats=None,
        imu_eps=args.imu_eps,
        imu_clip=args.imu_clip,
        sensor_names=("right_hip", "left_wrist"),
        sensor_roots=(args.right_hip_folder, args.left_wrist_folder),
        strict_sensor_identity=True,
    )
    if "align_mode" in inspect.signature(SmartFallPairedSlidingWindowDataset.__init__).parameters:
        ds_kwargs["align_mode"] = args.align_mode
    dataset = SmartFallPairedSlidingWindowDataset(**ds_kwargs)

    if len(dataset) == 0:
        raise ValueError("Dataset has zero windows for generation. Try --align_mode truncate_min and/or smaller --window_size.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    generated_samples = []
    sensor_model.eval()
    diffusion_model.eval()

    with torch.no_grad():
        batch = next(iter(dataloader))

        # Expect your PairedDataset returns dict (your codebase style)
        # batch["A1"] [B,T,3], batch["A2"] [B,T,3], batch["y"] [B]
        if isinstance(batch, dict):
            sensor1 = batch["A1"]
            sensor2 = batch["A2"]
            label_index = batch.get("y", None)
        else:
            # if your dataset returns tuples, map them here
            # (skeleton, sensor1, sensor2, label)
            _, sensor1, sensor2, label_index = batch

        sensor1 = sensor1.to(device)
        sensor2 = sensor2.to(device)

        # label handling same as your pattern
        if label_index is None:
            label_index = torch.zeros((sensor1.shape[0],), dtype=torch.long, device=device)
        else:
            label_index = label_index.to(device)
            if label_index.ndim == 2:
                label_index = torch.argmax(label_index, dim=1)
            else:
                label_index = label_index.long()

        # --- FIX: resample IMU to match window_size (IMU 50Hz vs skeleton 30Hz) ---
        sensor1 = _resample_to_len(sensor1, args.window_size)
        sensor2 = _resample_to_len(sensor2, args.window_size)

        # context (h) from your IMU encoder
        h_joint, _h_seq = sensor_model(sensor1, sensor2)  # h_joint [B,T,J,D]
        check_for_nans(h_joint, "context(h_joint)")

        # Generate latent using your diffusion model (Stage3Model)
        # This is the equivalent of diffusion_process.generate(...) in your sample
        adjacency = build_smartfall_bone_adjacency(num_joints=diffusion_model.num_joints, include_self=True, device=device)
        assert_smartfall_bone_adjacency(adjacency, include_self=True)

        batch_n = sensor1.shape[0]
        z0_hat = diffusion_model.sample_latent(
            batch_size=batch_n,
            window=args.window_size,
            adjacency=adjacency,
            device=device,
            h=h_joint,
            steps=args.timesteps,
        )
        check_for_nans(z0_hat, "generated_latent(z0_hat)")

        # Decode to skeleton coords: [B,T,J,3]
        x_hat = diffusion_model.decode(z0_hat, adjacency)
        check_for_nans(x_hat, "generated_sample(x_hat)")

        generated_samples.append(x_hat.cpu())

    generated_samples = torch.cat(generated_samples, dim=0)  # [B,T,J,3]
    return generated_samples


def main(args):
    set_seed_local(args.seed)
    device = resolve_device(args.device)

    if args.num_joints != 32:
        raise ValueError(
            f"num_joints must be 32 for SmartFall bone adjacency, got {args.num_joints}."
        )

    # === Load models (same pattern) ===
    sensor_model = SensorTGNNEncoder(
        latent_dim=args.latent_dim,
        num_joints=args.num_joints,
        hidden_dim=args.imu_hidden_dim,
    ).to(device)

    if args.sensor_ckpt is not None:
        load_checkpoint(args.sensor_ckpt, sensor_model, strict=False, map_location=device)

    diffusion_model = Stage3Model(
        latent_dim=args.latent_dim,
        num_joints=args.num_joints,
        num_classes=args.num_classes,
        diffusion_steps=args.diffusion_steps,
        window=args.window_size,
    ).to(device)

    load_checkpoint(args.stage3_ckpt, diffusion_model, strict=False, map_location=device)

    # === Generate samples ===
    print("Generating samples based on sensor inputs...")
    generated_samples = generate_samples(args, sensor_model, diffusion_model, device)
    if generated_samples.shape[2] != args.num_joints:
        raise ValueError(
            f"Generated joints={generated_samples.shape[2]} but args.num_joints={args.num_joints}."
        )

    connections = _bone_connections(args.num_joints)
    warn_if_generation_looks_bad(generated_samples, connections)

    visualize_skeleton(
        generated_samples.numpy(),
        save_path=args.out_gif
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Skeleton GIF (your codebase, same pattern)")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    # checkpoints
    parser.add_argument("--stage3_ckpt", type=str, required=True)
    parser.add_argument("--sensor_ckpt", type=str, required=True)

    # dataset paths (same style as your sample)
    parser.add_argument(
        "--right_hip_folder",
        type=str,
        required=True,
        help="Option A (accel-only): right-hip accelerometer CSV folder (proposal A/Omega mapping).",
    )
    parser.add_argument(
        "--left_wrist_folder",
        type=str,
        required=True,
        help="Option A (accel-only): left-wrist accelerometer CSV folder (proposal A/Omega mapping).",
    )
    parser.add_argument("--skeleton_folder", type=str, required=True)

    # generation params
    parser.add_argument("--window_size", type=int, default=90)
    parser.add_argument("--window_stride", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=None, help="sampling steps (<= diffusion_steps); None uses full")
    parser.add_argument(
        "--align_mode",
        type=str,
        default="truncate_min",
        choices=["strict", "truncate_min"],
        help="strict: require equal sequence lengths; truncate_min: use overlap across modalities",
    )
    parser.add_argument("--drop_misaligned", action="store_true", help="Only used with align_mode=strict")
    parser.set_defaults(drop_misaligned=False)
    parser.add_argument("--fall_activities", type=int, nargs="+", default=[10, 11, 12, 13, 14])
    parser.add_argument("--imu_norm", type=str, default="zscore", choices=["none", "zscore"])
    parser.add_argument("--imu_eps", type=float, default=1e-6)
    parser.add_argument("--imu_clip", type=float, default=6.0)

    # model params (must match how you trained Stage3)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_joints", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--diffusion_steps", type=int, default=500)
    parser.add_argument("--imu_hidden_dim", type=int, default=256)

    # output
    parser.add_argument("--out_gif", type=str, default="./gif_tl/generated.gif")

    args = parser.parse_args()
    main(args)
