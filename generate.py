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
from diffusion_model.util import build_chain_adjacency, resolve_device, set_seed


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


def visualize_skeleton(positions, save_path='skeleton_animation.gif'):
    # SAME pattern as your example: expects positions [B, T, D]
    # Your Stage3 outputs [B,T,J,3], so we convert before calling this.
    connections = [
        (0, 1),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9),
        (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15)
    ]

    frames = []
    sample_idx = 0

    num_frames = positions.shape[1]
    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_facecolor('white')
        ax.grid(False)
        ax.set_axis_off()

        for joint1, joint2 in connections:
            joint1_coords = positions[sample_idx, frame_idx, joint1 * 3:(joint1 * 3) + 3]
            joint2_coords = positions[sample_idx, frame_idx, joint2 * 3:(joint2 * 3) + 3]
            if len(joint1_coords) < 3 or len(joint2_coords) < 3:
                continue

            xs = [joint1_coords[0], joint2_coords[0]]
            ys = [joint1_coords[1], joint2_coords[1]]
            zs = [joint1_coords[2], joint2_coords[2]]

            ax.plot(xs, ys, zs, marker='o', color='darkblue')
            ax.scatter(joint1_coords[0], joint1_coords[1], joint1_coords[2], color='red', s=50)
            ax.scatter(joint2_coords[0], joint2_coords[1], joint2_coords[2], color='red', s=50)

        ax.set_box_aspect([1, 1, 1])
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
    sensor1_data = read_csv_files(args.sensor_folder1)
    sensor2_data = read_csv_files(args.sensor_folder2)

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
        sensor_names=("sensor1", "sensor2"),
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
        adjacency = build_chain_adjacency(num_joints=diffusion_model.num_joints, include_self=True, device=device)

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

    # Convert [B,T,J,3] -> [B,T,J*3] to match your visualize_skeleton() exactly
    B, T, J, D = generated_samples.shape
    flat = generated_samples.numpy().reshape(B, T, J * D)

    visualize_skeleton(
        flat,
        save_path=args.out_gif
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Skeleton GIF (your codebase, same pattern)")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    # checkpoints
    parser.add_argument("--stage3_ckpt", type=str, required=True)
    parser.add_argument("--sensor_ckpt", type=str, default=None)

    # dataset paths (same style as your sample)
    parser.add_argument("--sensor_folder1", type=str, required=True)
    parser.add_argument("--sensor_folder2", type=str, required=True)
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
    parser.add_argument("--imu_norm", type=str, default="none", choices=["none", "zscore"])
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
