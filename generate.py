"""Generation entrypoint for conditional skeleton synthesis."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion_model.dataset import IMUDataset, ToyConfig
from diffusion_model.model import Stage3Model
from diffusion_model.model_loader import freeze_module, load_checkpoint, verify_frozen
from diffusion_model.sensor_model import SensorTGNNEncoder
from diffusion_model.skeleton_model import SkeletonStage1Model
from diffusion_model.util import build_chain_adjacency, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sample generation.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate skeleton samples from IMU conditioning")
    parser.add_argument("--stage1-ckpt", type=str, default="checkpoints/stage1.pt")
    parser.add_argument("--stage2-ckpt", type=str, default="checkpoints/stage2.pt")
    parser.add_argument("--stage3-ckpt", type=str, default="checkpoints/stage3.pt")

    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window", type=int, default=90)
    parser.add_argument("--joints", type=int, default=32)
    parser.add_argument("--joint-dim", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=14)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--sampling-steps", type=int, default=100)

    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-torch-geometric", action="store_true")
    parser.add_argument("--output", type=str, default="outputs/generated.pt")
    return parser.parse_args()


def _load_stage1_model(args: argparse.Namespace, device: torch.device) -> SkeletonStage1Model:
    """Instantiate and load Stage 1 model.

    Args:
        args: Parsed CLI arguments.
        device: Target device.

    Returns:
        Loaded Stage 1 model.
    """
    model = SkeletonStage1Model(
        joint_dim=args.joint_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        diffusion_steps=args.diffusion_steps,
        use_torch_geometric=args.use_torch_geometric,
    ).to(device)

    checkpoint_path = Path(args.stage1_ckpt)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

    load_checkpoint(str(checkpoint_path), model.encoder, strict=False, map_location=device)
    load_checkpoint(str(checkpoint_path), model.decoder, strict=False, map_location=device)
    load_checkpoint(str(checkpoint_path), model.denoiser, strict=False, map_location=device)

    freeze_module(model.encoder)
    assert verify_frozen(model.encoder)
    model.eval()
    return model


def _load_stage2_sensor(args: argparse.Namespace, device: torch.device) -> SensorTGNNEncoder:
    """Instantiate and load Stage 2 sensor encoder.

    Args:
        args: Parsed CLI arguments.
        device: Target device.

    Returns:
        Loaded sensor encoder.
    """
    sensor = SensorTGNNEncoder(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        hidden_dim=args.hidden_dim,
    ).to(device)

    checkpoint_path = Path(args.stage2_ckpt)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {checkpoint_path}")

    load_checkpoint(str(checkpoint_path), sensor, strict=False, map_location=device)

    freeze_module(sensor)
    assert verify_frozen(sensor)
    sensor.eval()
    return sensor


def _build_stage3_sampler(args: argparse.Namespace, stage1_model: SkeletonStage1Model, device: torch.device) -> Stage3Model:
    """Instantiate Stage 3 sampler and load available weights.

    Args:
        args: Parsed CLI arguments.
        stage1_model: Loaded Stage 1 model.
        device: Target device.

    Returns:
        Stage 3 model with denoiser/decoder/classifier.
    """
    stage3 = Stage3Model(
        latent_dim=args.latent_dim,
        joint_dim=args.joint_dim,
        num_joints=args.joints,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        diffusion_steps=args.diffusion_steps,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        classifier_model_dim=args.hidden_dim,
        classifier_layers=max(args.num_layers, 2),
        classifier_heads=args.num_heads,
        window=args.window,
        use_torch_geometric=args.use_torch_geometric,
    ).to(device)

    stage3.decoder.load_state_dict(stage1_model.decoder.state_dict(), strict=False)
    stage3.denoiser.load_state_dict(stage1_model.denoiser.state_dict(), strict=False)

    checkpoint_path = Path(args.stage3_ckpt)
    if checkpoint_path.exists():
        load_checkpoint(str(checkpoint_path), stage3.denoiser, strict=False, map_location=device)
        load_checkpoint(str(checkpoint_path), stage3.decoder, strict=False, map_location=device)
        load_checkpoint(str(checkpoint_path), stage3.classifier, strict=False, map_location=device)

    stage3.eval()
    return stage3


def main() -> None:
    """Run conditional sampling and print resulting tensor shapes."""
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    stage1_model = _load_stage1_model(args, device)
    sensor_model = _load_stage2_sensor(args, device)
    stage3_model = _build_stage3_sampler(args, stage1_model, device)

    toy = ToyConfig(
        num_samples=max(args.num_samples, args.batch_size),
        window=args.window,
        joints=args.joints,
        joint_dim=args.joint_dim,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    imu_dataset = IMUDataset(toy=True, toy_config=toy)
    loader = DataLoader(imu_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    accel = batch["A"].to(device)
    gyro = batch["Omega"].to(device)
    batch_size = accel.shape[0]

    adjacency = build_chain_adjacency(args.joints, include_self=True, device=device)

    with torch.no_grad():
        h_joint, _ = sensor_model(accel, gyro)
        z0 = stage3_model.sample_conditional(
            batch_size=batch_size,
            window=args.window,
            adjacency=adjacency,
            device=device,
            h=h_joint,
            steps=args.sampling_steps,
        )
        x_hat = stage3_model.decode(z0, adjacency)

        print(f"conditioning shape: {tuple(h_joint.shape)}")
        print(f"sampled latent shape: {tuple(z0.shape)}")
        print(f"decoded skeleton shape: {tuple(x_hat.shape)}")

        output = {"z0": z0.cpu(), "x_hat": x_hat.cpu(), "conditioning": h_joint.cpu()}

        if args.classify:
            logits = stage3_model.classify(x_hat)
            predictions = torch.argmax(logits, dim=1)
            print(f"logits shape: {tuple(logits.shape)}")
            print(f"predictions shape: {tuple(predictions.shape)}")
            output["logits"] = logits.cpu()
            output["predictions"] = predictions.cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"saved output to: {output_path}")


if __name__ == "__main__":
    main()
