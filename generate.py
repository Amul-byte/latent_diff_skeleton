"""Generation entrypoint for conditional skeleton synthesis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from diffusion_model.dataset import IMUDataset, NormalizationConfig
from diffusion_model.model import Stage3Model
from diffusion_model.model_loader import freeze_module, load_checkpoint, verify_frozen
from diffusion_model.sensor_model import SensorTGNNEncoder
from diffusion_model.skeleton_model import SkeletonStage1Model
from diffusion_model.util import build_chain_adjacency, resolve_device, set_seed

ACCEL_SENSORS = ["meta_hip", "meta_wrist", "phone", "watch"]


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
    parser.add_argument("--paired-file", type=str, default=None, help="Required .pt file with paired tensors")
    parser.add_argument("--sensor-a", type=str, default="meta_hip", choices=ACCEL_SENSORS)
    parser.add_argument("--sensor-b", type=str, default="meta_wrist", choices=ACCEL_SENSORS)
    parser.add_argument("--accel-normalization", type=str, default="zscore", choices=["zscore", "none"])
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


def _load_tensor_payload(path: str) -> Dict[str, torch.Tensor]:
    """Load tensor payload from a `.pt` file."""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Expected dictionary payload in data file")
    return payload


def _resolve_accel_pair_from_payload(
    payload: Dict[str, torch.Tensor],
    sensor_a: str,
    sensor_b: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve two acceleration tensors from a data payload."""
    accel_by_sensor = payload.get("accel_by_sensor")
    if isinstance(accel_by_sensor, dict):
        if sensor_a not in accel_by_sensor:
            raise KeyError(f"Sensor '{sensor_a}' missing from accel_by_sensor")
        if sensor_b not in accel_by_sensor:
            raise KeyError(f"Sensor '{sensor_b}' missing from accel_by_sensor")
        return accel_by_sensor[sensor_a], accel_by_sensor[sensor_b]

    if "A1" in payload and "A2" in payload:
        return payload["A1"], payload["A2"]

    if "A_pair" in payload:
        accel_pair = payload["A_pair"]
        if accel_pair.ndim != 4 or accel_pair.shape[1] != 2 or accel_pair.shape[-1] != 3:
            raise ValueError("A_pair must have shape [N, 2, T, 3]")
        return accel_pair[:, 0], accel_pair[:, 1]

    raise KeyError("Unable to resolve two acceleration streams from paired payload")


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

    if args.paired_file is None or not Path(args.paired_file).exists():
        raise FileNotFoundError("Generation requires --paired-file pointing to a valid .pt file")
    payload = _load_tensor_payload(args.paired_file)
    accel_primary, accel_secondary = _resolve_accel_pair_from_payload(
        payload=payload,
        sensor_a=args.sensor_a,
        sensor_b=args.sensor_b,
    )
    imu_dataset = IMUDataset(
        accel_primary=accel_primary,
        accel_secondary=accel_secondary,
        labels=payload.get("y"),
        sensor_pair=(args.sensor_a, args.sensor_b),
        normalization=NormalizationConfig(mode=args.accel_normalization),
    )
    loader = DataLoader(imu_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    accel_primary = batch["A1"].to(device)
    accel_secondary = batch["A2"].to(device)
    batch_size = accel_primary.shape[0]

    adjacency = build_chain_adjacency(args.joints, include_self=True, device=device)

    with torch.no_grad():
        h_joint, _ = sensor_model(accel_primary, accel_secondary)
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
