"""Training entrypoint for 3-stage Joint-Aware Latent Diffusion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion_model.dataset import NormalizationConfig, PairedDataset, SkeletonDataset
from diffusion_model.model import Stage3Model
from diffusion_model.model_loader import freeze_module, load_checkpoint, verify_frozen
from diffusion_model.sensor_model import SensorTGNNEncoder
from diffusion_model.skeleton_model import SkeletonStage1Model
from diffusion_model.util import build_chain_adjacency, ensure_dir, get_logger, resolve_device, set_seed

ACCEL_SENSORS = ["meta_hip", "meta_wrist", "phone", "watch"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for stage-specific training.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train Joint-Aware Latent Diffusion model")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3], help="Training stage: 1, 2, or 3")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--stage1-phase",
        type=str,
        default="ae",
        choices=["ae", "diff", "diffusion_uncond"],
        help="Stage-1 phase: AE pretrain or unconditional diffusion pretrain",
    )

    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--window", type=int, default=90)
    parser.add_argument("--joints", type=int, default=32)
    parser.add_argument("--joint-dim", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=14)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)

    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--paired-file", type=str, default=None, help="Optional .pt file with paired tensors")
    parser.add_argument("--skeleton-file", type=str, default=None, help="Optional .pt file with skeleton tensors")
    parser.add_argument(
        "--sensor-a",
        type=str,
        default="meta_hip",
        choices=ACCEL_SENSORS,
        help="Primary acceleration sensor name",
    )
    parser.add_argument(
        "--sensor-b",
        type=str,
        default="meta_wrist",
        choices=ACCEL_SENSORS,
        help="Secondary acceleration sensor name",
    )
    parser.add_argument(
        "--accel-normalization",
        type=str,
        default="zscore",
        choices=["zscore", "none"],
        help="Acceleration normalization mode",
    )

    parser.add_argument("--stage1-ckpt", type=str, default="checkpoints/stage1.pt")
    parser.add_argument("--stage2-ckpt", type=str, default="checkpoints/stage2.pt")
    parser.add_argument("--stage3-ckpt", type=str, default="checkpoints/stage3.pt")
    parser.add_argument("--strict-load", action="store_true", help="Use strict checkpoint loading")

    parser.add_argument("--recon-weight", type=float, default=0.1)
    parser.add_argument("--diffusion-weight", type=float, default=1.0)
    parser.add_argument("--classifier-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-torch-geometric", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def _load_tensor_payload(path: str) -> Dict[str, torch.Tensor]:
    """Load tensor payload from a `.pt` file.

    Args:
        path: File path.

    Returns:
        Dictionary payload.
    """
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Expected dictionary payload in data file")
    return payload


def _resolve_accel_pair_from_payload(
    payload: Dict[str, torch.Tensor],
    sensor_a: str,
    sensor_b: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve two acceleration tensors from a data payload.

    Args:
        payload: Dictionary loaded from paired data file.
        sensor_a: Primary sensor name.
        sensor_b: Secondary sensor name.

    Returns:
        Tuple of acceleration tensors ``(A1, A2)``, each with shape ``[N, T, 3]``.
    """
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


def build_stage1_loader(args: argparse.Namespace) -> DataLoader:
    """Create Stage 1 dataloader using real skeleton data.

    Args:
        args: Parsed CLI arguments.

    Returns:
        DataLoader for Stage 1.
    """
    if args.skeleton_file is None or not Path(args.skeleton_file).exists():
        raise FileNotFoundError("Stage 1 requires --skeleton-file pointing to an existing .pt file")

    payload = _load_tensor_payload(args.skeleton_file)
    dataset = SkeletonDataset(
        skeleton=payload.get("X"),
        labels=payload.get("y"),
    )

    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)


def build_paired_loader(args: argparse.Namespace) -> DataLoader:
    """Create paired dataloader for Stage 2/3 using real data.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Paired DataLoader.
    """
    normalization = NormalizationConfig(mode=args.accel_normalization)
    sensor_pair = (args.sensor_a, args.sensor_b)

    if args.paired_file is None or not Path(args.paired_file).exists():
        raise FileNotFoundError("Stage 2/3 requires --paired-file pointing to an existing .pt file")

    payload = _load_tensor_payload(args.paired_file)
    accel_primary, accel_secondary = _resolve_accel_pair_from_payload(
        payload=payload,
        sensor_a=args.sensor_a,
        sensor_b=args.sensor_b,
    )
    dataset = PairedDataset(
        skeleton=payload.get("X"),
        accel_primary=accel_primary,
        accel_secondary=accel_secondary,
        labels=payload.get("y"),
        sensor_pair=sensor_pair,
        normalization=normalization,
    )

    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)


def train_stage1(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> None:
    """Train Stage 1 skeleton latent diffusion.

    Args:
        args: Parsed CLI arguments.
        device: Training device.
        logger: Logger instance.
    """
    loader = build_stage1_loader(args)
    adjacency = build_chain_adjacency(args.joints, include_self=True, device=device)

    model = SkeletonStage1Model(
        joint_dim=args.joint_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        diffusion_steps=args.diffusion_steps,
        use_torch_geometric=args.use_torch_geometric,
    ).to(device)

    stage1_mode = "diff" if args.stage1_phase == "diffusion_uncond" else args.stage1_phase

    if stage1_mode == "ae":
        freeze_module(model.denoiser)
        verify_frozen(model.denoiser)
    elif stage1_mode == "diff":
        freeze_module(model.encoder)
        freeze_module(model.decoder)
        verify_frozen(model.encoder)
        verify_frozen(model.decoder)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters found for stage1 mode '{stage1_mode}'")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            x = batch["X"].to(device)
            out = model.forward_stage1(
                x=x,
                adjacency=adjacency,
                mode=stage1_mode,
                recon_weight=args.recon_weight,
                diffusion_weight=args.diffusion_weight,
            )
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            global_step += 1
            if global_step % args.log_every == 0:
                logger.info(
                    "stage=1 epoch=%d step=%d loss=%.6f recon=%.6f diff=%.6f",
                    epoch,
                    global_step,
                    float(out["loss"].item()),
                    float(out["reconstruction_loss"].item()),
                    float(out["diffusion_loss"].item()),
                )
            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    ensure_dir(Path(args.stage1_ckpt).parent)
    torch.save(
        {
            "stage": 1,
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
            "denoiser": model.denoiser.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        args.stage1_ckpt,
    )
    logger.info("Saved Stage 1 checkpoint to %s", args.stage1_ckpt)


def _maybe_load_stage1_for_freeze(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> SkeletonStage1Model:
    """Build Stage 1 model and optionally load checkpoint.

    Args:
        args: Parsed CLI arguments.
        device: Target device.
        logger: Logger instance.

    Returns:
        Stage 1 model object.
    """
    stage1_model = SkeletonStage1Model(
        joint_dim=args.joint_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        diffusion_steps=args.diffusion_steps,
        use_torch_geometric=args.use_torch_geometric,
    ).to(device)

    if Path(args.stage1_ckpt).exists():
        load_checkpoint(args.stage1_ckpt, stage1_model.encoder, strict=args.strict_load, map_location=device)
        load_checkpoint(args.stage1_ckpt, stage1_model.decoder, strict=False, map_location=device)
        load_checkpoint(args.stage1_ckpt, stage1_model.denoiser, strict=False, map_location=device)
        logger.info("Loaded Stage 1 checkpoint from %s", args.stage1_ckpt)
    else:
        logger.warning("Stage 1 checkpoint not found at %s, using random initialization", args.stage1_ckpt)

    freeze_module(stage1_model.encoder)
    assert verify_frozen(stage1_model.encoder)
    return stage1_model


def train_stage2(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> None:
    """Train Stage 2 IMU-to-latent regression with frozen Stage 1 encoder.

    Args:
        args: Parsed CLI arguments.
        device: Training device.
        logger: Logger instance.
    """
    loader = build_paired_loader(args)
    adjacency = build_chain_adjacency(args.joints, include_self=True, device=device)
    stage1_model = _maybe_load_stage1_for_freeze(args, device, logger)

    sensor_model = SensorTGNNEncoder(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(sensor_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    sensor_model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            x = batch["X"].to(device)
            accel_primary = batch["A1"].to(device)
            accel_secondary = batch["A2"].to(device)

            with torch.no_grad():
                z0_target = stage1_model.encoder(x, adjacency)

            h_joint, _ = sensor_model(accel_primary, accel_secondary)
            if h_joint.shape != z0_target.shape:
                raise AssertionError("Stage 2 target and prediction shapes must match")

            loss = F.mse_loss(h_joint, z0_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            for parameter in stage1_model.encoder.parameters():
                if parameter.grad is not None:
                    raise AssertionError("Frozen Stage 1 encoder received gradients during Stage 2")

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(sensor_model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            global_step += 1
            if global_step % args.log_every == 0:
                logger.info("stage=2 epoch=%d step=%d reg_loss=%.6f", epoch, global_step, float(loss.item()))
            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    ensure_dir(Path(args.stage2_ckpt).parent)
    torch.save(
        {
            "stage": 2,
            "sensor_encoder": sensor_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        args.stage2_ckpt,
    )
    logger.info("Saved Stage 2 checkpoint to %s", args.stage2_ckpt)


def _load_stage2_sensor(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> SensorTGNNEncoder:
    """Build Stage 2 sensor encoder and optionally load checkpoint.

    Args:
        args: Parsed CLI arguments.
        device: Target device.
        logger: Logger instance.

    Returns:
        Sensor encoder model.
    """
    sensor_model = SensorTGNNEncoder(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        hidden_dim=args.hidden_dim,
    ).to(device)

    if Path(args.stage2_ckpt).exists():
        load_checkpoint(args.stage2_ckpt, sensor_model, strict=args.strict_load, map_location=device)
        logger.info("Loaded Stage 2 checkpoint from %s", args.stage2_ckpt)
    else:
        logger.warning("Stage 2 checkpoint not found at %s, using random initialization", args.stage2_ckpt)

    freeze_module(sensor_model)
    assert verify_frozen(sensor_model)
    return sensor_model


def train_stage3(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> None:
    """Train Stage 3 conditional diffusion and classifier.

    Args:
        args: Parsed CLI arguments.
        device: Training device.
        logger: Logger instance.
    """
    loader = build_paired_loader(args)
    adjacency = build_chain_adjacency(args.joints, include_self=True, device=device)

    stage1_model = _maybe_load_stage1_for_freeze(args, device, logger)
    sensor_model = _load_stage2_sensor(args, device, logger)

    stage3_model = Stage3Model(
        latent_dim=args.latent_dim,
        joint_dim=args.joint_dim,
        num_joints=args.joints,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        diffusion_steps=args.diffusion_steps,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        classifier_model_dim=args.hidden_dim,
        classifier_layers=max(2, args.num_layers),
        classifier_heads=args.num_heads,
        window=args.window,
        use_torch_geometric=args.use_torch_geometric,
    ).to(device)

    stage3_model.decoder.load_state_dict(stage1_model.decoder.state_dict(), strict=False)
    stage3_model.denoiser.load_state_dict(stage1_model.denoiser.state_dict(), strict=False)

    optimizer = torch.optim.AdamW(stage3_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    stage3_model.train()
    global_step = 0
    unconditional_checked = False

    for epoch in range(args.epochs):
        for batch in loader:
            x = batch["X"].to(device)
            accel_primary = batch["A1"].to(device)
            accel_secondary = batch["A2"].to(device)
            labels = batch["y"].to(device)

            with torch.no_grad():
                z0_target = stage1_model.encoder(x, adjacency)
                h_joint, _ = sensor_model(accel_primary, accel_secondary)

            if not unconditional_checked:
                with torch.no_grad():
                    uncond_out = stage3_model.forward_stage3(
                        z0_target=z0_target,
                        adjacency=adjacency,
                        h=None,
                        labels=None,
                        diffusion_weight=args.diffusion_weight,
                        classifier_weight=args.classifier_weight,
                    )
                logger.info("stage=3 unconditional conditioning check passed diff=%.6f", float(uncond_out["diffusion_loss"]))
                unconditional_checked = True

            out = stage3_model.forward_stage3(
                z0_target=z0_target,
                adjacency=adjacency,
                h=h_joint,
                labels=labels,
                diffusion_weight=args.diffusion_weight,
                classifier_weight=args.classifier_weight,
            )
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            for parameter in stage1_model.encoder.parameters():
                if parameter.grad is not None:
                    raise AssertionError("Frozen Stage 1 encoder received gradients during Stage 3")
            for parameter in sensor_model.parameters():
                if parameter.grad is not None:
                    raise AssertionError("Frozen Stage 2 sensor encoder received gradients during Stage 3")

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(stage3_model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            global_step += 1
            if global_step % args.log_every == 0:
                logger.info(
                    "stage=3 epoch=%d step=%d loss=%.6f diff=%.6f cls=%.6f",
                    epoch,
                    global_step,
                    float(out["loss"].item()),
                    float(out["diffusion_loss"].item()),
                    float(out["classification_loss"].item()),
                )
            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    ensure_dir(Path(args.stage3_ckpt).parent)
    torch.save(
        {
            "stage": 3,
            "denoiser": stage3_model.denoiser.state_dict(),
            "decoder": stage3_model.decoder.state_dict(),
            "classifier": stage3_model.classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        args.stage3_ckpt,
    )
    logger.info("Saved Stage 3 checkpoint to %s", args.stage3_ckpt)


def main() -> None:
    """CLI entrypoint for training all stages."""
    args = parse_args()
    ensure_dir(args.output_dir)
    logger = get_logger("train", str(Path(args.output_dir) / "train.log"))

    set_seed(args.seed)
    device = resolve_device(args.device)
    logger.info("Running stage %d on device=%s", args.stage, device)

    if args.stage == 1:
        train_stage1(args, device, logger)
    elif args.stage == 2:
        train_stage2(args, device, logger)
    elif args.stage == 3:
        train_stage3(args, device, logger)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
