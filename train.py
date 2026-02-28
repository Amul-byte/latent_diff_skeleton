#!/usr/bin/env python3
"""
train.py

Unified trainer aligned with the code you shared:

Data (from dataset.py):
- Skeleton CSVs: 32 joints -> 96 cols/frame -> dataset returns X: [B,T,32,3]
- Two accel sensors: A1: [B,T,3], A2: [B,T,3], A_pair: [B,2,T,3]
- Labels: binary by filename (A10-A14 fall => 1 else 0)  [for now]

Models (from your repo):
- Stage 1 (skeleton_model.py): SkeletonStage1Model
    - mode="ae"  : encoder+decoder reconstruct X
    - mode="diff": latent diffusion uncond (encoder -> z0, denoiser predicts noise)
- IMU encoder (imu_encoder.py / sensor_model.py alias): SensorTGNNEncoder (TwoSensorIMUEncoder)
    - produces conditioning h_joint [B,T,J,D] and h_seq [B,T,D]
- Stage 2 (not provided in your files): this train.py implements a simple IMU->latent regressor head
    - learns z_pred ~= z0_target (from frozen Stage-1 encoder)
- Stage 3 (model.py): Stage3Model
    - learns conditional diffusion denoising in latent, decodes, and (optionally) classifies

Splits:
- Subject-wise split by parsing Sxx from filenames (subject_from_filename from dataset.py).
- No window leakage across subjects.

Outputs:
- checkpoints/{run_name}/stage{1,2,3}/...
- Saves best checkpoints by validation loss.

NOTE:
- This is single-process (one GPU). You can wrap with torchrun later if you want DDP,
  but this will be stable and aligned with your modules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_model.dataset import (
    SmartFallPairedSlidingWindowDataset,  # you pasted this earlier
    read_csv_files,
    subject_from_filename,
)
from diffusion_model.sensor_model import SensorTGNNEncoder  # alias to TwoSensorIMUEncoder
from diffusion_model.model import Stage3Model
from diffusion_model.model_loader import freeze_module, load_checkpoint
from diffusion_model.skeleton_model import SkeletonStage1Model
from diffusion_model.util import (
    ensure_dir,
    get_logger,
    resolve_device,
    set_seed,
    build_smartfall_bone_adjacency,
    count_trainable_parameters,
)

# ---------------------------
# Small helpers
# ---------------------------

def save_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def split_subjects(
    all_files: List[str],
    train_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Return (train_subjects, val_subjects). Subject-wise split."""
    subjects = sorted({subject_from_filename(f) for f in all_files})
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(subjects), generator=g).tolist()
    subjects = [subjects[i] for i in perm]

    n_train = max(1, int(round(train_ratio * len(subjects))))
    train_subj = sorted(subjects[:n_train])
    val_subj = sorted(subjects[n_train:])
    if len(val_subj) == 0:
        # if too few subjects, keep at least 1 in val
        train_subj = sorted(subjects[:-1])
        val_subj = sorted(subjects[-1:])
    return train_subj, val_subj


def make_loaders(
    skeleton_folder: str,
    right_hip_folder: str,
    left_wrist_folder: str,
    window: int,
    stride: int,
    train_ratio: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    imu_norm: str,
    imu_eps: float,
    imu_clip: Optional[float],
    fall_activities: Tuple[int, ...],
    drop_misaligned: bool = True,
    align_mode: str = "strict",
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Creates subject-wise train/val datasets and returns (train_loader, val_loader, meta).
    Ensures val uses train IMU stats (if zscore).
    """
    skeleton_data = read_csv_files(skeleton_folder)
    sensor1_data = read_csv_files(right_hip_folder)
    sensor2_data = read_csv_files(left_wrist_folder)

    # Build a temporary dataset (to compute common files list through its init logic)
    tmp_ds = SmartFallPairedSlidingWindowDataset(
        skeleton_data=skeleton_data,
        sensor1_data=sensor1_data,
        sensor2_data=sensor2_data,
        window_size=window,
        stride=stride,
        fall_activities=fall_activities,
        drop_misaligned=drop_misaligned,
        align_mode=align_mode,
        allowed_files=None,
        allowed_subjects=None,
        imu_normalization=imu_norm,
        imu_stats=None,
        imu_eps=imu_eps,
        imu_clip=imu_clip,
        sensor_names=("right_hip", "left_wrist"),
        sensor_roots=(right_hip_folder, left_wrist_folder),
        strict_sensor_identity=True,
    )
    all_files = tmp_ds.files
    train_subj, val_subj = split_subjects(all_files, train_ratio=train_ratio, seed=seed)

    # Train dataset computes stats (if zscore and imu_stats None)
    train_ds = SmartFallPairedSlidingWindowDataset(
        skeleton_data=skeleton_data,
        sensor1_data=sensor1_data,
        sensor2_data=sensor2_data,
        window_size=window,
        stride=stride,
        fall_activities=fall_activities,
        drop_misaligned=drop_misaligned,
        align_mode=align_mode,
        allowed_subjects=train_subj,
        imu_normalization=imu_norm,
        imu_stats=None,  # compute from train only
        imu_eps=imu_eps,
        imu_clip=imu_clip,
        sensor_names=("right_hip", "left_wrist"),
        sensor_roots=(right_hip_folder, left_wrist_folder),
        strict_sensor_identity=True,
    )
    train_stats = train_ds.get_normalization_stats()

    # Val dataset reuses train stats to avoid leakage
    val_ds = SmartFallPairedSlidingWindowDataset(
        skeleton_data=skeleton_data,
        sensor1_data=sensor1_data,
        sensor2_data=sensor2_data,
        window_size=window,
        stride=stride,
        fall_activities=fall_activities,
        drop_misaligned=drop_misaligned,
        align_mode=align_mode,
        allowed_subjects=val_subj,
        imu_normalization=imu_norm,
        imu_stats=train_stats if imu_norm == "zscore" else None,
        imu_eps=imu_eps,
        imu_clip=imu_clip,
        sensor_names=("right_hip", "left_wrist"),
        sensor_roots=(right_hip_folder, left_wrist_folder),
        strict_sensor_identity=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    meta = {
        "subjects_train": train_subj,
        "subjects_val": val_subj,
        "files_total_common": len(all_files),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "imu_normalization": imu_norm,
        "align_mode": align_mode,
        "imu_stats_train": None if train_stats is None else {
            k: {"mean": v["mean"].tolist(), "std": v["std"].tolist()} for k, v in train_stats.items()
        },
    }
    return train_loader, val_loader, meta


# ---------------------------
# Train/eval loops
# ---------------------------

@torch.no_grad()
def eval_stage1(model: SkeletonStage1Model, loader: DataLoader, adjacency: torch.Tensor, device: torch.device, mode: str) -> float:
    model.eval()
    losses = []
    for batch in loader:
        X = batch["X"].to(device, non_blocking=True)  # [B,T,J,3]
        out = model.forward_stage1(X, adjacency=adjacency, mode=mode)
        losses.append(out["loss"].detach().float().item())
    return float(sum(losses) / max(1, len(losses)))


def train_stage1(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: Path,
    device: torch.device,
    logger,
) -> None:
    stage_dir = run_dir / "stage1"
    ensure_dir(stage_dir)

    if args.stage1_mode != "diff":
        raise ValueError(
            "EXACT proposal mode requires Stage 1 diffusion-only. "
            "Use --stage1_mode diff."
        )

    adjacency = build_smartfall_bone_adjacency(args.num_joints, include_self=True, device=device)

    model = SkeletonStage1Model(
        joint_dim=3,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.graph_layers,
        num_heads=args.graph_heads,
        diffusion_steps=args.diffusion_steps,
    ).to(device)

    logger.info(f"[Stage1] Trainable params: {count_trainable_parameters(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = math.inf
    mode = args.stage1_mode  # diffusion-only for exact proposal

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Stage1/{mode} Epoch {epoch}/{args.epochs}")
        running = 0.0

        for batch in pbar:
            X = batch["X"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            out = model.forward_stage1(X, adjacency=adjacency, mode=mode)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            running += float(loss.detach().cpu().item())
            pbar.set_postfix(loss=f"{running/max(1,pbar.n):.4f}")

        val_loss = eval_stage1(model, val_loader, adjacency, device, mode=mode)
        logger.info(f"[Stage1/{mode}] epoch={epoch} train_loss={running/max(1,len(train_loader)):.4f} val_loss={val_loss:.4f}")

        # Save last
        torch.save({"model": model.state_dict(), "args": vars(args)}, stage_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "args": vars(args)}, stage_dir / "best.pt")
            logger.info(f"[Stage1/{mode}] Saved best.pt (val_loss={best_val:.4f})")


def _resample_to_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resample [B,T,3] IMU stream to target time length (linear interpolation)."""
    if x.shape[1] == target_len:
        return x
    x_bct = x.transpose(1, 2)  # [B,3,T]
    x_rs = torch.nn.functional.interpolate(x_bct, size=target_len, mode="linear", align_corners=False)
    return x_rs.transpose(1, 2)  # [B,target_len,3]


@torch.no_grad()
def eval_stage2(
    imu_encoder: SensorTGNNEncoder,
    stage1: SkeletonStage1Model,
    loader: DataLoader,
    adjacency: torch.Tensor,
    device: torch.device,
) -> float:
    imu_encoder.eval()
    stage1.eval()
    losses = []
    for batch in loader:
        X = batch["X"].to(device, non_blocking=True)
        A1 = batch["A1"].to(device, non_blocking=True)
        A2 = batch["A2"].to(device, non_blocking=True)
        A1 = _resample_to_len(A1, X.shape[1])
        A2 = _resample_to_len(A2, X.shape[1])

        z0 = stage1.encoder(X, adjacency)  # frozen
        h_joint, _h_seq = imu_encoder(A1, A2)
        assert h_joint.shape == z0.shape, (
            f"Shape mismatch: h_joint={tuple(h_joint.shape)} vs z0={tuple(z0.shape)}. "
            "SensorTGNNEncoder must output h_joint with shape [B,T,J,D]."
        )
        loss = F.mse_loss(h_joint, z0)
        losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(1, len(losses)))


def train_stage2(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: Path,
    device: torch.device,
    logger,
) -> None:
    """
    Stage 2 in your proposal: IMU -> latent alignment (regression).
    We freeze Stage-1 encoder and learn only IMUEncoder:
      h_joint ~= z0_target
    """
    stage_dir = run_dir / "stage2"
    ensure_dir(stage_dir)

    adjacency = build_smartfall_bone_adjacency(args.num_joints, include_self=True, device=device)

    # Load Stage1 checkpoint and freeze
    stage1 = SkeletonStage1Model(
        joint_dim=3,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.graph_layers,
        num_heads=args.graph_heads,
        diffusion_steps=args.diffusion_steps,
    ).to(device)
    ckpt = str(Path(args.stage1_ckpt).expanduser())
    load_checkpoint(ckpt, stage1, strict=False, map_location=device)
    freeze_module(stage1)  # stage1 encoder/decoder/denoiser frozen

    imu_encoder = SensorTGNNEncoder(
        latent_dim=args.latent_dim,
        num_joints=args.num_joints,
        hidden_dim=args.imu_hidden_dim,
    ).to(device)

    params = list(imu_encoder.parameters())
    logger.info(f"[Stage2] Trainable params: {sum(p.numel() for p in params if p.requires_grad):,}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        imu_encoder.train()

        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch}/{args.epochs}")
        running = 0.0

        for batch in pbar:
            X = batch["X"].to(device, non_blocking=True)
            A1 = batch["A1"].to(device, non_blocking=True)
            A2 = batch["A2"].to(device, non_blocking=True)
            A1 = _resample_to_len(A1, X.shape[1])
            A2 = _resample_to_len(A2, X.shape[1])

            with torch.no_grad():
                z0 = stage1.encoder(X, adjacency)

            h_joint, _h_seq = imu_encoder(A1, A2)
            assert h_joint.shape == z0.shape, (
                f"Shape mismatch: h_joint={tuple(h_joint.shape)} vs z0={tuple(z0.shape)}. "
                "SensorTGNNEncoder must output h_joint with shape [B,T,J,D]."
            )
            loss = F.mse_loss(h_joint, z0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()

            running += float(loss.detach().cpu().item())
            pbar.set_postfix(loss=f"{running/max(1,pbar.n):.4f}")

        val_loss = eval_stage2(imu_encoder, stage1, val_loader, adjacency, device)
        logger.info(f"[Stage2] epoch={epoch} train_loss={running/max(1,len(train_loader)):.4f} val_loss={val_loss:.4f}")

        torch.save(
            {"imu_encoder": imu_encoder.state_dict(), "args": vars(args)},
            stage_dir / "last.pt",
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"imu_encoder": imu_encoder.state_dict(), "args": vars(args)},
                stage_dir / "best.pt",
            )
            logger.info(f"[Stage2] Saved best.pt (val_loss={best_val:.4f})")


@torch.no_grad()
def eval_stage3(
    stage3: Stage3Model,
    stage1: SkeletonStage1Model,
    imu_encoder: Optional[SensorTGNNEncoder],
    loader: DataLoader,
    adjacency: torch.Tensor,
    device: torch.device,
    use_imu: bool,
) -> Dict[str, float]:
    stage3.eval()
    stage1.eval()
    if imu_encoder is not None:
        imu_encoder.eval()

    losses = []
    dlosses = []
    closses = []
    acc_num = 0
    acc_den = 0

    for batch in loader:
        X = batch["X"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with torch.no_grad():
            z0_target = stage1.encoder(X, adjacency)

        h = None
        if use_imu:
            A1 = batch["A1"].to(device, non_blocking=True)
            A2 = batch["A2"].to(device, non_blocking=True)
            A1 = _resample_to_len(A1, X.shape[1])
            A2 = _resample_to_len(A2, X.shape[1])
            h_joint, h_seq = imu_encoder(A1, A2)  # type: ignore[union-attr]
            h = h_joint if stage3_cond_mode(stage3) == "joint" else h_seq

        out = stage3(
            z0_target=z0_target,
            adjacency=adjacency,
            h=h,
            labels=y,
            diffusion_weight=1.0
        )

        losses.append(float(out["loss"].detach().cpu().item()))
        dlosses.append(float(out["diffusion_loss"].detach().cpu().item()))
        closses.append(float(out["classification_loss"].detach().cpu().item()))

        # accuracy (only if labels are valid)
        if torch.all(y >= 0):
            pred = out["logits"].argmax(dim=1)
            acc_num += int((pred == y).sum().item())
            acc_den += int(y.numel())

    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "diffusion_loss": float(sum(dlosses) / max(1, len(dlosses))),
        "cls_loss": float(sum(closses) / max(1, len(closses))),
        "acc": float(acc_num / max(1, acc_den)),
    }


def stage3_cond_mode(stage3: Stage3Model) -> str:
    """
    Your Stage3Model supports h=[B,T,D] or h=[B,T,J,D]. We choose which by args.
    This helper exists so eval can match train-time mode.
    """
    # not stored inside Stage3Model; we infer by how user trains
    return "joint"  # default, overridden by args.stage3_cond in train_stage3


def train_stage3(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: Path,
    device: torch.device,
    logger,
) -> None:
    stage_dir = run_dir / "stage3"
    ensure_dir(stage_dir)

    adjacency = build_smartfall_bone_adjacency(args.num_joints, include_self=True, device=device)

    # Load + freeze Stage1
    stage1 = SkeletonStage1Model(
        joint_dim=3,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.graph_layers,
        num_heads=args.graph_heads,
        diffusion_steps=args.diffusion_steps,
    ).to(device)
    load_checkpoint(str(Path(args.stage1_ckpt).expanduser()), stage1, strict=False, map_location=device)
    freeze_module(stage1)

    # Optional IMU encoder (conditioning)
    imu_encoder = None
    use_imu = args.stage3_use_imu

    if use_imu:
        imu_encoder = SensorTGNNEncoder(
            latent_dim=args.latent_dim,
            num_joints=args.num_joints,
            hidden_dim=args.imu_hidden_dim,
        ).to(device)

        # optionally load stage2 weights if provided
        if args.stage2_ckpt:
            payload, _, _ = load_checkpoint(str(Path(args.stage2_ckpt).expanduser()), imu_encoder, strict=False, map_location=device)
            # If your stage2 ckpt was saved as {"imu_encoder": ...}, we handle that:
            if isinstance(payload, dict) and "imu_encoder" in payload and isinstance(payload["imu_encoder"], dict):
                imu_encoder.load_state_dict(payload["imu_encoder"], strict=False)

    # Stage3 model
    stage3 = Stage3Model(
        latent_dim=args.latent_dim,
        joint_dim=3,
        num_joints=args.num_joints,
        num_classes=args.num_classes,  # for binary now => 2
        hidden_dim=args.hidden_dim,
        diffusion_steps=args.diffusion_steps,
        graph_layers=args.graph_layers,
        graph_heads=args.graph_heads,
        classifier_dim=args.classifier_dim,
        classifier_layers=args.classifier_layers,
        classifier_heads=args.classifier_heads,
        window=args.window,
    ).to(device)

    # What trains in stage3?
    # - Always: stage3 parameters
    # - If use_imu and not frozen: imu_encoder also trains
    params = list(stage3.parameters())
    if imu_encoder is not None and not args.freeze_imu_in_stage3:
        params += list(imu_encoder.parameters())

    logger.info(f"[Stage3] Trainable params: {sum(p.numel() for p in params if p.requires_grad):,}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    best_val = math.inf
    cond_mode = args.stage3_cond  # "joint" or "seq"

    for epoch in range(1, args.epochs + 1):
        stage3.train()
        if imu_encoder is not None and not args.freeze_imu_in_stage3:
            imu_encoder.train()

        pbar = tqdm(train_loader, desc=f"Stage3({cond_mode}) Epoch {epoch}/{args.epochs}")
        running = 0.0

        for batch in pbar:
            X = batch["X"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with torch.no_grad():
                z0_target = stage1.encoder(X, adjacency)

            h = None
            if use_imu:
                A1 = batch["A1"].to(device, non_blocking=True)
                A2 = batch["A2"].to(device, non_blocking=True)
                A1 = _resample_to_len(A1, X.shape[1])
                A2 = _resample_to_len(A2, X.shape[1])
                h_joint, h_seq = imu_encoder(A1, A2)  # type: ignore[misc]
                h = h_joint if cond_mode == "joint" else h_seq

            out = stage3(
                z0_target=z0_target,
                adjacency=adjacency,
                h=h,
                labels=y,
                diffusion_weight=1.0,
                classifier_weight=args.cls_weight,
            )

            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()

            running += float(loss.detach().cpu().item())
            pbar.set_postfix(loss=f"{running/max(1,pbar.n):.4f}")

        metrics = eval_stage3(stage3, stage1, imu_encoder, val_loader, adjacency, device, use_imu=use_imu)
        logger.info(
            f"[Stage3] epoch={epoch} train_loss={running/max(1,len(train_loader)):.4f} "
            f"val_loss={metrics['loss']:.4f} val_dloss={metrics['diffusion_loss']:.4f} "
            f"val_closs={metrics['cls_loss']:.4f} val_acc={metrics['acc']*100:.2f}%"
        )

        # Save last
        save_obj = {"stage3": stage3.state_dict(), "args": vars(args)}
        if imu_encoder is not None:
            save_obj["imu_encoder"] = imu_encoder.state_dict()
        torch.save(save_obj, stage_dir / "last.pt")

        if metrics["loss"] < best_val:
            best_val = metrics["loss"]
            torch.save(save_obj, stage_dir / "best.pt")
            logger.info(f"[Stage3] Saved best.pt (val_loss={best_val:.4f})")


# ---------------------------
# Main
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Joint-Aware Latent Diffusion Trainer (Stage1/2/3)")

    # I/O
    p.add_argument("--run_name", type=str, default="run1")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--log_file", type=str, default=None)

    # Data
    p.add_argument("--skeleton_folder", type=str, required=True)
    p.add_argument("--right_hip_folder", type=str, required=True)
    p.add_argument("--left_wrist_folder", type=str, required=True)

    p.add_argument("--window", type=int, default=60)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--drop_misaligned", action="store_true", help="Skip trials where skel/sensor lengths mismatch")
    p.set_defaults(drop_misaligned=False)
    p.add_argument(
        "--align_mode",
        type=str,
        default="truncate_min",
        choices=["strict", "truncate_min"],
        help="strict: require equal lengths; truncate_min: build windows from overlapping frames",
    )

    # Labels (binary for now)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--fall_activities", type=int, nargs="+", default=[10, 11, 12, 13, 14])

    # IMU normalization
    p.add_argument("--imu_norm", type=str, default="zscore", choices=["zscore", "none"])
    p.add_argument("--imu_eps", type=float, default=1e-6)
    p.add_argument("--imu_clip", type=float, default=6.0)

    # Train
    p.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)

    # Architecture
    p.add_argument("--num_joints", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--graph_layers", type=int, default=3)
    p.add_argument("--graph_heads", type=int, default=8)
    p.add_argument("--diffusion_steps", type=int, default=500)

    # Stage 1 specific (EXACT proposal: diffusion-only)
    p.add_argument("--stage1_mode", type=str, default="diff", choices=["diff"])

    # Stage 2 specific
    p.add_argument("--stage1_ckpt", type=str, default=None, help="Required for stage 2/3")
    p.add_argument("--imu_hidden_dim", type=int, default=256)

    # Stage 3 specific
    p.add_argument("--stage3_use_imu", action="store_true", help="Use IMU conditioning in stage 3")
    p.set_defaults(stage3_use_imu=True)
    p.add_argument("--stage3_cond", type=str, default="joint", choices=["joint", "seq"])
    p.add_argument("--cls_weight", type=float, default=1.0)
    p.add_argument("--freeze_imu_in_stage3", action="store_true")
    p.set_defaults(freeze_imu_in_stage3=False)
    p.add_argument("--stage2_ckpt", type=str, default=None, help="Optional: load IMU encoder weights for stage 3")

    # Classifier
    p.add_argument("--classifier_dim", type=int, default=256)
    p.add_argument("--classifier_layers", type=int, default=4)
    p.add_argument("--classifier_heads", type=int, default=8)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = Path(args.out_dir) / args.run_name
    ensure_dir(run_dir)

    logger = get_logger("train", log_file=args.log_file)
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Device: {device}")

    # Save args
    save_json(run_dir / "args.json", vars(args))

    # Data loaders (subject-wise split)
    train_loader, val_loader, meta = make_loaders(
        skeleton_folder=args.skeleton_folder,
        right_hip_folder=args.right_hip_folder,
        left_wrist_folder=args.left_wrist_folder,
        window=args.window,
        stride=args.stride,
        train_ratio=args.train_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        imu_norm=args.imu_norm,
        imu_eps=args.imu_eps,
        imu_clip=args.imu_clip,
        fall_activities=tuple(args.fall_activities),
        drop_misaligned=args.drop_misaligned,
        align_mode=args.align_mode,
    )
    save_json(run_dir / "data_split.json", meta)
    logger.info(f"Data split saved: {run_dir/'data_split.json'}")

    # Train selected stage
    if args.stage == 1:
        train_stage1(args, train_loader, val_loader, run_dir, device, logger)
        return

    if args.stage in (2, 3) and not args.stage1_ckpt:
        raise ValueError("--stage1_ckpt is required for stage 2 or stage 3")

    if args.stage == 2:
        train_stage2(args, train_loader, val_loader, run_dir, device, logger)
        return

    if args.stage == 3:
        train_stage3(args, train_loader, val_loader, run_dir, device, logger)
        return


if __name__ == "__main__":
    main()
