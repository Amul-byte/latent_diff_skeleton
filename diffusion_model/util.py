"""
util.py

This file contains small helper functions used everywhere in the project.

It helps with:
1) Reproducibility (same random results every run)
2) Logging (printing messages nicely)
3) Tensor shape checks (catch bugs early)
4) Simple adjacency creation (for skeleton graph)
5) Device selection (cpu vs gpu)
6) Creating folders (so saving doesn't crash)

Everything is written in a very simple + readable way.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch


# ------------------------------------------------------------
# 1) Reproducibility: set all random seeds
# ------------------------------------------------------------
def set_seed(seed: int) -> None:
    """
    Make randomness predictable.

    If you use the same seed number, you will get the same random results
    (as much as possible).

    This sets the seed for:
      - Python random
      - NumPy random
      - PyTorch random (CPU)
      - PyTorch random (GPU)
    """
    random.seed(seed)              # Python random
    np.random.seed(seed)           # NumPy random
    torch.manual_seed(seed)        # PyTorch CPU random
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random (if you have CUDA)


# ------------------------------------------------------------
# 2) Logging: print nice messages (and optionally save to a file)
# ------------------------------------------------------------
def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger.

    Logger = a tool that prints messages like:
      2026-02-25 12:00:00 | INFO | train | Starting training...

    Args:
      name: name of the logger (example: "train", "eval")
      log_file: optional path to save logs into a file too
      level: how important messages should be printed (INFO, WARNING, ERROR)

    Returns:
      logger object
    """
    logger = logging.getLogger(name)   # create/find logger by name
    logger.setLevel(level)             # set importance level
    logger.propagate = False           # stop duplicate printing from root loggers

    # Only add handlers once (so we don't print same thing 10 times)
    if not logger.handlers:
        # Format: time | level | name | message
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        # Print to terminal
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        # Optionally also write logs to a file
        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)  # make folder if needed

            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# ------------------------------------------------------------
# 3) Tensor checks: catch shape bugs early
# ------------------------------------------------------------
def assert_rank(tensor: torch.Tensor, expected_rank: int, name: str) -> None:
    """
    Check number of dimensions.

    Example:
      tensor shape [B,T,J,3] has rank = 4

    If rank is wrong, raise an error.
    """
    if tensor.ndim != expected_rank:
        raise AssertionError(f"{name} must have rank {expected_rank}, got shape {tuple(tensor.shape)}")


def assert_last_dim(tensor: torch.Tensor, expected_dim: int, name: str) -> None:
    """
    Check the last dimension size.

    Example:
      skeleton x_hat: [B,T,J,3] -> last dim must be 3
    """
    if tensor.shape[-1] != expected_dim:
        raise AssertionError(f"{name} last dim must be {expected_dim}, got {tensor.shape[-1]}")


def assert_shape_prefix(tensor: torch.Tensor, expected_prefix: Sequence[int], name: str) -> None:
    """
    Check that the first few dimensions match exactly.

    Example:
      expected_prefix = (B, T)
      tensor shape might be [B,T,J,3]
      we check prefix [B,T]

    This is useful to ensure tensors align before doing operations.
    """
    prefix = tuple(tensor.shape[: len(expected_prefix)])
    target = tuple(expected_prefix)
    if prefix != target:
        raise AssertionError(f"{name} prefix shape must be {target}, got {prefix}")


# ------------------------------------------------------------
# 4) Count trainable parameters (helps debugging model size)
# ------------------------------------------------------------
def count_trainable_parameters(module: torch.nn.Module) -> int:
    """
    Count how many parameters will actually learn (requires_grad=True).

    This is useful if you freeze a model and want to confirm it worked.
    """
    total = 0
    for p in module.parameters():
        if p.requires_grad:
            total += p.numel()
    return total


# ------------------------------------------------------------
# 5) Build a simple chain adjacency matrix
# ------------------------------------------------------------
def build_chain_adjacency(
    num_joints: int,
    include_self: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build an adjacency matrix for a simple chain skeleton:

    joint 0 -- joint 1 -- joint 2 -- ... -- joint (J-1)

    adjacency shape: [J, J]
    adjacency[i,j] = 1 means joint i is connected to joint j.

    Args:
      num_joints: how many joints (J)
      include_self: add adjacency[i,i] = 1 if True
      device: cpu or cuda device

    Returns:
      torch float tensor [J,J]
    """
    J = int(num_joints)

    # start with all zeros
    adj = torch.zeros((J, J), dtype=torch.float32, device=device)

    # connect neighbors
    for i in range(J - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    # optional self connections
    if include_self:
        adj.fill_diagonal_(1.0)

    return adj


# ------------------------------------------------------------
# 5b) Build SmartFall anatomical bone adjacency matrix (32 joints)
# ------------------------------------------------------------
def build_smartfall_bone_adjacency(
    num_joints: int = 32,
    include_self: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build adjacency for SmartFall 32-joint skeleton using anatomical bone edges.

    Expected joint indexing (Kinect-style 32-joint layout):
      0 pelvis, 1 spine_navel, 2 spine_chest, 3 neck, 4 clavicle_l, 5 shoulder_l,
      6 elbow_l, 7 wrist_l, 8 hand_l, 9 handtip_l, 10 thumb_l, 11 clavicle_r,
      12 shoulder_r, 13 elbow_r, 14 wrist_r, 15 hand_r, 16 handtip_r, 17 thumb_r,
      18 hip_l, 19 knee_l, 20 ankle_l, 21 foot_l, 22 hip_r, 23 knee_r, 24 ankle_r,
      25 foot_r, 26 head, 27 nose, 28 eye_l, 29 ear_l, 30 eye_r, 31 ear_r.
    """
    J = int(num_joints)
    if J != 32:
        raise ValueError(
            f"build_smartfall_bone_adjacency expects num_joints=32, got {J}. "
            "Update the edge list if your dataset uses a different joint layout."
        )

    edges = [
        # trunk
        (0, 1), (1, 2), (2, 3), (3, 26), (26, 27),
        # face
        (27, 28), (28, 29), (27, 30), (30, 31),
        # left arm
        (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 10),
        # right arm
        (2, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (14, 17),
        # left leg
        (0, 18), (18, 19), (19, 20), (20, 21),
        # right leg
        (0, 22), (22, 23), (23, 24), (24, 25),
    ]

    adj = torch.zeros((J, J), dtype=torch.float32, device=device)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    if include_self:
        adj.fill_diagonal_(1.0)

    return adj


# ------------------------------------------------------------
# 6) Choose device (cpu or gpu)
# ------------------------------------------------------------
def resolve_device(device: Optional[str] = None) -> torch.device:
    """
    Decide which device to use.

    If user gives device string, we use it:
      resolve_device("cuda") -> cuda
      resolve_device("cpu")  -> cpu

    If user does not give device:
      if CUDA is available -> cuda
      else -> cpu
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ------------------------------------------------------------
# 7) Ensure a directory exists (so saving files won't crash)
# ------------------------------------------------------------
def ensure_dir(path: Union[str, os.PathLike]) -> None:
    """
    Create a folder if it doesn't exist.

    Example:
      ensure_dir("checkpoints/stage3")
      (now you can safely save into that folder)
    """
    Path(path).mkdir(parents=True, exist_ok=True)
