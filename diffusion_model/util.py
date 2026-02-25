"""Utility helpers for reproducibility, logging, and tensor validation."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set global random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Global seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger.

    Args:
        name: Logger name.
        log_file: Optional path for file logging.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def assert_rank(tensor: torch.Tensor, expected_rank: int, name: str) -> None:
    """Assert a tensor has a specific rank.

    Args:
        tensor: Tensor to check.
        expected_rank: Required rank.
        name: Human-readable tensor name.
    """
    if tensor.ndim != expected_rank:
        raise AssertionError(f"{name} must have rank {expected_rank}, got shape {tuple(tensor.shape)}")


def assert_last_dim(tensor: torch.Tensor, expected_dim: int, name: str) -> None:
    """Assert a tensor's last dimension matches expectation.

    Args:
        tensor: Tensor to check.
        expected_dim: Expected last dimension size.
        name: Human-readable tensor name.
    """
    if tensor.shape[-1] != expected_dim:
        raise AssertionError(f"{name} last dim must be {expected_dim}, got {tensor.shape[-1]}")


def assert_shape_prefix(tensor: torch.Tensor, expected_prefix: Sequence[int], name: str) -> None:
    """Assert leading dimensions match a fixed prefix.

    Args:
        tensor: Tensor to check.
        expected_prefix: Expected leading dimensions.
        name: Human-readable tensor name.
    """
    prefix = tuple(tensor.shape[: len(expected_prefix)])
    target = tuple(expected_prefix)
    if prefix != target:
        raise AssertionError(f"{name} prefix shape must be {target}, got {prefix}")


def count_trainable_parameters(module: torch.nn.Module) -> int:
    """Count trainable parameters of a module.

    Args:
        module: PyTorch module.

    Returns:
        Number of trainable parameters.
    """
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def build_chain_adjacency(
    num_joints: int,
    include_self: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a simple chain skeleton adjacency matrix.

    Args:
        num_joints: Number of joints.
        include_self: Whether to include identity connections.
        device: Optional output device.

    Returns:
        Float adjacency matrix with shape [J, J].
    """
    adjacency = torch.zeros((num_joints, num_joints), dtype=torch.float32, device=device)
    for idx in range(num_joints - 1):
        adjacency[idx, idx + 1] = 1.0
        adjacency[idx + 1, idx] = 1.0
    if include_self:
        adjacency.fill_diagonal_(1.0)
    return adjacency


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve a target computation device.

    Args:
        device: Optional explicit device string.

    Returns:
        PyTorch device.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Union[str, os.PathLike]) -> None:
    """Ensure a directory exists.

    Args:
        path: Directory path.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
