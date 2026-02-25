"""Datasets for skeleton, IMU, and paired multimodal training with toy fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from diffusion_model.util import assert_last_dim, assert_rank


@dataclass
class ToyConfig:
    """Configuration for synthetic dataset generation.

    Attributes:
        num_samples: Number of samples.
        window: Sequence length.
        joints: Number of joints.
        joint_dim: Joint feature dimension.
        num_classes: Number of classes.
        seed: Random seed.
    """

    num_samples: int = 128
    window: int = 90
    joints: int = 32
    joint_dim: int = 3
    num_classes: int = 14
    seed: int = 42


def _make_generator(seed: int) -> torch.Generator:
    """Build a deterministic CPU random generator.

    Args:
        seed: Random seed.

    Returns:
        Configured torch.Generator instance.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _toy_labels(num_samples: int, num_classes: int, seed: int) -> torch.Tensor:
    """Generate synthetic integer labels.

    Args:
        num_samples: Number of labels.
        num_classes: Number of class IDs.
        seed: Random seed.

    Returns:
        Tensor with shape [N].
    """
    return torch.randint(0, num_classes, (num_samples,), generator=_make_generator(seed), dtype=torch.long)


class SkeletonDataset(Dataset[Dict[str, torch.Tensor]]):
    """Skeleton sequence dataset.

    Returns per sample:
        - ``X``: skeleton tensor with shape [T, J, 3]
        - ``y``: label scalar (or -1 when unavailable)
    """

    def __init__(
        self,
        skeleton: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        toy: bool = False,
        toy_config: Optional[ToyConfig] = None,
    ) -> None:
        """Initialize the skeleton dataset.

        Args:
            skeleton: Optional skeleton tensor [N, T, J, 3].
            labels: Optional label tensor [N].
            toy: Whether to generate synthetic data.
            toy_config: Optional synthetic generation settings.
        """
        if toy or skeleton is None:
            cfg = toy_config or ToyConfig()
            skeleton = torch.randn(
                cfg.num_samples,
                cfg.window,
                cfg.joints,
                cfg.joint_dim,
                generator=_make_generator(cfg.seed),
            )
            labels = _toy_labels(cfg.num_samples, cfg.num_classes, cfg.seed + 17)

        assert_rank(skeleton, 4, "skeleton")
        assert_last_dim(skeleton, 3, "skeleton")
        if labels is not None and labels.shape[0] != skeleton.shape[0]:
            raise AssertionError("labels and skeleton must share the sample dimension")

        self.skeleton = skeleton.float()
        self.labels = labels.long() if labels is not None else None

    def __len__(self) -> int:
        """Return dataset length."""
        return int(self.skeleton.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get one skeleton sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary with ``X`` and ``y``.
        """
        label = torch.tensor(-1, dtype=torch.long) if self.labels is None else self.labels[index]
        return {"X": self.skeleton[index], "y": label}


class IMUDataset(Dataset[Dict[str, torch.Tensor]]):
    """IMU sequence dataset.

    Returns per sample:
        - ``A``: accelerometer stream [T, 3]
        - ``Omega``: gyroscope stream [T, 3]
        - ``y``: label scalar (or -1 when unavailable)
    """

    def __init__(
        self,
        accel: Optional[torch.Tensor] = None,
        gyro: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        toy: bool = False,
        toy_config: Optional[ToyConfig] = None,
    ) -> None:
        """Initialize the IMU dataset.

        Args:
            accel: Optional accelerometer tensor [N, T, 3].
            gyro: Optional gyroscope tensor [N, T, 3].
            labels: Optional label tensor [N].
            toy: Whether to generate synthetic data.
            toy_config: Optional synthetic generation settings.
        """
        if toy or accel is None or gyro is None:
            cfg = toy_config or ToyConfig()
            accel = torch.randn(cfg.num_samples, cfg.window, 3, generator=_make_generator(cfg.seed + 1))
            gyro = torch.randn(cfg.num_samples, cfg.window, 3, generator=_make_generator(cfg.seed + 2))
            labels = _toy_labels(cfg.num_samples, cfg.num_classes, cfg.seed + 19)

        assert_rank(accel, 3, "accel")
        assert_rank(gyro, 3, "gyro")
        assert_last_dim(accel, 3, "accel")
        assert_last_dim(gyro, 3, "gyro")
        if accel.shape != gyro.shape:
            raise AssertionError("accel and gyro shapes must match")
        if labels is not None and labels.shape[0] != accel.shape[0]:
            raise AssertionError("labels and imu must share the sample dimension")

        self.accel = accel.float()
        self.gyro = gyro.float()
        self.labels = labels.long() if labels is not None else None

    def __len__(self) -> int:
        """Return dataset length."""
        return int(self.accel.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get one IMU sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary with ``A``, ``Omega``, and ``y``.
        """
        label = torch.tensor(-1, dtype=torch.long) if self.labels is None else self.labels[index]
        return {"A": self.accel[index], "Omega": self.gyro[index], "y": label}


class PairedDataset(Dataset[Dict[str, torch.Tensor]]):
    """Paired skeleton and IMU dataset.

    Returns per sample:
        - ``X``: skeleton [T, J, 3]
        - ``A``: accel [T, 3]
        - ``Omega``: gyro [T, 3]
        - ``y``: label scalar (or -1 when unavailable)
    """

    def __init__(
        self,
        skeleton: Optional[torch.Tensor] = None,
        accel: Optional[torch.Tensor] = None,
        gyro: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        toy: bool = False,
        toy_config: Optional[ToyConfig] = None,
    ) -> None:
        """Initialize paired dataset.

        Args:
            skeleton: Optional skeleton tensor [N, T, J, 3].
            accel: Optional accelerometer tensor [N, T, 3].
            gyro: Optional gyroscope tensor [N, T, 3].
            labels: Optional label tensor [N].
            toy: Whether to generate synthetic data.
            toy_config: Optional synthetic generation settings.
        """
        if toy or skeleton is None or accel is None or gyro is None:
            cfg = toy_config or ToyConfig()
            base = torch.randn(cfg.num_samples, cfg.window, 1, generator=_make_generator(cfg.seed + 3))
            skeleton_noise = torch.randn(
                cfg.num_samples,
                cfg.window,
                cfg.joints,
                cfg.joint_dim,
                generator=_make_generator(cfg.seed + 4),
            )
            accel_noise = torch.randn(cfg.num_samples, cfg.window, 3, generator=_make_generator(cfg.seed + 5))
            gyro_noise = torch.randn(cfg.num_samples, cfg.window, 3, generator=_make_generator(cfg.seed + 6))

            skeleton = 0.35 * base.unsqueeze(-1) + skeleton_noise
            accel = 0.60 * base + accel_noise
            gyro = -0.40 * base + gyro_noise
            labels = _toy_labels(cfg.num_samples, cfg.num_classes, cfg.seed + 21)

        assert_rank(skeleton, 4, "skeleton")
        assert_rank(accel, 3, "accel")
        assert_rank(gyro, 3, "gyro")
        assert_last_dim(skeleton, 3, "skeleton")
        assert_last_dim(accel, 3, "accel")
        assert_last_dim(gyro, 3, "gyro")
        if accel.shape != gyro.shape:
            raise AssertionError("accel and gyro shapes must match")
        if skeleton.shape[0] != accel.shape[0] or skeleton.shape[1] != accel.shape[1]:
            raise AssertionError("skeleton and imu must match on [N, T]")
        if labels is not None and labels.shape[0] != skeleton.shape[0]:
            raise AssertionError("labels and paired tensors must share sample dimension")

        self.skeleton = skeleton.float()
        self.accel = accel.float()
        self.gyro = gyro.float()
        self.labels = labels.long() if labels is not None else None

    def __len__(self) -> int:
        """Return dataset length."""
        return int(self.skeleton.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get one paired sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary with ``X``, ``A``, ``Omega``, and ``y``.
        """
        label = torch.tensor(-1, dtype=torch.long) if self.labels is None else self.labels[index]
        return {
            "X": self.skeleton[index],
            "A": self.accel[index],
            "Omega": self.gyro[index],
            "y": label,
        }
