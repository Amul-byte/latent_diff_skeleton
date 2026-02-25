"""Datasets for skeleton and acceleration-only multimodal training.

This module supports two acceleration sensors (for example: ``meta_hip`` +
``meta_wrist`` or ``phone`` + ``watch``) and applies per-sensor normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch.utils.data import Dataset

from diffusion_model.util import assert_last_dim, assert_rank


@dataclass
class NormalizationConfig:
    """Configuration for acceleration normalization.

    Attributes:
        mode: Normalization mode, either ``"zscore"`` or ``"none"``.
        eps: Small epsilon used to avoid division by zero.
        clip: Optional post-normalization clipping threshold.
    """

    mode: str = "zscore"
    eps: float = 1e-6
    clip: Optional[float] = 6.0


def _validate_accel_tensor(name: str, tensor: torch.Tensor) -> None:
    """Validate acceleration tensor shape.

    Args:
        name: Tensor name for diagnostics.
        tensor: Tensor expected to be ``[N, T, 3]``.
    """
    assert_rank(tensor, 3, name)
    assert_last_dim(tensor, 3, name)


def _compute_zscore_stats(accel: torch.Tensor, eps: float) -> Dict[str, torch.Tensor]:
    """Compute dataset-level per-axis z-score statistics.

    Args:
        accel: Acceleration tensor ``[N, T, 3]``.
        eps: Minimum std clamp.

    Returns:
        Dictionary containing ``mean`` and ``std`` with shape ``[1, 1, 3]``.
    """
    mean = accel.mean(dim=(0, 1), keepdim=True)
    std = accel.std(dim=(0, 1), keepdim=True).clamp_min(eps)
    return {"mean": mean, "std": std}


def _normalize_accel(
    accel: torch.Tensor,
    config: NormalizationConfig,
    stats: Optional[Mapping[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Normalize one acceleration stream.

    Args:
        accel: Input acceleration tensor ``[N, T, 3]``.
        config: Normalization settings.
        stats: Optional precomputed statistics with keys ``mean`` and ``std``.

    Returns:
        Tuple of normalized tensor and resolved stats dictionary.
    """
    if config.mode not in {"zscore", "none"}:
        raise ValueError(f"Unsupported normalization mode: {config.mode}")

    accel = accel.float()
    if config.mode == "none":
        resolved_stats = {
            "mean": torch.zeros((1, 1, 3), dtype=accel.dtype),
            "std": torch.ones((1, 1, 3), dtype=accel.dtype),
        }
        return accel, resolved_stats

    if stats is not None and "mean" in stats and "std" in stats:
        mean = torch.as_tensor(stats["mean"], dtype=accel.dtype)
        std = torch.as_tensor(stats["std"], dtype=accel.dtype).clamp_min(config.eps)
        if mean.shape != (1, 1, 3) or std.shape != (1, 1, 3):
            raise AssertionError("Provided normalization stats must have shape [1, 1, 3]")
        resolved_stats = {"mean": mean, "std": std}
    else:
        resolved_stats = _compute_zscore_stats(accel, config.eps)

    normalized = (accel - resolved_stats["mean"]) / resolved_stats["std"]
    if config.clip is not None:
        normalized = normalized.clamp(-float(config.clip), float(config.clip))
    return normalized, resolved_stats


def _resolve_sensor_pair(
    accel_by_sensor: Mapping[str, torch.Tensor],
    sensor_pair: Tuple[str, str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve acceleration tensors from a named sensor dictionary.

    Args:
        accel_by_sensor: Mapping from sensor name to tensor ``[N, T, 3]``.
        sensor_pair: Pair of sensor names to select.

    Returns:
        Selected pair of acceleration tensors.
    """
    sensor_a, sensor_b = sensor_pair
    if sensor_a not in accel_by_sensor:
        raise KeyError(f"Sensor '{sensor_a}' not present in accel_by_sensor")
    if sensor_b not in accel_by_sensor:
        raise KeyError(f"Sensor '{sensor_b}' not present in accel_by_sensor")
    return accel_by_sensor[sensor_a], accel_by_sensor[sensor_b]


class SkeletonDataset(Dataset[Dict[str, torch.Tensor]]):
    """Skeleton sequence dataset.

    Returns per sample:
        - ``X``: skeleton tensor with shape ``[T, J, 3]``
        - ``y``: label scalar (or ``-1`` when unavailable)
    """

    def __init__(
        self,
        skeleton: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the skeleton dataset.

        Args:
            skeleton: Skeleton tensor ``[N, T, J, 3]``.
            labels: Optional label tensor ``[N]``.
        """
        if skeleton is None:
            raise ValueError("skeleton is required")

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
    """Acceleration-only dataset with two configurable sensor streams.

    Returns per sample:
        - ``A1``: first acceleration sensor stream ``[T, 3]``
        - ``A2``: second acceleration sensor stream ``[T, 3]``
        - ``A_pair``: stacked acceleration pair ``[2, T, 3]``
        - ``y``: label scalar (or ``-1`` when unavailable)
    """

    def __init__(
        self,
        accel_primary: Optional[torch.Tensor] = None,
        accel_secondary: Optional[torch.Tensor] = None,
        accel_by_sensor: Optional[Mapping[str, torch.Tensor]] = None,
        sensor_pair: Tuple[str, str] = ("meta_hip", "meta_wrist"),
        labels: Optional[torch.Tensor] = None,
        normalization: Optional[NormalizationConfig] = None,
        normalization_stats: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
    ) -> None:
        """Initialize two-sensor acceleration dataset.

        Args:
            accel_primary: Optional first acceleration tensor ``[N, T, 3]``.
            accel_secondary: Optional second acceleration tensor ``[N, T, 3]``.
            accel_by_sensor: Optional mapping of sensor names to tensors.
            sensor_pair: Selected sensor names from ``accel_by_sensor``.
            labels: Optional label tensor ``[N]``.
            normalization: Normalization settings.
            normalization_stats: Optional precomputed normalization stats.
        """
        norm_cfg = normalization or NormalizationConfig()

        if accel_by_sensor is not None:
            accel_primary, accel_secondary = _resolve_sensor_pair(accel_by_sensor, sensor_pair)

        if accel_primary is None or accel_secondary is None:
            raise ValueError("Both accel_primary and accel_secondary must be provided")

        _validate_accel_tensor("accel_primary", accel_primary)
        _validate_accel_tensor("accel_secondary", accel_secondary)
        if accel_primary.shape != accel_secondary.shape:
            raise AssertionError("accel_primary and accel_secondary shapes must match")
        if labels is not None and labels.shape[0] != accel_primary.shape[0]:
            raise AssertionError("labels and acceleration tensors must share the sample dimension")

        self.sensor_pair = sensor_pair
        stats_input = normalization_stats or {}
        primary_stats = stats_input.get(sensor_pair[0])
        secondary_stats = stats_input.get(sensor_pair[1])

        normalized_primary, resolved_primary_stats = _normalize_accel(accel_primary, norm_cfg, stats=primary_stats)
        normalized_secondary, resolved_secondary_stats = _normalize_accel(accel_secondary, norm_cfg, stats=secondary_stats)

        self.accel_primary = normalized_primary
        self.accel_secondary = normalized_secondary
        self.labels = labels.long() if labels is not None else None
        self.normalization_config = norm_cfg
        self.normalization_stats: Dict[str, Dict[str, torch.Tensor]] = {
            sensor_pair[0]: resolved_primary_stats,
            sensor_pair[1]: resolved_secondary_stats,
        }

    def __len__(self) -> int:
        """Return dataset length."""
        return int(self.accel_primary.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get one acceleration sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary with ``A1``, ``A2``, ``A_pair``, and ``y``.
        """
        a1 = self.accel_primary[index]
        a2 = self.accel_secondary[index]
        label = torch.tensor(-1, dtype=torch.long) if self.labels is None else self.labels[index]
        return {
            "A1": a1,
            "A2": a2,
            "A_pair": torch.stack([a1, a2], dim=0),
            "y": label,
        }

    def get_normalization_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return a copy of normalization stats for both sensors.

        Returns:
            Sensor-keyed dictionary with ``mean`` and ``std`` tensors.
        """
        stats: Dict[str, Dict[str, torch.Tensor]] = {}
        for sensor_name, sensor_stats in self.normalization_stats.items():
            stats[sensor_name] = {
                "mean": sensor_stats["mean"].clone(),
                "std": sensor_stats["std"].clone(),
            }
        return stats


class PairedDataset(Dataset[Dict[str, torch.Tensor]]):
    """Paired skeleton and acceleration-only dataset.

    Returns per sample:
        - ``X``: skeleton ``[T, J, 3]``
        - ``A1``: first acceleration stream ``[T, 3]``
        - ``A2``: second acceleration stream ``[T, 3]``
        - ``A_pair``: stacked acceleration pair ``[2, T, 3]``
        - ``y``: label scalar (or ``-1`` when unavailable)
    """

    def __init__(
        self,
        skeleton: Optional[torch.Tensor] = None,
        accel_primary: Optional[torch.Tensor] = None,
        accel_secondary: Optional[torch.Tensor] = None,
        accel_by_sensor: Optional[Mapping[str, torch.Tensor]] = None,
        sensor_pair: Tuple[str, str] = ("meta_hip", "meta_wrist"),
        labels: Optional[torch.Tensor] = None,
        normalization: Optional[NormalizationConfig] = None,
        normalization_stats: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
    ) -> None:
        """Initialize paired skeleton and acceleration dataset.

        Args:
            skeleton: Optional skeleton tensor ``[N, T, J, 3]``.
            accel_primary: Optional first acceleration tensor ``[N, T, 3]``.
            accel_secondary: Optional second acceleration tensor ``[N, T, 3]``.
            accel_by_sensor: Optional sensor-name mapping for accelerations.
            sensor_pair: Selected pair from ``accel_by_sensor``.
            labels: Optional label tensor ``[N]``.
            normalization: Normalization settings.
            normalization_stats: Optional precomputed normalization stats.
        """
        norm_cfg = normalization or NormalizationConfig()

        if accel_by_sensor is not None:
            accel_primary, accel_secondary = _resolve_sensor_pair(accel_by_sensor, sensor_pair)

        if skeleton is None or accel_primary is None or accel_secondary is None:
            raise ValueError("skeleton, accel_primary, and accel_secondary are required")

        assert_rank(skeleton, 4, "skeleton")
        assert_last_dim(skeleton, 3, "skeleton")
        _validate_accel_tensor("accel_primary", accel_primary)
        _validate_accel_tensor("accel_secondary", accel_secondary)

        if accel_primary.shape != accel_secondary.shape:
            raise AssertionError("accel_primary and accel_secondary shapes must match")
        if skeleton.shape[0] != accel_primary.shape[0] or skeleton.shape[1] != accel_primary.shape[1]:
            raise AssertionError("skeleton and acceleration streams must match on [N, T]")
        if labels is not None and labels.shape[0] != skeleton.shape[0]:
            raise AssertionError("labels and paired tensors must share sample dimension")

        self.skeleton = skeleton.float()
        self.sensor_pair = sensor_pair
        stats_input = normalization_stats or {}
        primary_stats = stats_input.get(sensor_pair[0])
        secondary_stats = stats_input.get(sensor_pair[1])

        normalized_primary, resolved_primary_stats = _normalize_accel(accel_primary, norm_cfg, stats=primary_stats)
        normalized_secondary, resolved_secondary_stats = _normalize_accel(accel_secondary, norm_cfg, stats=secondary_stats)

        self.accel_primary = normalized_primary
        self.accel_secondary = normalized_secondary
        self.labels = labels.long() if labels is not None else None
        self.normalization_config = norm_cfg
        self.normalization_stats: Dict[str, Dict[str, torch.Tensor]] = {
            sensor_pair[0]: resolved_primary_stats,
            sensor_pair[1]: resolved_secondary_stats,
        }

    def __len__(self) -> int:
        """Return dataset length."""
        return int(self.skeleton.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get one paired sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary with ``X``, ``A1``, ``A2``, ``A_pair``, and ``y``.
        """
        a1 = self.accel_primary[index]
        a2 = self.accel_secondary[index]
        label = torch.tensor(-1, dtype=torch.long) if self.labels is None else self.labels[index]
        return {
            "X": self.skeleton[index],
            "A1": a1,
            "A2": a2,
            "A_pair": torch.stack([a1, a2], dim=0),
            "y": label,
        }

    def get_normalization_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return a copy of normalization stats for both acceleration sensors.

        Returns:
            Sensor-keyed dictionary with ``mean`` and ``std`` tensors.
        """
        stats: Dict[str, Dict[str, torch.Tensor]] = {}
        for sensor_name, sensor_stats in self.normalization_stats.items():
            stats[sensor_name] = {
                "mean": sensor_stats["mean"].clone(),
                "std": sensor_stats["std"].clone(),
            }
        return stats
