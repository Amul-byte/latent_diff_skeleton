"""IMU encoder modules for Stage 2 latent regression."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from diffusion_model.util import assert_last_dim, assert_rank


class IMUBranch(nn.Module):
    """Temporal branch for a single IMU stream."""

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        """Initialize IMU branch.

        Args:
            hidden_dim: Intermediate channel size.
            out_dim: Output feature size per timestep.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(3, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, stream: torch.Tensor) -> torch.Tensor:
        """Encode one IMU stream.

        Args:
            stream: Tensor [B, T, 3].

        Returns:
            Feature tensor [B, T, C].
        """
        assert_rank(stream, 3, "stream")
        assert_last_dim(stream, 3, "stream")

        encoded = self.network(stream.transpose(1, 2)).transpose(1, 2)
        assert_rank(encoded, 3, "encoded")
        return encoded


class SensorTGNNEncoder(nn.Module):
    """Two-branch T-GNN style IMU encoder for hip and wrist streams.

    Input:
        - ``hip_accel``: [B, T, 3]
        - ``wrist_gyro``: [B, T, 3]

    Output:
        - ``h_joint``: [B, T, J, D] (latent aligned with Stage 1 ``z0``)
        - ``h_seq``: [B, T, D] (sequence-level conditioning)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_joints: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize two-branch IMU encoder.

        Args:
            latent_dim: Output latent feature size.
            num_joints: Number of skeleton joints.
            hidden_dim: Hidden channel size.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints

        branch_out = max(latent_dim // 2, 16)
        self.hip_branch = IMUBranch(hidden_dim=hidden_dim, out_dim=branch_out)
        self.wrist_branch = IMUBranch(hidden_dim=hidden_dim, out_dim=branch_out)

        self.fusion = nn.Sequential(
            nn.Linear(branch_out * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.sequence_norm = nn.LayerNorm(latent_dim)
        self.joint_tokens = nn.Parameter(torch.randn(1, 1, num_joints, latent_dim) * 0.02)

    def forward(self, hip_accel: torch.Tensor, wrist_gyro: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode hip and wrist IMU streams.

        Args:
            hip_accel: Hip accelerometer stream [B, T, 3].
            wrist_gyro: Wrist gyroscope stream [B, T, 3].

        Returns:
            Tuple of ``h_joint`` [B, T, J, D] and ``h_seq`` [B, T, D].
        """
        assert_rank(hip_accel, 3, "hip_accel")
        assert_rank(wrist_gyro, 3, "wrist_gyro")
        assert_last_dim(hip_accel, 3, "hip_accel")
        assert_last_dim(wrist_gyro, 3, "wrist_gyro")
        if hip_accel.shape[:2] != wrist_gyro.shape[:2]:
            raise AssertionError("hip_accel and wrist_gyro must match on [B, T]")

        hip_features = self.hip_branch(hip_accel)
        wrist_features = self.wrist_branch(wrist_gyro)
        fused = torch.cat([hip_features, wrist_features], dim=-1)

        h_seq = self.sequence_norm(self.fusion(fused))
        assert_rank(h_seq, 3, "h_seq")
        assert_last_dim(h_seq, self.latent_dim, "h_seq")

        batch, time, latent_dim = h_seq.shape
        if latent_dim != self.latent_dim:
            raise AssertionError("fusion output latent_dim mismatch")

        h_joint = h_seq.unsqueeze(2) + self.joint_tokens.expand(batch, time, self.num_joints, latent_dim)
        assert_rank(h_joint, 4, "h_joint")
        if h_joint.shape != (batch, time, self.num_joints, self.latent_dim):
            raise AssertionError("h_joint shape mismatch with expected [B, T, J, D]")

        return h_joint, h_seq
