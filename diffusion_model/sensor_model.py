"""
imu_encoder.py

This file turns IMU acceleration signals into "features" that the diffusion model can use.

Proposal alignment note (Option A, accel-only):
- We map S=(A, Omega) to two accelerometer sources:
  A1 = right-hip accel, A2 = left-wrist accel.
- Gyroscope is intentionally not used in this option.

✅ Matches your dataset.py:
- dataset returns:
    batch["A1"]    -> [B, T, 3]   (sensor 1 accel)
    batch["A2"]    -> [B, T, 3]   (sensor 2 accel)
    batch["A_pair"]-> [B, 2, T, 3]

So this encoder can accept:
- either two tensors (A1, A2)
- or a batch dict (and it will take A1/A2 from it)

What this file does (very simple):
1) For each IMU stream (sensor 1 and sensor 2), run a small 1D CNN over time.
2) Join the two results together (concatenate).
3) Convert that into:
   - h_seq:  [B, T, D]  (sequence features per time step)
   - h_joint:[B, T, J, D] (same info repeated for each joint, with tiny joint-specific tokens)

Why we need h_joint:
- Your Stage-1 skeleton latent z0 usually looks like [B, T, J, D]
- So h_joint lines up with that shape and can be used for conditioning.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union, Optional

import torch
import torch.nn as nn


# ------------------------------------------------------------
# Small helper checks (easy to read)
# ------------------------------------------------------------
def _check_imu_stream(name: str, x: torch.Tensor) -> None:
    """
    We expect IMU stream to be shaped [B, T, 3]
    B = batch size
    T = time frames (window length)
    3 = accel x,y,z
    """
    if x.ndim != 3:
        raise ValueError(f"{name} must be 3D [B,T,3]. Got shape {tuple(x.shape)}")
    if x.shape[-1] != 3:
        raise ValueError(f"{name} last dim must be 3 (x,y,z). Got shape {tuple(x.shape)}")


# ------------------------------------------------------------
# IMUBranch: 1 stream -> features
# ------------------------------------------------------------
class IMUBranch(nn.Module):
    """
    This processes ONE sensor stream [B,T,3] and outputs features [B,T,C].

    We use Conv1D because it can learn patterns over time (like spikes, waves, etc).
    """

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()

        # A simple CNN stack:
        # - First conv changes channels from 3 -> hidden_dim
        # - Next conv keeps hidden_dim
        # - Last conv changes hidden_dim -> out_dim
        #
        # NOTE: Conv1d expects shape [B, C, T], not [B, T, C]
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, stream: torch.Tensor) -> torch.Tensor:
        """
        stream: [B,T,3]
        return: [B,T,out_dim]
        """

        # Make sure shape is correct
        _check_imu_stream("stream", stream)

        # Convert [B,T,3] -> [B,3,T] for Conv1d
        x = stream.transpose(1, 2)

        # Run the CNN
        x = self.net(x)

        # Convert back [B,out_dim,T] -> [B,T,out_dim]
        x = x.transpose(1, 2)

        return x


# ------------------------------------------------------------
# Two-sensor encoder: sensor1 + sensor2 -> conditioning features
# ------------------------------------------------------------
class TwoSensorIMUEncoder(nn.Module):
    """
    This is the full IMU encoder.

    Input:
      - sensor1_accel: [B,T,3]
      - sensor2_accel: [B,T,3]

    Output:
      - h_joint: [B,T,J,D]
      - h_seq:   [B,T,D]

    Where:
      B = batch size
      T = time
      J = num_joints (usually 32)
      D = latent_dim (usually 256)
    """

    def __init__(
        self,
        latent_dim: int = 256,  # final feature size D
        num_joints: int = 32,   # J (must match skeleton joints)
        hidden_dim: int = 256,  # CNN hidden channels
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.num_joints = num_joints

        # Each branch outputs about half the latent features
        branch_out = max(latent_dim // 2, 16)

        # One branch for sensor1 and one for sensor2
        self.branch1 = IMUBranch(hidden_dim=hidden_dim, out_dim=branch_out)
        self.branch2 = IMUBranch(hidden_dim=hidden_dim, out_dim=branch_out)

        # Fusion turns (branch_out*2) -> latent_dim
        self.fusion = nn.Sequential(
            nn.Linear(branch_out * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Normalization helps training be stable
        self.norm = nn.LayerNorm(latent_dim)

        # These are tiny learnable "joint tokens" added per joint
        # Shape: [1,1,J,D] so it can broadcast to [B,T,J,D]
        self.joint_tokens = nn.Parameter(torch.randn(1, 1, num_joints, latent_dim) * 0.02)

    # ----------------------------
    # Helper: get A1 and A2 from dict OR from tensors
    # ----------------------------
    def _get_streams(
        self,
        a1_or_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        a2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If input is a dict (batch), we use batch["A1"] and batch["A2"].
        Otherwise we assume the user passed a1 and a2 as tensors.
        """
        if isinstance(a1_or_batch, dict):
            batch = a1_or_batch
            if "A1" not in batch or "A2" not in batch:
                raise KeyError("Batch dict must contain 'A1' and 'A2'.")
            return batch["A1"], batch["A2"]

        # if not a dict, then user must pass both a1 and a2
        if a2 is None:
            raise ValueError("If you don't pass a batch dict, you must pass both a1 and a2 tensors.")
        return a1_or_batch, a2

    def forward(
        self,
        a1_or_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        a2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Accepts:
          - forward(batch_dict)
          OR
          - forward(a1_tensor, a2_tensor)

        Returns:
          h_joint: [B,T,J,D]
          h_seq:   [B,T,D]
        """

        # Get both sensor streams
        sensor1_accel, sensor2_accel = self._get_streams(a1_or_batch, a2)

        # Check shapes are correct
        _check_imu_stream("sensor1_accel", sensor1_accel)
        _check_imu_stream("sensor2_accel", sensor2_accel)

        # Make sure they match in batch size and time length
        if sensor1_accel.shape[:2] != sensor2_accel.shape[:2]:
            raise ValueError("sensor1_accel and sensor2_accel must have the same [B,T].")

        # Run each sensor through its own branch
        f1 = self.branch1(sensor1_accel)  # [B,T,branch_out]
        f2 = self.branch2(sensor2_accel)  # [B,T,branch_out]

        # Join features side-by-side
        fused = torch.cat([f1, f2], dim=-1)  # [B,T,branch_out*2]

        # Turn into final latent features D
        h_seq = self.fusion(fused)           # [B,T,D]
        h_seq = self.norm(h_seq)             # [B,T,D]

        # Create h_joint by repeating h_seq for every joint
        # h_seq.unsqueeze(2) -> [B,T,1,D]
        # add joint_tokens -> [B,T,J,D]
        B, T, D = h_seq.shape
        h_joint = h_seq.unsqueeze(2) + self.joint_tokens.expand(B, T, self.num_joints, D)

        return h_joint, h_seq


class SensorTGNNEncoder(TwoSensorIMUEncoder):
    """Backward-compatible alias used by train.py/generate.py."""

    pass
