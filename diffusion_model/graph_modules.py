"""
graph_modules.py

This file contains the "graph neural network" pieces used by your models.

✅ Matches your dataset.py:
- skeleton input X from dataset: [B, T, J, 3]  where J = 32
- adjacency matrix: [J, J]

✅ Used in:
- GraphEncoder:  X -> latent Z
- GraphDecoder:  latent Z -> skeleton X_hat
- GraphDenoiser: takes noisy latent Z_t and predicts noise (for diffusion)

Goal here:
- Make it easier to understand
- Keep the SAME behavior and SAME shapes
- Keep conditioning support:
    h can be:
      - None (unconditional)
      - [B, T, D]
      - [B, T, J, D]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Small helper checks (easy to read)
# ------------------------------------------------------------
def _check_rank(name: str, x: torch.Tensor, rank: int) -> None:
    if x.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(x.shape)}")


def _check_last_dim(name: str, x: torch.Tensor, last_dim: int) -> None:
    if x.shape[-1] != last_dim:
        raise ValueError(f"{name} last dim must be {last_dim}, got shape {tuple(x.shape)}")


# ------------------------------------------------------------
# 1) Timestep embedding (turn integer t into a vector)
# ------------------------------------------------------------
def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Convert timestep t into a sinusoidal embedding (like Transformers).

    Input:
      t: [B] (integers)
      dim: embedding size

    Output:
      [B, dim]
    """
    _check_rank("t", t, 1)

    half = dim // 2

    # create frequencies
    # (classic transformer sinusoid trick)
    freq_factor = math.log(10000.0) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -freq_factor)

    # angles = t * frequency
    angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]

    # build sin + cos
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [B, 2*half]

    # if dim is odd, pad one extra
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


# ------------------------------------------------------------
# 2) Temporal block (1D conv over time, for each joint)
# ------------------------------------------------------------
class TemporalResidualBlock(nn.Module):
    """
    This runs a small CNN over time, separately for each joint.

    Input:  x [B, T, J, C]
    Output: x [B, T, J, C]  (same shape)
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()

        # Two Conv1d layers for time modeling
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # GELU activation
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_rank("x", x, 4)

        B, T, J, C = x.shape

        # Conv1d expects [B, C, T]
        # We want to apply it per joint, so we flatten joints into batch:
        # [B, T, J, C] -> [B, J, C, T] -> [B*J, C, T]
        y = x.permute(0, 2, 3, 1).contiguous().view(B * J, C, T)

        # Apply convs
        y = self.conv1(y)
        y = self.act(y)
        y = self.conv2(y)

        # reshape back:
        # [B*J, C, T] -> [B, J, C, T] -> [B, T, J, C]
        y = y.view(B, J, C, T).permute(0, 3, 1, 2).contiguous()

        # residual connection: output = x + y
        return x + y


# ------------------------------------------------------------
# 3) Graph attention block with adjacency mask
# ------------------------------------------------------------
class MaskedGraphAttentionBlock(nn.Module):
    """
    This does attention across joints, but ONLY allows edges given by adjacency.

    Input:  x [B, T, J, C]
    Output: x [B, T, J, C]
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Feed-forward network (MLP like transformer)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def _masked_attention(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Do masked multi-head attention across joints.

        x: [B, T, J, C]
        adjacency: [J, J] (0/1 or bool)

        returns: [B, T, J, C]
        """

        B, T, J, C = x.shape

        # Make Q,K,V and split into heads
        # [B,T,J,C] -> [B,T,J,H,HD]
        q = self.q_proj(x).view(B, T, J, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, J, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, J, self.num_heads, self.head_dim)

        # Move heads earlier:
        # q,k,v: [B,T,H,J,HD]
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        # Attention scores across joints:
        # scores: [B,T,H,J,J]
        scores = torch.einsum("bthid,bthjd->bthij", q, k) / math.sqrt(self.head_dim)

        # Mask using adjacency:
        # adjacency [J,J] -> mask [1,1,1,J,J]
        mask = adjacency.bool().unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Put very negative number where mask is False
        scores = scores.masked_fill(~mask, -1e4)

        # Softmax to get weights
        w = torch.softmax(scores, dim=-1)

        # Weighted sum: [B,T,H,J,HD]
        attended = torch.einsum("bthij,bthjd->bthid", w, v)

        # Merge heads back:
        # [B,T,H,J,HD] -> [B,T,J,H,HD] -> [B,T,J,C]
        attended = attended.permute(0, 1, 3, 2, 4).contiguous().view(B, T, J, C)

        # Final projection
        return self.out_proj(attended)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        _check_rank("x", x, 4)
        _check_rank("adjacency", adjacency, 2)

        B, T, J, C = x.shape

        # adjacency must match joints
        if adjacency.shape != (J, J):
            raise ValueError(f"adjacency must be [{J},{J}], got {tuple(adjacency.shape)}")

        # feature dim must match hidden_dim
        if C != self.hidden_dim:
            raise ValueError(f"x feature dim must be {self.hidden_dim}, got {C}")

        # ensure adjacency is float/bool on right device
        adjacency = adjacency.to(device=x.device)

        # add self-loops (each joint can attend to itself)
        # We make a copy so we don't modify the original
        adjacency = adjacency.clone()
        adjacency.fill_diagonal_(1)

        # ---- Attention block (with residual) ----
        x_norm = self.norm1(x)
        attn_out = self._masked_attention(x_norm, adjacency)
        x = x + attn_out

        # ---- Feed-forward block (with residual) ----
        x_norm2 = self.norm2(x)
        x = x + self.ffn(x_norm2)

        return x


# ------------------------------------------------------------
# 4) GraphEncoder: skeleton -> latent
# ------------------------------------------------------------
class GraphEncoder(nn.Module):
    """
    Input:
      X: [B,T,J,3]
    Output:
      Z: [B,T,J,latent_dim]
    """

    def __init__(
        self,
        joint_dim: int = 3,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        use_torch_geometric: bool = False,
    ) -> None:
        super().__init__()
        _ = use_torch_geometric

        self.joint_dim = joint_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # project xyz into hidden space
        self.in_proj = nn.Linear(joint_dim, hidden_dim)

        # graph attention layers
        self.graph_layers = nn.ModuleList(
            [MaskedGraphAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )

        # temporal layers (over time)
        self.temp_layers = nn.ModuleList([TemporalResidualBlock(hidden_dim) for _ in range(2)])

        # project hidden into latent space
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        _check_rank("x", x, 4)
        _check_last_dim("x", x, self.joint_dim)
        _check_rank("adjacency", adjacency, 2)

        # xyz -> hidden
        h = self.in_proj(x)

        # run graph layers
        for layer in self.graph_layers:
            h = layer(h, adjacency)

        # run temporal layers
        for layer in self.temp_layers:
            h = layer(h)

        # hidden -> latent
        z = self.out_proj(h)

        _check_rank("z", z, 4)
        _check_last_dim("z", z, self.latent_dim)

        return z


# ------------------------------------------------------------
# 5) GraphDecoder: latent -> skeleton
# ------------------------------------------------------------
class GraphDecoder(nn.Module):
    """
    Input:
      Z: [B,T,J,latent_dim]
    Output:
      X_hat: [B,T,J,3]
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        joint_dim: int = 3,
        num_layers: int = 3,
        num_heads: int = 8,
        use_torch_geometric: bool = False,
    ) -> None:
        super().__init__()
        _ = use_torch_geometric

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.joint_dim = joint_dim

        # latent -> hidden
        self.in_proj = nn.Linear(latent_dim, hidden_dim)

        # temporal layers first
        self.temp_layers = nn.ModuleList([TemporalResidualBlock(hidden_dim) for _ in range(2)])

        # graph layers
        self.graph_layers = nn.ModuleList(
            [MaskedGraphAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )

        # hidden -> xyz
        self.out_proj = nn.Linear(hidden_dim, joint_dim)

    def forward(self, z: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        _check_rank("z", z, 4)
        _check_last_dim("z", z, self.latent_dim)
        _check_rank("adjacency", adjacency, 2)

        h = self.in_proj(z)

        for layer in self.temp_layers:
            h = layer(h)

        for layer in self.graph_layers:
            h = layer(h, adjacency)

        x_hat = self.out_proj(h)

        _check_rank("x_hat", x_hat, 4)
        _check_last_dim("x_hat", x_hat, self.joint_dim)

        return x_hat


# ------------------------------------------------------------
# 6) GraphDenoiserMasked: noisy latent -> predicted noise
# ------------------------------------------------------------
class GraphDenoiserMasked(nn.Module):
    """
    This predicts the noise added to the latent.

    Input:
      z_t: [B,T,J,D]
      t:   [B]
      h:   optional conditioning
           - None
           - [B,T,D]
           - [B,T,J,D]
    Output:
      noise_hat: [B,T,J,D]
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        use_torch_geometric: bool = False,
    ) -> None:
        super().__init__()
        _ = use_torch_geometric

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # latent -> hidden
        self.in_proj = nn.Linear(latent_dim, hidden_dim)

        # time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # conditioning projection (expects last dim = latent_dim)
        self.cond_proj = nn.Linear(latent_dim, hidden_dim)

        # graph layers
        self.graph_layers = nn.ModuleList(
            [MaskedGraphAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )

        # temporal
        self.temp = TemporalResidualBlock(hidden_dim)

        # output
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def _prepare_h(self, h: Optional[torch.Tensor], target_shape: torch.Size) -> Optional[torch.Tensor]:
        """
        Convert h into shape [B,T,J,D] so it can be added to hidden.

        target_shape is z_t.shape which is [B,T,J,D]
        """
        if h is None:
            return None

        # h could be [B,T,D]
        if h.ndim == 3:
            B, T, D = h.shape
            if (B, T) != (target_shape[0], target_shape[1]):
                raise ValueError("h [B,T,D] must match z_t [B,T,J,D] on first 2 dims")
            # expand to [B,T,J,D]
            h = h.unsqueeze(2).expand(B, T, target_shape[2], D)

        # or h could already be [B,T,J,D]
        elif h.ndim == 4:
            if tuple(h.shape[:3]) != tuple(target_shape[:3]):
                raise ValueError("h [B,T,J,D] must match z_t on [B,T,J]")

        else:
            raise ValueError("h must be rank 3 or 4")

        _check_last_dim("h", h, self.latent_dim)
        return h

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        adjacency: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _check_rank("z_t", z_t, 4)
        _check_rank("t", t, 1)
        _check_rank("adjacency", adjacency, 2)
        _check_last_dim("z_t", z_t, self.latent_dim)

        # t must match batch size
        if t.shape[0] != z_t.shape[0]:
            raise ValueError("t must have one timestep per batch item")

        B, T, J, D = z_t.shape

        # adjacency must match joints
        if adjacency.shape != (J, J):
            raise ValueError(f"adjacency must be [{J},{J}], got {tuple(adjacency.shape)}")

        # Make conditioning shape [B,T,J,D] if provided
        h = self._prepare_h(h, z_t.shape)

        # ---- Build hidden ----
        hidden = self.in_proj(z_t)  # [B,T,J,hidden_dim]

        # time embedding: [B,hidden_dim]
        t_emb = sinusoidal_timestep_embedding(t, self.hidden_dim)

        # pass through time MLP: [B,hidden_dim]
        t_emb = self.time_mlp(t_emb)

        # add time embedding to every (T,J) token:
        # [B,hidden_dim] -> [B,1,1,hidden_dim]
        hidden = hidden + t_emb.view(B, 1, 1, self.hidden_dim)

        # add conditioning if we have it
        if h is not None:
            hidden = hidden + self.cond_proj(h)

        # ---- Graph + temporal blocks ----
        for layer in self.graph_layers:
            hidden = layer(hidden, adjacency)

        hidden = self.temp(hidden)

        # ---- Output predicted noise ----
        noise_hat = self.out_proj(self.out_norm(hidden))

        # final safety checks
        if noise_hat.shape != z_t.shape:
            raise ValueError("noise_hat must have same shape as z_t")

        return noise_hat
