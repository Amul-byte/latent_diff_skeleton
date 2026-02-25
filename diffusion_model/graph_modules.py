"""Graph-based encoder/decoder/denoiser blocks for joint-aware latent modeling."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from diffusion_model.util import assert_last_dim, assert_rank


try:  # pragma: no cover - optional dependency
    from torch_geometric.nn import GATConv

    HAS_TORCH_GEOMETRIC = True
except Exception:  # pragma: no cover - optional dependency
    GATConv = None
    HAS_TORCH_GEOMETRIC = False


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: Integer diffusion steps [B].
        dim: Embedding dimension.

    Returns:
        Sinusoidal embeddings with shape [B, dim].
    """
    assert_rank(timesteps, 1, "timesteps")
    half_dim = dim // 2
    frequency = math.log(10000.0) / max(half_dim - 1, 1)
    frequency = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -frequency)
    angles = timesteps.float().unsqueeze(1) * frequency.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TemporalResidualBlock(nn.Module):
    """Temporal residual 1D convolution applied independently for each joint."""

    def __init__(self, hidden_dim: int) -> None:
        """Initialize temporal residual block.

        Args:
            hidden_dim: Feature size per joint token.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run temporal residual block.

        Args:
            x: Input tensor [B, T, J, C].

        Returns:
            Output tensor [B, T, J, C].
        """
        assert_rank(x, 4, "x")
        batch, time, joints, channels = x.shape
        y = x.permute(0, 2, 3, 1).contiguous().view(batch * joints, channels, time)
        y = self.conv2(self.activation(self.conv1(y)))
        y = y.view(batch, joints, channels, time).permute(0, 3, 1, 2).contiguous()
        return x + y


class MaskedGraphAttentionBlock(nn.Module):
    """Adjacency-masked graph attention block.

    Uses `torch_geometric` GAT when available and enabled, otherwise falls back to a
    pure PyTorch masked multi-head attention implementation.
    """

    def __init__(self, hidden_dim: int, num_heads: int, use_torch_geometric: bool = False) -> None:
        """Initialize graph attention block.

        Args:
            hidden_dim: Hidden feature dimension.
            num_heads: Number of attention heads.
            use_torch_geometric: Whether to prefer `torch_geometric` implementation.
        """
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_torch_geometric = bool(use_torch_geometric and HAS_TORCH_GEOMETRIC)

        self.norm = nn.LayerNorm(hidden_dim)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        if self.use_torch_geometric:
            if GATConv is None:
                raise RuntimeError("GATConv must be available when use_torch_geometric=True")
            self.gat = GATConv(
                in_channels=hidden_dim,
                out_channels=self.head_dim,
                heads=self.num_heads,
                concat=True,
                dropout=0.0,
            )
        else:
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.value_proj = nn.Linear(hidden_dim, hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def _masked_attention(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply pure PyTorch masked attention.

        Args:
            x: Normalized input [B, T, J, C].
            adjacency: Adjacency matrix [J, J].

        Returns:
            Attended tensor [B, T, J, C].
        """
        batch, time, joints, _ = x.shape
        q = self.query_proj(x).view(batch, time, joints, self.num_heads, self.head_dim)
        k = self.key_proj(x).view(batch, time, joints, self.num_heads, self.head_dim)
        v = self.value_proj(x).view(batch, time, joints, self.num_heads, self.head_dim)

        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        attention_scores = torch.einsum("bthid,bthjd->bthij", q, k) / math.sqrt(self.head_dim)
        mask = adjacency.bool().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        attention_scores = attention_scores.masked_fill(~mask, -1e4)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attended = torch.einsum("bthij,bthjd->bthid", attention_weights, v)
        attended = attended.permute(0, 1, 3, 2, 4).contiguous().view(batch, time, joints, self.hidden_dim)
        return self.output_proj(attended)

    def _pyg_attention(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply graph attention with `torch_geometric`.

        Args:
            x: Normalized input [B, T, J, C].
            adjacency: Adjacency matrix [J, J].

        Returns:
            Attended tensor [B, T, J, C].
        """
        if GATConv is None:
            raise RuntimeError("torch_geometric is unavailable")

        batch, time, joints, channels = x.shape
        edge_index = adjacency.bool().nonzero(as_tuple=False).t().contiguous()
        if edge_index.numel() == 0:
            raise AssertionError("adjacency must contain at least one edge")

        x_flat = x.view(batch * time, joints, channels)
        outputs = []
        for idx in range(batch * time):
            outputs.append(self.gat(x_flat[idx], edge_index))
        return torch.stack(outputs, dim=0).view(batch, time, joints, channels)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Run adjacency-constrained graph attention.

        Args:
            x: Input tensor [B, T, J, C].
            adjacency: Adjacency matrix [J, J].

        Returns:
            Output tensor [B, T, J, C].
        """
        assert_rank(x, 4, "x")
        assert_rank(adjacency, 2, "adjacency")
        _, _, joints, channels = x.shape
        if adjacency.shape != (joints, joints):
            raise AssertionError(f"adjacency must be [{joints}, {joints}], got {tuple(adjacency.shape)}")
        if channels != self.hidden_dim:
            raise AssertionError(f"x feature dim must be {self.hidden_dim}, got {channels}")

        adjacency = adjacency.to(device=x.device, dtype=torch.float32).clone()
        adjacency.fill_diagonal_(1.0)

        normalized = self.norm(x)
        if self.use_torch_geometric:
            attended = self._pyg_attention(normalized, adjacency)
        else:
            attended = self._masked_attention(normalized, adjacency)

        x = x + attended
        x = x + self.ffn(self.ff_norm(x))
        return x


class GraphEncoder(nn.Module):
    """Joint-aware graph encoder mapping skeleton coordinates to latent trajectories."""

    def __init__(
        self,
        joint_dim: int = 3,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        use_torch_geometric: bool = False,
    ) -> None:
        """Initialize graph encoder.

        Args:
            joint_dim: Input per-joint feature dimension.
            hidden_dim: Internal hidden size.
            latent_dim: Output latent size.
            num_layers: Number of graph attention blocks.
            num_heads: Attention heads per block.
            use_torch_geometric: Whether to use PyG GAT path.
        """
        super().__init__()
        self.joint_dim = joint_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(joint_dim, hidden_dim)
        self.graph_blocks = nn.ModuleList(
            [
                MaskedGraphAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    use_torch_geometric=use_torch_geometric,
                )
                for _ in range(num_layers)
            ]
        )
        self.temporal_blocks = nn.ModuleList([TemporalResidualBlock(hidden_dim) for _ in range(2)])
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Encode skeleton sequences.

        Args:
            x: Skeleton tensor [B, T, J, 3].
            adjacency: Adjacency matrix [J, J].

        Returns:
            Latent tensor [B, T, J, latent_dim].
        """
        assert_rank(x, 4, "x")
        assert_last_dim(x, self.joint_dim, "x")
        assert_rank(adjacency, 2, "adjacency")

        hidden = self.input_proj(x)
        for graph_block in self.graph_blocks:
            hidden = graph_block(hidden, adjacency)
        for temporal_block in self.temporal_blocks:
            hidden = temporal_block(hidden)
        latent = self.output_proj(hidden)
        assert_rank(latent, 4, "latent")
        assert_last_dim(latent, self.latent_dim, "latent")
        return latent


class GraphDecoder(nn.Module):
    """Joint-aware graph decoder mapping latent trajectories back to skeleton coordinates."""

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        joint_dim: int = 3,
        num_layers: int = 3,
        num_heads: int = 8,
        use_torch_geometric: bool = False,
    ) -> None:
        """Initialize graph decoder.

        Args:
            latent_dim: Input latent size.
            hidden_dim: Internal hidden size.
            joint_dim: Output per-joint coordinate size.
            num_layers: Number of graph attention blocks.
            num_heads: Attention heads per block.
            use_torch_geometric: Whether to use PyG GAT path.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.joint_dim = joint_dim

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.temporal_blocks = nn.ModuleList([TemporalResidualBlock(hidden_dim) for _ in range(2)])
        self.graph_blocks = nn.ModuleList(
            [
                MaskedGraphAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    use_torch_geometric=use_torch_geometric,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(hidden_dim, joint_dim)

    def forward(self, latent: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Decode latent trajectories.

        Args:
            latent: Latent tensor [B, T, J, latent_dim].
            adjacency: Adjacency matrix [J, J].

        Returns:
            Reconstructed skeleton tensor [B, T, J, 3].
        """
        assert_rank(latent, 4, "latent")
        assert_last_dim(latent, self.latent_dim, "latent")
        assert_rank(adjacency, 2, "adjacency")

        hidden = self.input_proj(latent)
        for temporal_block in self.temporal_blocks:
            hidden = temporal_block(hidden)
        for graph_block in self.graph_blocks:
            hidden = graph_block(hidden, adjacency)
        reconstruction = self.output_proj(hidden)
        assert_rank(reconstruction, 4, "reconstruction")
        assert_last_dim(reconstruction, self.joint_dim, "reconstruction")
        return reconstruction


class GraphDenoiserMasked(nn.Module):
    """Joint-aware latent denoiser conditioned on diffusion timestep and optional IMU embedding."""

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        use_torch_geometric: bool = False,
    ) -> None:
        """Initialize conditional denoiser.

        Args:
            latent_dim: Latent input/output size.
            hidden_dim: Hidden feature size.
            num_layers: Number of graph attention blocks.
            num_heads: Attention heads.
            use_torch_geometric: Whether to use PyG GAT path.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(latent_dim, hidden_dim)

        self.graph_blocks = nn.ModuleList(
            [
                MaskedGraphAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    use_torch_geometric=use_torch_geometric,
                )
                for _ in range(num_layers)
            ]
        )
        self.temporal_block = TemporalResidualBlock(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def _prepare_conditioning(self, h: Optional[torch.Tensor], target_shape: torch.Size) -> Optional[torch.Tensor]:
        """Normalize conditioning tensor to shape [B, T, J, D].

        Args:
            h: Optional conditioning tensor, [B, T, D] or [B, T, J, D].
            target_shape: Target latent shape [B, T, J, D].

        Returns:
            Conditioning tensor with shape [B, T, J, D], or ``None``.
        """
        if h is None:
            return None

        if h.ndim == 3:
            batch, time, dim = h.shape
            if (batch, time) != (target_shape[0], target_shape[1]):
                raise AssertionError("conditioning [B, T, D] must align with latent [B, T, J, D]")
            h = h.unsqueeze(2).expand(batch, time, target_shape[2], dim)
        elif h.ndim == 4:
            if tuple(h.shape[:3]) != tuple(target_shape[:3]):
                raise AssertionError("conditioning [B, T, J, D] must align with latent first 3 dims")
        else:
            raise AssertionError(f"conditioning rank must be 3 or 4, got {h.ndim}")

        assert_last_dim(h, self.latent_dim, "conditioning")
        return h

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        adjacency: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict diffusion noise.

        Args:
            z_t: Noisy latent tensor [B, T, J, D].
            t: Timesteps [B].
            adjacency: Adjacency matrix [J, J].
            h: Optional conditioning tensor [B, T, D] or [B, T, J, D].

        Returns:
            Predicted noise tensor [B, T, J, D].
        """
        assert_rank(z_t, 4, "z_t")
        assert_rank(t, 1, "t")
        assert_last_dim(z_t, self.latent_dim, "z_t")
        assert_rank(adjacency, 2, "adjacency")
        if t.shape[0] != z_t.shape[0]:
            raise AssertionError("t must contain one timestep per batch element")

        batch, _, joints, _ = z_t.shape
        if adjacency.shape != (joints, joints):
            raise AssertionError(f"adjacency must be [{joints}, {joints}], got {tuple(adjacency.shape)}")

        conditioning = self._prepare_conditioning(h, z_t.shape)

        hidden = self.input_proj(z_t)
        time_embedding = sinusoidal_timestep_embedding(t, self.hidden_dim)
        hidden = hidden + self.time_mlp(time_embedding).view(batch, 1, 1, self.hidden_dim)
        if conditioning is not None:
            hidden = hidden + self.cond_proj(conditioning)

        for graph_block in self.graph_blocks:
            hidden = graph_block(hidden, adjacency)
        hidden = self.temporal_block(hidden)
        noise = self.output_proj(self.output_norm(hidden))

        assert_rank(noise, 4, "noise")
        assert noise.shape == z_t.shape, "predicted noise must match z_t shape"
        return noise
