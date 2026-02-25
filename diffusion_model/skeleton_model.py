"""Stage 1 skeleton latent diffusion wrapper."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.diffusion import LatentDiffusion
from diffusion_model.graph_modules import GraphDecoder, GraphDenoiserMasked, GraphEncoder
from diffusion_model.util import assert_rank


class SkeletonStage1Model(nn.Module):
    """Stage 1 model: graph encoder, latent diffusion denoiser, and graph decoder."""

    def __init__(
        self,
        joint_dim: int = 3,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        diffusion_steps: int = 500,
        use_torch_geometric: bool = False,
    ) -> None:
        """Initialize Stage 1 modules.

        Args:
            joint_dim: Per-joint input coordinate size.
            latent_dim: Latent feature size.
            hidden_dim: Hidden feature size.
            num_layers: Number of graph layers.
            num_heads: Number of attention heads.
            diffusion_steps: Number of diffusion timesteps.
            use_torch_geometric: Whether to enable PyG graph attention.
        """
        super().__init__()
        self.encoder = GraphEncoder(
            joint_dim=joint_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            use_torch_geometric=use_torch_geometric,
        )
        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            joint_dim=joint_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            use_torch_geometric=use_torch_geometric,
        )
        self.denoiser = GraphDenoiserMasked(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            use_torch_geometric=use_torch_geometric,
        )
        self.diffusion = LatentDiffusion(num_steps=diffusion_steps)

    def forward_stage1(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        recon_weight: float = 0.1,
        diffusion_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Run Stage 1 forward pass and losses.

        Args:
            x: Skeleton sequence [B, T, J, 3].
            adjacency: Adjacency matrix [J, J].
            recon_weight: Optional reconstruction loss weight.
            diffusion_weight: Diffusion loss weight.

        Returns:
            Dictionary with latent tensors and scalar losses.
        """
        assert_rank(x, 4, "x")
        assert_rank(adjacency, 2, "adjacency")

        z0 = self.encoder(x, adjacency)
        x_hat = self.decoder(z0, adjacency)
        reconstruction_loss = F.mse_loss(x_hat, x)

        diffusion_out = self.diffusion.predict_noise_loss(
            denoiser=self.denoiser,
            z0=z0,
            adjacency=adjacency,
            h=None,
        )
        diffusion_loss = diffusion_out["loss"]
        total_loss = diffusion_weight * diffusion_loss + recon_weight * reconstruction_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "diffusion_loss": diffusion_loss,
            "x_hat": x_hat,
            "z0": z0,
            "z_t": diffusion_out["z_t"],
            "noise": diffusion_out["noise"],
            "noise_hat": diffusion_out["noise_hat"],
            "t": diffusion_out["t"],
        }
