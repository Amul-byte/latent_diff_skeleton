"""Stage 1 skeleton latent diffusion wrapper."""

from __future__ import annotations

import inspect
from typing import Dict, Optional

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
        if "t" not in inspect.signature(self.denoiser.forward).parameters:
            raise TypeError("GraphDenoiserMasked.forward must accept timestep argument 't'")
        self.diffusion = LatentDiffusion(num_steps=diffusion_steps)

    def forward_ae(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        recon_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Stage 1A: pretrain skeleton autoencoder (E + D only)."""
        assert_rank(x, 4, "x")
        assert_rank(adjacency, 2, "adjacency")

        z0 = self.encoder(x, adjacency)
        x_hat = self.decoder(z0, adjacency)
        reconstruction_loss = F.mse_loss(x_hat, x)
        diffusion_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        total_loss = recon_weight * reconstruction_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "diffusion_loss": diffusion_loss,
            "x_hat": x_hat,
            "z0": z0,
        }

    def forward_diffusion_uncond(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        diffusion_weight: float = 1.0,
        detach_encoder: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Stage 1B: pretrain unconditional latent diffusion (denoiser only)."""
        assert_rank(x, 4, "x")
        assert_rank(adjacency, 2, "adjacency")

        if detach_encoder:
            # Detach z0 so encoder/decoder are not part of diffusion gradients.
            with torch.no_grad():
                z0 = self.encoder(x, adjacency)
            z0 = z0.detach()
        else:
            z0 = self.encoder(x, adjacency)

        diffusion_out = self.diffusion.predict_noise_loss(
            denoiser=self.denoiser,
            z0=z0,
            adjacency=adjacency,
            h=None,
        )
        diffusion_loss = diffusion_out["loss"]
        reconstruction_loss = torch.zeros((), device=z0.device, dtype=z0.dtype)
        total_loss = diffusion_weight * diffusion_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "diffusion_loss": diffusion_loss,
            "z0": z0,
            "z_t": diffusion_out["z_t"],
            "noise": diffusion_out["noise"],
            "noise_hat": diffusion_out["noise_hat"],
            "t": diffusion_out["t"],
        }

    def forward_stage1(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        mode: str = "ae",
        recon_weight: float = 0.1,
        diffusion_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Optional Stage-1 wrapper for ``ae`` and ``diff`` modes.

        Args:
            x: Skeleton sequence [B, T, J, 3].
            adjacency: Adjacency matrix [J, J].
            mode: One of ``\"ae\"`` or ``\"diff\"``.
            recon_weight: Reconstruction loss weight.
            diffusion_weight: Diffusion loss weight.
        """
        if mode == "ae":
            return self.forward_ae(x=x, adjacency=adjacency, recon_weight=recon_weight)
        if mode in {"diff", "diffusion_uncond"}:
            return self.forward_diffusion_uncond(
                x=x,
                adjacency=adjacency,
                diffusion_weight=diffusion_weight,
                detach_encoder=True,
            )
        raise ValueError(f"Unsupported stage1 mode: {mode}. Use 'ae' or 'diff'.")

    @torch.no_grad()
    def sample_uncond(
        self,
        adjacency: torch.Tensor,
        batch_size: int,
        T: int,
        J: int,
        device: torch.device,
        steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample Stage-1 unconditional latents and decode to skeletons."""
        assert_rank(adjacency, 2, "adjacency")
        if adjacency.shape != (J, J):
            raise AssertionError(f"adjacency must be [{J}, {J}], got {tuple(adjacency.shape)}")
        shape = (batch_size, T, J, self.denoiser.latent_dim)
        z0_hat = self.diffusion.sample(
            denoiser=self.denoiser,
            shape=shape,
            device=device,
            adjacency=adjacency,
            h=None,
            steps=steps,
        )
        x_hat = self.decoder(z0_hat, adjacency)
        return {"z0_hat": z0_hat, "x_hat": x_hat}
