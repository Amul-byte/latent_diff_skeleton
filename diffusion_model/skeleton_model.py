"""
skeleton_model.py

This is the "Stage 1" skeleton model.

What it does (super simple):
1) Encoder:  takes skeleton X and turns it into latent Z (smaller / easier space)
2) Denoiser: learns diffusion in latent space (predicts noise)
3) Decoder:  turns latent Z back into skeleton X_hat

IMPORTANT (matches your dataset.py):
- dataset returns a dict with key "X" = skeleton window [T, 32, 3]
- DataLoader will make it [B, T, 32, 3]
So this model can accept:
- either x directly: x = [B,T,J,3]
- or a batch dict: batch["X"] = [B,T,J,3]
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.diffusion import LatentDiffusion
from diffusion_model.graph_modules import GraphDecoder, GraphDenoiserMasked, GraphEncoder
from diffusion_model.util import assert_rank


class SkeletonStage1Model(nn.Module):
    """
    Stage 1 model for skeletons.

    Two training modes:
      - "ae"   : autoencoder only (encoder + decoder). Learns to reconstruct X.
      - "diff" : diffusion only (denoiser in latent space). Learns to predict noise in Z.

    Inputs:
      - X: skeleton tensor [B, T, J, 3]
      - adjacency: adjacency matrix [J, J]
    """

    def __init__(
        self,
        joint_dim: int = 3,         # xyz per joint
        latent_dim: int = 256,      # latent feature size per joint
        hidden_dim: int = 256,      # internal size
        num_layers: int = 3,
        num_heads: int = 8,
        diffusion_steps: int = 500,
        use_torch_geometric: bool = False,
    ) -> None:
        super().__init__()

        # Encoder: X -> Z
        self.encoder = GraphEncoder(
            joint_dim=joint_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            use_torch_geometric=use_torch_geometric,
        )

        # Decoder: Z -> X_hat
        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            joint_dim=joint_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            use_torch_geometric=use_torch_geometric,
        )

        # Denoiser: predicts noise in latent diffusion
        self.denoiser = GraphDenoiserMasked(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            use_torch_geometric=use_torch_geometric,
        )

        # Diffusion helper (handles "add noise" and "compute diffusion loss")
        self.diffusion = LatentDiffusion(num_steps=diffusion_steps)

    # ---------------------------------------------------------
    # Small helper: get X from either a batch dict or a tensor
    # ---------------------------------------------------------
    def _get_x(self, x_or_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        If user passes a batch dict from dataset.py, it contains "X".
        If user passes a tensor directly, just return it.
        """
        if isinstance(x_or_batch, dict):
            if "X" not in x_or_batch:
                raise KeyError("Batch dict must contain key 'X'.")
            return x_or_batch["X"]
        return x_or_batch

    # ---------------------------------------------------------
    # Mode 1: Autoencoder (AE)
    # ---------------------------------------------------------
    def forward_ae(
        self,
        x_or_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        adjacency: torch.Tensor,
        recon_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Autoencoder training:
          X -> encoder -> Z -> decoder -> X_hat
          loss = MSE(X_hat, X)

        Returns a dict so training code can log everything.
        """

        # Get the skeleton tensor [B,T,J,3]
        x = self._get_x(x_or_batch)

        # Safety checks (helps catch shape bugs early)
        assert_rank(x, 4, "x")              # must be 4D
        assert_rank(adjacency, 2, "adjacency")  # must be [J,J]

        # Encode skeleton into latent Z
        z0 = self.encoder(x, adjacency)     # [B,T,J,latent_dim] (usually)

        # Decode latent back into skeleton
        x_hat = self.decoder(z0, adjacency) # [B,T,J,3]

        # Reconstruction loss: how close is x_hat to x?
        reconstruction_loss = F.mse_loss(x_hat, x)

        # No diffusion in AE mode, so diffusion loss is 0
        diffusion_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        # Total loss for this mode
        total_loss = recon_weight * reconstruction_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "diffusion_loss": diffusion_loss,
            "x_hat": x_hat,
            "z0": z0,
        }

    # ---------------------------------------------------------
    # Mode 2: Diffusion in latent space (unconditional)
    # ---------------------------------------------------------
    def forward_diffusion_uncond(
        self,
        x_or_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        adjacency: torch.Tensor,
        diffusion_weight: float = 1.0,
        detach_encoder: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Diffusion training (unconditional):
          1) Encode X -> Z0 (latent)
          2) Diffusion adds noise to Z0 at time t -> Zt
          3) Denoiser tries to predict that noise
          4) Loss = MSE(predicted_noise, true_noise)

        We usually detach encoder so only the denoiser learns here.
        """

        # Get skeleton tensor [B,T,J,3]
        x = self._get_x(x_or_batch)

        # Safety checks
        assert_rank(x, 4, "x")
        assert_rank(adjacency, 2, "adjacency")

        # Step 1: get the clean latent Z0
        if detach_encoder:
            # No gradient through encoder (encoder stays fixed in this step)
            with torch.no_grad():
                z0 = self.encoder(x, adjacency)
            z0 = z0.detach()
        else:
            # Allow gradients through encoder (less common for this stage)
            z0 = self.encoder(x, adjacency)

        # Step 2-4: diffusion helper does:
        # - sample timestep t
        # - add noise to z0 -> z_t
        # - run denoiser(z_t, t, adjacency) -> noise_hat
        # - compute loss between noise_hat and true noise
        diffusion_out = self.diffusion.predict_noise_loss(
            denoiser=self.denoiser,
            z0=z0,
            adjacency=adjacency,
            h=None,  # no conditioning in unconditional mode
        )

        diffusion_loss = diffusion_out["loss"]

        # No reconstruction loss in diffusion-only mode
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

    # ---------------------------------------------------------
    # One simple "forward" that selects mode
    # ---------------------------------------------------------
    def forward(
        self,
        x_or_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        adjacency: torch.Tensor,
        mode: str = "ae",
        recon_weight: float = 1.0,
        diffusion_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Call the model in a simple way.

        mode:
          - "ae"   : autoencoder training step
          - "diff" : diffusion training step
        """
        if mode == "ae":
            return self.forward_ae(x_or_batch=x_or_batch, adjacency=adjacency, recon_weight=recon_weight)

        if mode in ("diff", "diffusion_uncond"):
            return self.forward_diffusion_uncond(
                x_or_batch=x_or_batch,
                adjacency=adjacency,
                diffusion_weight=diffusion_weight,
                detach_encoder=True,
            )

        raise ValueError(f"Unsupported mode: {mode}. Use 'ae' or 'diff'.")

    def forward_stage1(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        mode: str = "ae",
        recon_weight: float = 1.0,
        diffusion_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compatibility wrapper used by train.py."""
        return self.forward(
            x_or_batch=x,
            adjacency=adjacency,
            mode=mode,
            recon_weight=recon_weight,
            diffusion_weight=diffusion_weight,
        )

    # ---------------------------------------------------------
    # Sampling: generate skeletons with diffusion (unconditional)
    # ---------------------------------------------------------
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
        """
        Generate fake skeleton sequences with no conditioning.

        Steps:
          1) Start from random noise in latent space
          2) Diffusion sampling produces z0_hat
          3) Decode z0_hat -> x_hat
        """

        # Check adjacency is [J,J]
        assert_rank(adjacency, 2, "adjacency")
        if adjacency.shape != (J, J):
            raise AssertionError(f"adjacency must be [{J},{J}], got {tuple(adjacency.shape)}")

        # The latent tensor shape we want to sample
        # z shape: [B, T, J, latent_dim]
        shape = (batch_size, T, J, self.denoiser.latent_dim)

        # Sample a clean latent z0_hat using diffusion sampling
        z0_hat = self.diffusion.sample(
            denoiser=self.denoiser,
            shape=shape,
            device=device,
            adjacency=adjacency,
            h=None,
            steps=steps,
        )

        # Decode latent -> skeleton
        x_hat = self.decoder(z0_hat, adjacency)

        return {"z0_hat": z0_hat, "x_hat": x_hat}
