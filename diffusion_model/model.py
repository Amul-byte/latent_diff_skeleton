"""Stage 3 conditional diffusion model with decoder and transformer classifier."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.diffusion import LatentDiffusion
from diffusion_model.graph_modules import GraphDecoder, GraphDenoiserMasked
from diffusion_model.util import assert_last_dim, assert_rank


class TransformerClassifier(nn.Module):
    """Transformer classifier over decoded skeleton sequences."""

    def __init__(
        self,
        num_joints: int = 32,
        joint_dim: int = 3,
        num_classes: int = 14,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_time: int = 90,
    ) -> None:
        """Initialize transformer classifier.

        Args:
            num_joints: Number of joints in skeleton sequence.
            joint_dim: Coordinate dimension per joint.
            num_classes: Number of target classes.
            model_dim: Transformer hidden dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            max_time: Maximum supported sequence length.
        """
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.max_time = max_time

        input_dim = num_joints * joint_dim
        self.token_proj = nn.Linear(input_dim, model_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, max_time, model_dim) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, num_classes)

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Classify decoded skeleton sequences.

        Args:
            x_hat: Decoded skeleton tensor [B, T, J, 3].

        Returns:
            Class logits [B, num_classes].
        """
        assert_rank(x_hat, 4, "x_hat")
        batch, time, joints, joint_dim = x_hat.shape
        if joints != self.num_joints or joint_dim != self.joint_dim:
            raise AssertionError(
                f"x_hat expected joints={self.num_joints}, joint_dim={self.joint_dim}, got {joints}, {joint_dim}"
            )
        if time > self.max_time:
            raise AssertionError(f"sequence length {time} exceeds max_time {self.max_time}")

        tokens = x_hat.reshape(batch, time, joints * joint_dim)
        hidden = self.token_proj(tokens) + self.position_embedding[:, :time, :]
        hidden = self.encoder(hidden)
        pooled = self.norm(hidden.mean(dim=1))
        logits = self.head(pooled)
        assert_rank(logits, 2, "logits")
        return logits


class Stage3Model(nn.Module):
    """Stage 3 model for conditional latent diffusion, decoding, and classification."""

    def __init__(
        self,
        latent_dim: int = 256,
        joint_dim: int = 3,
        num_joints: int = 32,
        num_classes: int = 14,
        hidden_dim: int = 256,
        diffusion_steps: int = 500,
        num_layers: int = 3,
        num_heads: int = 8,
        classifier_model_dim: int = 256,
        classifier_layers: int = 4,
        classifier_heads: int = 8,
        window: int = 90,
        use_torch_geometric: bool = False,
    ) -> None:
        """Initialize Stage 3 sub-modules.

        Args:
            latent_dim: Latent feature size.
            joint_dim: Joint coordinate dimension.
            num_joints: Number of joints.
            num_classes: Number of classification targets.
            hidden_dim: Hidden feature size for graph modules.
            diffusion_steps: Number of diffusion timesteps.
            num_layers: Number of graph attention layers.
            num_heads: Graph attention heads.
            classifier_model_dim: Classifier transformer hidden dimension.
            classifier_layers: Number of transformer layers.
            classifier_heads: Number of transformer attention heads.
            window: Maximum sequence length.
            use_torch_geometric: Whether to use PyG GAT path.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints

        self.denoiser = GraphDenoiserMasked(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
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
        self.classifier = TransformerClassifier(
            num_joints=num_joints,
            joint_dim=joint_dim,
            num_classes=num_classes,
            model_dim=classifier_model_dim,
            num_layers=classifier_layers,
            num_heads=classifier_heads,
            max_time=window,
        )
        self.diffusion = LatentDiffusion(num_steps=diffusion_steps)

    def forward_stage3(
        self,
        z0_target: torch.Tensor,
        adjacency: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        diffusion_weight: float = 1.0,
        classifier_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Run Stage 3 training forward pass.

        Args:
            z0_target: Clean latent target [B, T, J, D].
            adjacency: Adjacency matrix [J, J].
            h: Optional conditioning tensor [B, T, D] or [B, T, J, D].
            labels: Optional labels [B].
            diffusion_weight: Diffusion objective weight.
            classifier_weight: Classification objective weight.

        Returns:
            Dictionary containing losses, intermediate tensors, and logits.
        """
        assert_rank(z0_target, 4, "z0_target")
        assert_last_dim(z0_target, self.latent_dim, "z0_target")
        assert_rank(adjacency, 2, "adjacency")

        diffusion_out = self.diffusion.predict_noise_loss(
            denoiser=self.denoiser,
            z0=z0_target,
            adjacency=adjacency,
            h=h,
        )
        diffusion_loss = diffusion_out["loss"]
        z0_est = self.diffusion.predict_x0_from_noise(
            z_t=diffusion_out["z_t"],
            t=diffusion_out["t"],
            noise_hat=diffusion_out["noise_hat"],
        )
        x_hat = self.decoder(z0_est, adjacency)
        logits = self.classifier(x_hat)

        cls_loss = torch.tensor(0.0, device=z0_target.device)
        if labels is not None:
            assert_rank(labels, 1, "labels")
            if torch.all(labels >= 0):
                cls_loss = F.cross_entropy(logits, labels)

        total_loss = diffusion_weight * diffusion_loss + classifier_weight * cls_loss
        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "classification_loss": cls_loss,
            "z0_est": z0_est,
            "x_hat": x_hat,
            "logits": logits,
            "noise_hat": diffusion_out["noise_hat"],
            "noise": diffusion_out["noise"],
            "z_t": diffusion_out["z_t"],
            "t": diffusion_out["t"],
        }

    @torch.no_grad()
    def sample_conditional(
        self,
        batch_size: int,
        window: int,
        adjacency: torch.Tensor,
        device: torch.device,
        h: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample latent trajectories via conditional reverse diffusion.

        Args:
            batch_size: Number of samples.
            window: Sequence length.
            adjacency: Adjacency matrix [J, J].
            device: Device for sampling.
            h: Optional conditioning tensor.
            steps: Optional reverse diffusion steps.

        Returns:
            Sampled latent tensor [B, T, J, D].
        """
        shape = (batch_size, window, self.num_joints, self.latent_dim)
        return self.diffusion.sample(
            denoiser=self.denoiser,
            shape=shape,
            device=device,
            adjacency=adjacency,
            h=h,
            steps=steps,
        )

    def decode(self, latent: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Decode latent trajectories to skeleton coordinates.

        Args:
            latent: Latent tensor [B, T, J, D].
            adjacency: Adjacency matrix [J, J].

        Returns:
            Skeleton tensor [B, T, J, 3].
        """
        return self.decoder(latent, adjacency)

    def classify(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Classify decoded skeleton sequences.

        Args:
            x_hat: Decoded skeleton tensor [B, T, J, 3].

        Returns:
            Class logits [B, K].
        """
        return self.classifier(x_hat)
