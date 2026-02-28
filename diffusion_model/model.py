"""
model.py

This file is your Stage 3 model.

Stage 3 does 3 jobs:
1) Diffusion: learn to denoise latent skeletons Z (predict noise)
2) Decode: turn latent skeleton Z into real skeleton coordinates X_hat
3) Classify: use a transformer to predict the activity label from X_hat

✅ Matches dataset.py + earlier edits:
- If you have a batch dict from dataset.py, it contains:
    batch["X"]  -> [B,T,J,3]    (skeleton coords)
    batch["y"]  -> [B]          (label)
    batch["A1"], batch["A2"] -> IMU (if you want conditioning)

- For Stage 3 training we typically already have:
    z0_target = Stage1Encoder(batch["X"], adjacency)  -> [B,T,J,D]
    h = TwoSensorIMUEncoder(batch) -> h_joint [B,T,J,D] or h_seq [B,T,D]

This Stage3Model supports:
- h=None (unconditional)
- h=[B,T,D]  (sequence conditioning)
- h=[B,T,J,D] (joint conditioning)

One important caution (not “wrong”, but you should know)

In your forward pass, classification is performed on:

x_hat decoded from z0_est computed from a single random timestep t

That means classifier is learning on reconstructions coming from various noise levels (indirectly). This can be good regularization, but if you intended classification on clean decoded latents, you’d instead decode z0_target (or decode fully sampled z0). Your current design is valid—just a design choice.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.diffusion import LatentDiffusion
from diffusion_model.graph_modules import GraphDecoder, GraphDenoiserMasked


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
# Transformer classifier (reads decoded skeleton and predicts label)
# ------------------------------------------------------------
class TransformerClassifier(nn.Module):
    """
    This classifier takes a skeleton sequence and predicts a class.

    Input:
      x_hat: [B, T, J, 3]

    Steps:
      1) Flatten joints so each time step is one token: [B,T,J*3]
      2) Add position embeddings (so it knows order of time)
      3) Transformer encoder
      4) Average over time and classify
    """

    def __init__(
        self,
        num_joints: int = 32,       # J
        joint_dim: int = 3,         # xyz
        num_classes: int = 14,      # output classes
        model_dim: int = 256,       # transformer hidden size
        num_layers: int = 4,
        num_heads: int = 8,
        max_time: int = 90,         # max window length
    ) -> None:
        super().__init__()

        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.max_time = max_time

        # Each time step becomes a token of size J*3
        input_dim = num_joints * joint_dim

        # Project that token to transformer model_dim
        self.token_proj = nn.Linear(input_dim, model_dim)

        # Learnable position embedding for time (shape supports up to max_time)
        self.pos_embed = nn.Parameter(torch.randn(1, max_time, model_dim) * 0.01)

        # Transformer encoder layer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        # Stack multiple layers
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Final norm + linear head
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, num_classes)

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        """
        x_hat: [B,T,J,3]
        returns logits: [B,num_classes]
        """
        _check_rank("x_hat", x_hat, 4)

        B, T, J, D = x_hat.shape

        # Check expected skeleton shape
        if J != self.num_joints or D != self.joint_dim:
            raise ValueError(f"Expected joints={self.num_joints}, joint_dim={self.joint_dim}, got {J},{D}")
        if T > self.max_time:
            raise ValueError(f"T={T} is larger than max_time={self.max_time}")

        # Flatten joints into one vector per time step: [B,T,J*3]
        tokens = x_hat.reshape(B, T, J * D)

        # Project + add position embeddings
        h = self.token_proj(tokens) + self.pos_embed[:, :T, :]

        # Transformer encoder
        h = self.encoder(h)

        # Pool over time (mean): [B,model_dim]
        pooled = h.mean(dim=1)

        # Norm + classify
        logits = self.head(self.norm(pooled))

        _check_rank("logits", logits, 2)
        return logits


# ------------------------------------------------------------
# Stage 3 model
# ------------------------------------------------------------
class Stage3Model(nn.Module):
    """
    Stage3Model does:
      - diffusion in latent space (denoiser)
      - decode latent -> skeleton (decoder)
      - classify skeleton (classifier)

    Inputs for training:
      z0_target: [B,T,J,latent_dim]     (this is the "clean" latent from Stage-1 encoder)
      adjacency: [J,J]
      h: optional conditioning (None or [B,T,D] or [B,T,J,D])
      labels: optional [B]
    """

    def __init__(
        self,
        latent_dim: int = 256,     # D (latent per joint)
        joint_dim: int = 3,        # xyz
        num_joints: int = 32,      # J
        num_classes: int = 14,     # K
        hidden_dim: int = 256,
        diffusion_steps: int = 500,
        graph_layers: Optional[int] = None,
        graph_heads: Optional[int] = None,
        classifier_dim: Optional[int] = None,
        classifier_layers: int = 4,
        classifier_heads: int = 8,
        window: int = 90,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        classifier_model_dim: Optional[int] = None,
        use_torch_geometric: bool = False,
    ) -> None:
        super().__init__()
        _ = use_torch_geometric

        self.latent_dim = latent_dim
        self.num_joints = num_joints
        graph_layers = graph_layers if graph_layers is not None else (num_layers if num_layers is not None else 3)
        graph_heads = graph_heads if graph_heads is not None else (num_heads if num_heads is not None else 8)
        classifier_dim = (
            classifier_dim
            if classifier_dim is not None
            else (classifier_model_dim if classifier_model_dim is not None else hidden_dim)
        )

        # Denoiser predicts noise in latent space
        self.denoiser = GraphDenoiserMasked(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=graph_layers,
            num_heads=graph_heads,
        )

        # Decoder turns latent -> skeleton coords
        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            joint_dim=joint_dim,
            num_layers=graph_layers,
            num_heads=graph_heads,
        )

        # Classifier predicts label from decoded skeleton
        self.classifier = TransformerClassifier(
            num_joints=num_joints,
            joint_dim=joint_dim,
            num_classes=num_classes,
            model_dim=classifier_dim,
            num_layers=classifier_layers,
            num_heads=classifier_heads,
            max_time=window,
        )

        # Diffusion utilities (adds noise + sampling)
        self.diffusion = LatentDiffusion(num_steps=diffusion_steps)

    # --------------------------------------------------------
    # Training forward
    # --------------------------------------------------------
    def forward(
        self,
        z0_target: torch.Tensor,
        adjacency: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        diffusion_weight: float = 1.0,
        classifier_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        One forward pass for training.

        1) Diffusion loss: denoiser predicts noise added to z0_target
        2) Reconstruct z0_est from (z_t, predicted noise)
        3) Decode z0_est -> x_hat
        4) Classify x_hat -> logits
        5) If labels provided, compute cross-entropy
        6) total loss = diffusion_weight * diffusion_loss + classifier_weight * cls_loss
        """

        # Basic checks
        _check_rank("z0_target", z0_target, 4)
        _check_last_dim("z0_target", z0_target, self.latent_dim)
        _check_rank("adjacency", adjacency, 2)

        # ---- 1) diffusion training step ----
        diff_out = self.diffusion.predict_noise_loss(
            denoiser=self.denoiser,
            z0=z0_target,
            adjacency=adjacency,
            h=h,  # can be None or conditioning tensor
        )
        diffusion_loss = diff_out["loss"]

        # ---- 2) estimate clean latent z0 from noisy latent and predicted noise ----
        z0_est = self.diffusion.predict_x0_from_noise(
            z_t=diff_out["z_t"],
            t=diff_out["t"],
            noise_hat=diff_out["noise_hat"],
        )

        # ---- 3) decode latent into skeleton coords ----
        x_hat = self.decoder(z0_est, adjacency)

        # ---- 4) classify skeleton ----
        logits = self.classifier(x_hat)

        # ---- 5) classification loss (only if labels exist) ----
        cls_loss = torch.tensor(0.0, device=z0_target.device)

        if labels is not None:
            _check_rank("labels", labels, 1)

            # your dataset uses -1 sometimes for "unknown label"
            # so we only compute CE if all labels are >= 0
            if torch.all(labels >= 0):
                cls_loss = F.cross_entropy(logits, labels)

        # ---- 6) total loss ----
        total_loss = diffusion_weight * diffusion_loss + classifier_weight * cls_loss

        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "classification_loss": cls_loss,
            "z0_est": z0_est,
            "x_hat": x_hat,
            "logits": logits,
            # extra debug outputs (helpful for research)
            "noise_hat": diff_out["noise_hat"],
            "noise": diff_out["noise"],
            "z_t": diff_out["z_t"],
            "t": diff_out["t"],
        }

    def forward_stage3(
        self,
        z0_target: torch.Tensor,
        adjacency: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        diffusion_weight: float = 1.0,
        classifier_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compatibility wrapper used by train.py."""
        return self.forward(
            z0_target=z0_target,
            adjacency=adjacency,
            h=h,
            labels=labels,
            diffusion_weight=diffusion_weight,
            classifier_weight=classifier_weight,
        )

    # --------------------------------------------------------
    # Sampling (conditional diffusion)
    # --------------------------------------------------------
    @torch.no_grad()
    def sample_latent(
        self,
        batch_size: int,
        window: int,
        adjacency: torch.Tensor,
        device: torch.device,
        h: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample a latent sequence using reverse diffusion.

        Returns:
          z0_hat: [B,T,J,D]
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

    @torch.no_grad()
    def sample_skeleton(
        self,
        batch_size: int,
        window: int,
        adjacency: torch.Tensor,
        device: torch.device,
        h: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample skeletons directly:
          latent -> decode -> skeleton

        Returns:
          x_hat: [B,T,J,3]
        """
        z0_hat = self.sample_latent(
            batch_size=batch_size,
            window=window,
            adjacency=adjacency,
            device=device,
            h=h,
            steps=steps,
        )
        return self.decoder(z0_hat, adjacency)

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
        """Compatibility wrapper used by generate.py."""
        return self.sample_latent(
            batch_size=batch_size,
            window=window,
            adjacency=adjacency,
            device=device,
            h=h,
            steps=steps,
        )

    # --------------------------------------------------------
    # Decode / classify helpers
    # --------------------------------------------------------
    def decode(self, z: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Decode latent z [B,T,J,D] -> skeleton [B,T,J,3]."""
        return self.decoder(z, adjacency)

    def classify(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Classify skeleton [B,T,J,3] -> logits [B,K]."""
        return self.classifier(x_hat)
