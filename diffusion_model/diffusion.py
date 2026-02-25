"""Latent diffusion utilities for forward noising and reverse sampling."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.util import assert_rank


def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """Build a linear beta schedule.

    Args:
        num_steps: Number of diffusion timesteps.
        beta_start: Initial beta.
        beta_end: Final beta.

    Returns:
        Beta tensor with shape [num_steps].
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if not (0.0 < beta_start < 1.0 and 0.0 < beta_end < 1.0):
        raise ValueError("beta_start and beta_end must be in (0, 1)")
    return torch.linspace(beta_start, beta_end, steps=num_steps, dtype=torch.float32)


class LatentDiffusion(nn.Module):
    """DDPM-style latent diffusion process with optional conditioning support."""

    def __init__(
        self,
        num_steps: int = 500,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        clip_x0: bool = True,
    ) -> None:
        """Initialize diffusion buffers.

        Args:
            num_steps: Number of diffusion steps.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
            clip_x0: Whether to clamp reconstructed x0.
        """
        super().__init__()
        self.num_steps = int(num_steps)
        self.clip_x0 = bool(clip_x0)

        betas = linear_beta_schedule(self.num_steps, beta_start=beta_start, beta_end=beta_end)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(1e-12, 1.0))

    @staticmethod
    def _extract(coeff: torch.Tensor, t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Extract timestep-dependent coefficients and broadcast.

        Args:
            coeff: Coefficient table [num_steps].
            t: Timesteps [B].
            target_shape: Target broadcast shape.

        Returns:
            Broadcast coefficients with shape [B, 1, 1, 1].
        """
        assert_rank(t, 1, "t")
        gathered = coeff.gather(0, t)
        return gathered.view(t.shape[0], *([1] * (len(target_shape) - 1)))

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random diffusion timesteps.

        Args:
            batch_size: Batch size.
            device: Output device.

        Returns:
            Random timestep tensor with shape [B].
        """
        return torch.randint(0, self.num_steps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample noisy latent ``z_t`` from clean latent ``z0``.

        Args:
            z0: Clean latent tensor [B, T, J, D].
            t: Timesteps [B].
            noise: Optional Gaussian noise [B, T, J, D].

        Returns:
            Noisy latent tensor [B, T, J, D].
        """
        assert_rank(z0, 4, "z0")
        assert_rank(t, 1, "t")
        if t.shape[0] != z0.shape[0]:
            raise AssertionError("t must have one timestep per batch element")

        if noise is None:
            noise = torch.randn_like(z0)
        if noise.shape != z0.shape:
            raise AssertionError("noise must match z0 shape")

        coeff_a = self._extract(self.sqrt_alpha_cumprod, t, z0.shape)
        coeff_b = self._extract(self.sqrt_one_minus_alpha_cumprod, t, z0.shape)
        return coeff_a * z0 + coeff_b * noise

    def predict_x0_from_noise(self, z_t: torch.Tensor, t: torch.Tensor, noise_hat: torch.Tensor) -> torch.Tensor:
        """Recover clean latent estimate from noisy latent and predicted noise.

        Args:
            z_t: Noisy latent tensor [B, T, J, D].
            t: Timesteps [B].
            noise_hat: Predicted noise [B, T, J, D].

        Returns:
            Clean latent estimate [B, T, J, D].
        """
        assert_rank(z_t, 4, "z_t")
        assert_rank(t, 1, "t")
        if z_t.shape != noise_hat.shape:
            raise AssertionError("z_t and noise_hat must have the same shape")

        coeff_a = self._extract(self.sqrt_alpha_cumprod, t, z_t.shape)
        coeff_b = self._extract(self.sqrt_one_minus_alpha_cumprod, t, z_t.shape)
        z0_hat = (z_t - coeff_b * noise_hat) / coeff_a
        if self.clip_x0:
            z0_hat = z0_hat.clamp(-3.0, 3.0)
        return z0_hat

    def predict_noise_loss(
        self,
        denoiser: nn.Module,
        z0: torch.Tensor,
        adjacency: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute MSE noise prediction loss for diffusion training.

        Args:
            denoiser: Noise predictor module.
            z0: Clean latent target [B, T, J, D].
            adjacency: Adjacency matrix [J, J].
            h: Optional conditioning tensor [B, T, D] or [B, T, J, D].
            t: Optional predefined timesteps [B].
            noise: Optional predefined noise tensor [B, T, J, D].

        Returns:
            Dictionary containing ``loss``, ``t``, ``z_t``, ``noise``, and ``noise_hat``.
        """
        assert_rank(z0, 4, "z0")
        assert_rank(adjacency, 2, "adjacency")

        batch = z0.shape[0]
        if t is None:
            t = self.sample_timesteps(batch, device=z0.device)
        if noise is None:
            noise = torch.randn_like(z0)

        z_t = self.q_sample(z0=z0, t=t, noise=noise)
        noise_hat = denoiser(z_t=z_t, t=t, adjacency=adjacency, h=h)
        if noise_hat.shape != noise.shape:
            raise AssertionError("noise prediction must match noise shape")

        loss = F.mse_loss(noise_hat, noise)
        return {"loss": loss, "t": t, "z_t": z_t, "noise": noise, "noise_hat": noise_hat}

    @torch.no_grad()
    def p_sample(
        self,
        denoiser: nn.Module,
        z_t: torch.Tensor,
        t: torch.Tensor,
        adjacency: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one reverse diffusion step.

        Args:
            denoiser: Noise predictor module.
            z_t: Current noisy latent [B, T, J, D].
            t: Timesteps [B].
            adjacency: Adjacency matrix [J, J].
            h: Optional conditioning tensor.

        Returns:
            Previous latent sample ``z_{t-1}``.
        """
        noise_hat = denoiser(z_t=z_t, t=t, adjacency=adjacency, h=h)
        beta_t = self._extract(self.betas, t, z_t.shape)
        sqrt_one_minus_alpha_t = self._extract(self.sqrt_one_minus_alpha_cumprod, t, z_t.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alpha, t, z_t.shape)

        mean = sqrt_recip_alpha_t * (z_t - (beta_t / sqrt_one_minus_alpha_t) * noise_hat)
        variance = self._extract(self.posterior_variance, t, z_t.shape)
        random_noise = torch.randn_like(z_t)
        nonzero_mask = (t != 0).float().view(t.shape[0], *([1] * (z_t.ndim - 1)))
        return mean + nonzero_mask * torch.sqrt(variance) * random_noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        denoiser: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device,
        adjacency: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Run reverse diffusion from Gaussian noise to latent sample.

        Args:
            denoiser: Noise predictor module.
            shape: Output shape [B, T, J, D].
            device: Target device.
            adjacency: Adjacency matrix [J, J].
            h: Optional conditioning tensor.
            steps: Optional number of reverse steps (defaults to full schedule).

        Returns:
            Sampled clean latent tensor [B, T, J, D].
        """
        if len(shape) != 4:
            raise AssertionError("shape must be rank-4")

        total_steps = self.num_steps if steps is None else int(steps)
        if total_steps <= 0 or total_steps > self.num_steps:
            raise AssertionError("steps must be in [1, num_steps]")

        z_t = torch.randn(*shape, device=device)
        for step in reversed(range(total_steps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            z_t = self.p_sample(denoiser=denoiser, z_t=z_t, t=t, adjacency=adjacency, h=h)
        return z_t
