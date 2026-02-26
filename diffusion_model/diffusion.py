"""
diffusion.py

This file implements the "diffusion process" in latent space.

✅ Very simple idea:
1) We start with clean latent z0  (from the skeleton encoder)
2) We pick a random time step t
3) We add noise to z0 -> z_t
4) The denoiser tries to guess the noise that was added
5) Loss = MSE(guessed_noise, real_noise)

✅ Matches your other files:
- dataset.py gives skeleton windows and IMU windows
- skeleton_model.py produces z0 = encoder(X)
- imu_encoder.py produces h_seq or h_joint for conditioning
- This file supports h=None (unconditional) OR h=some tensor (conditional)

Shapes we use:
- z0:      [B, T, J, D]
- z_t:     [B, T, J, D]
- noise:   [B, T, J, D]
- noise_hat (prediction): [B, T, J, D]
- t:       [B] (one timestep per batch item)

Important:
- denoiser.forward MUST accept: (z_t, t, adjacency, h)
"""

from __future__ import annotations

import inspect
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Small helper: easy shape checks (instead of fancy utilities)
# ------------------------------------------------------------
def _check_rank(name: str, x: torch.Tensor, rank: int) -> None:
    """Make sure tensor x has exactly `rank` dimensions."""
    if x.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape {tuple(x.shape)}")


# ------------------------------------------------------------
# 1) Beta schedule (how fast we add noise)
# ------------------------------------------------------------
def linear_beta_schedule(
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    """
    Build a simple linear schedule for betas.

    betas are small numbers that control noise amount at each step.
    """

    # safety check
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    # beta values should be between 0 and 1
    if not (0.0 < beta_start < 1.0 and 0.0 < beta_end < 1.0):
        raise ValueError("beta_start and beta_end must be in (0, 1)")

    # make linearly spaced tensor: [beta_start, ..., beta_end]
    return torch.linspace(beta_start, beta_end, steps=num_steps, dtype=torch.float32)


# ------------------------------------------------------------
# 2) The diffusion class
# ------------------------------------------------------------
class LatentDiffusion(nn.Module):
    """
    This class knows how to:
    - add noise forward (q_sample)
    - compute training loss (predict_noise_loss)
    - remove noise to sample (p_sample_loop / sample)
    """

    def __init__(
        self,
        num_steps: int = 500,       # total diffusion steps
        beta_start: float = 1e-4,   # first beta value
        beta_end: float = 2e-2,     # last beta value
        clip_x0: bool = True,      # clamp predicted x0 to avoid explosions
    ) -> None:
        super().__init__()

        # store settings
        self.num_steps = int(num_steps)
        self.clip_x0 = bool(clip_x0)

        # build betas: [num_steps]
        betas = linear_beta_schedule(self.num_steps, beta_start=beta_start, beta_end=beta_end)

        # alphas = 1 - beta
        alphas = 1.0 - betas

        # alpha_cumprod[t] = alpha_0 * alpha_1 * ... * alpha_t
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        # alpha_cumprod_prev[t] = alpha_cumprod[t-1], with alpha_cumprod_prev[0] = 1
        alpha_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_cumprod[:-1]], dim=0)

        # register buffers => saved in model, moves with device, but not trained
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)

        # useful precomputed terms
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / alphas))

        # variance used in reverse process
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(1e-12, 1.0))

    # --------------------------------------------------------
    # Helper: check denoiser signature
    # --------------------------------------------------------
    @staticmethod
    def _validate_denoiser_accepts_timestep(denoiser: nn.Module) -> None:
        """
        We REQUIRE the denoiser forward to have argument 't'
        so we can pass timestep in.
        """
        try:
            sig = inspect.signature(denoiser.forward)
        except (TypeError, ValueError) as exc:
            raise TypeError("Unable to inspect denoiser.forward signature") from exc

        if "t" not in sig.parameters:
            raise TypeError("denoiser.forward must define a timestep argument named 't'")

    # --------------------------------------------------------
    # Helper: call denoiser in a safe way
    # --------------------------------------------------------
    def _call_denoiser(
        self,
        denoiser: nn.Module,
        z_t: torch.Tensor,
        t: torch.Tensor,
        adjacency: torch.Tensor,
        h: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Call denoiser like:
          denoiser(z_t=z_t, t=t, adjacency=adjacency, h=h)
        """
        self._validate_denoiser_accepts_timestep(denoiser)
        return denoiser(z_t=z_t, t=t, adjacency=adjacency, h=h)

    # --------------------------------------------------------
    # Helper: get coefficients for each batch timestep
    # --------------------------------------------------------
    @staticmethod
    def _extract(coeff_table: torch.Tensor, t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        coeff_table is size [num_steps]
        t is size [B]

        We want to select coeff_table[t[i]] for each i in batch,
        then reshape so it can multiply z tensors like [B,T,J,D].
        """
        _check_rank("t", t, 1)

        # gather values at indices t
        gathered = coeff_table.gather(0, t)  # shape [B]

        # reshape to [B, 1, 1, 1] (same rank as z tensors)
        return gathered.view(t.shape[0], *([1] * (len(target_shape) - 1)))

    # --------------------------------------------------------
    # Pick random timesteps
    # --------------------------------------------------------
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Return random integer timesteps t in [0, num_steps-1]
        Shape: [B]
        """
        return torch.randint(0, self.num_steps, (batch_size,), device=device, dtype=torch.long)

    # --------------------------------------------------------
    # Forward diffusion: add noise
    # --------------------------------------------------------
    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create z_t from z0:
            z_t = sqrt(alpha_bar_t)*z0 + sqrt(1-alpha_bar_t)*noise

        z0 shape: [B,T,J,D]
        t shape:  [B]
        """

        _check_rank("z0", z0, 4)
        _check_rank("t", t, 1)

        # t must have one value per batch item
        if t.shape[0] != z0.shape[0]:
            raise ValueError("t must have one timestep per batch element")

        # if noise not given, create random noise
        if noise is None:
            noise = torch.randn_like(z0)

        # noise must match z0 shape
        if noise.shape != z0.shape:
            raise ValueError("noise must match z0 shape")

        # get coefficients for each batch item
        coeff_a = self._extract(self.sqrt_alpha_cumprod, t, z0.shape)
        coeff_b = self._extract(self.sqrt_one_minus_alpha_cumprod, t, z0.shape)

        # return z_t
        return coeff_a * z0 + coeff_b * noise

    # --------------------------------------------------------
    # Recover z0 from z_t and predicted noise
    # --------------------------------------------------------
    def predict_x0_from_noise(self, z_t: torch.Tensor, t: torch.Tensor, noise_hat: torch.Tensor) -> torch.Tensor:
        """
        Estimate z0 from z_t:
            z0_hat = (z_t - sqrt(1-alpha_bar_t)*noise_hat) / sqrt(alpha_bar_t)
        """

        _check_rank("z_t", z_t, 4)
        _check_rank("t", t, 1)

        if z_t.shape != noise_hat.shape:
            raise ValueError("z_t and noise_hat must have same shape")

        coeff_a = self._extract(self.sqrt_alpha_cumprod, t, z_t.shape)
        coeff_b = self._extract(self.sqrt_one_minus_alpha_cumprod, t, z_t.shape)

        z0_hat = (z_t - coeff_b * noise_hat) / coeff_a

        # optional clamp
        if self.clip_x0:
            z0_hat = z0_hat.clamp(-3.0, 3.0)

        return z0_hat

    # --------------------------------------------------------
    # Training loss: denoiser predicts noise
    # --------------------------------------------------------
    def predict_noise_loss(
        self,
        denoiser: nn.Module,
        z0: torch.Tensor,
        adjacency: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss.

        Steps:
          1) pick timestep t
          2) make noise
          3) make z_t = noisy(z0, t, noise)
          4) denoiser predicts noise_hat from (z_t, t, adjacency, h)
          5) loss = MSE(noise_hat, noise)
        """

        _check_rank("z0", z0, 4)
        _check_rank("adjacency", adjacency, 2)

        B = z0.shape[0]

        # sample t if not provided
        if t is None:
            t = self.sample_timesteps(B, device=z0.device)

        # sample noise if not provided
        if noise is None:
            noise = torch.randn_like(z0)

        # make noisy latent
        z_t = self.q_sample(z0=z0, t=t, noise=noise)

        # predict noise
        noise_hat = self._call_denoiser(denoiser=denoiser, z_t=z_t, t=t, adjacency=adjacency, h=h)

        # sanity check
        if noise_hat.shape != noise.shape:
            raise ValueError("noise_hat must have same shape as noise")

        # compute MSE loss
        loss = F.mse_loss(noise_hat, noise)

        return {"loss": loss, "t": t, "z_t": z_t, "noise": noise, "noise_hat": noise_hat}

    # --------------------------------------------------------
    # Reverse diffusion: one step
    # --------------------------------------------------------
    @torch.no_grad()
    def p_sample(
        self,
        denoiser: nn.Module,
        z_t: torch.Tensor,
        t: torch.Tensor,
        adjacency: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Do ONE reverse step: z_t -> z_{t-1}
        """

        # predict noise at this timestep
        noise_hat = self._call_denoiser(denoiser=denoiser, z_t=z_t, t=t, adjacency=adjacency, h=h)

        # pull out coefficients for this timestep
        beta_t = self._extract(self.betas, t, z_t.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_cumprod, t, z_t.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alpha, t, z_t.shape)

        # compute mean of reverse distribution
        mean = sqrt_recip_alpha_t * (z_t - (beta_t / sqrt_one_minus_alpha_bar_t) * noise_hat)

        # compute variance
        variance = self._extract(self.posterior_variance, t, z_t.shape)

        # random noise for sampling
        random_noise = torch.randn_like(z_t)

        # no noise when t==0 (final step)
        nonzero_mask = (t != 0).float().view(t.shape[0], *([1] * (z_t.ndim - 1)))

        return mean + nonzero_mask * torch.sqrt(variance) * random_noise

    # --------------------------------------------------------
    # Reverse diffusion loop: many steps
    # --------------------------------------------------------
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
        """
        Start from random noise and run reverse steps until we get z0.

        shape is [B,T,J,D]
        """

        if len(shape) != 4:
            raise ValueError("shape must be rank-4 like (B,T,J,D)")

        # decide which timesteps to use
        if steps is None:
            # full schedule: num_steps-1 down to 0
            timesteps = torch.arange(self.num_steps - 1, -1, -1, device=device, dtype=torch.long)
        else:
            # fewer steps (faster sampling)
            total_steps = int(steps)
            if total_steps <= 0 or total_steps > self.num_steps:
                raise ValueError("steps must be in [1, num_steps]")

            # create a list of timesteps that spans from max -> 0
            timesteps = torch.linspace(self.num_steps - 1, 0, steps=total_steps, device=device)
            timesteps = timesteps.round().long()
            timesteps = torch.unique_consecutive(timesteps)

            # ensure we start at the last step and end at 0
            if timesteps[0].item() != self.num_steps - 1:
                timesteps = torch.cat([torch.tensor([self.num_steps - 1], device=device), timesteps])
            if timesteps[-1].item() != 0:
                timesteps = torch.cat([timesteps, torch.tensor([0], device=device)])

        # start from pure noise
        z_t = torch.randn(*shape, device=device)

        # loop over timesteps
        for step in timesteps:
            # make t tensor shape [B] filled with this step
            t = torch.full((shape[0],), int(step.item()), device=device, dtype=torch.long)

            # one reverse step
            z_t = self.p_sample(denoiser=denoiser, z_t=z_t, t=t, adjacency=adjacency, h=h)

        return z_t

    # --------------------------------------------------------
    # Public sampling function
    # --------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device,
        adjacency: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Wrapper that calls p_sample_loop.
        """
        return self.p_sample_loop(
            denoiser=denoiser,
            shape=shape,
            device=device,
            adjacency=adjacency,
            h=h,
            steps=steps,
        )