"""Diffusion policy for the bridge dataset, mirroring dp_policy.py structure.

Architecture:
  - Small ResNet image encoder (single RGB view, 128x128)
  - MLP proprioceptive state encoder (8-dim bridge state)
  - FiLM-conditioned MLP denoiser with sinusoidal time embedding
  - DDPM training (100 steps) / DDIM inference (10 steps)
  - Action chunking with _pending_actions queue, identical to DPPolicy

Action space:  7-D  [dx, dy, dz, droll, dpitch, dyaw, gripper]
               dims 0-5: Cartesian deltas  (~[-0.3, 0.3])
               dim 6:    binary gripper    {0.0, 1.0}
State space:   8-D  proprioceptive (7 joints + gripper or equivalent)

Trained by scripts/train_bridge_bc.py.
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openworld.policies.base_policy import Policy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_DIM = 7
STATE_DIM = 8
IMAGE_SIZE = 128
VIS_DIM = 256
STATE_ENC_DIM = 64
HIDDEN_DIM = 256
HORIZON = 8            # action chunk length
DDPM_TRAIN_STEPS = 100
DDIM_INFER_STEPS = 10
BETA_START = 1e-4
BETA_END = 2e-2


# ---------------------------------------------------------------------------
# DDPM schedule (cosine-like linear)
# ---------------------------------------------------------------------------

def _make_schedule(T: int, beta_start: float, beta_end: float) -> dict[str, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": alphas_cumprod.sqrt(),
        "sqrt_one_minus_alphas_cumprod": (1.0 - alphas_cumprod).sqrt(),
    }


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class _SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timesteps → (B, dim)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.GroupNorm(min(8, out_ch), out_ch))
            if in_ch != out_ch or stride != 1 else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class _VisualEncoder(nn.Module):
    """ResNet-style CNN: (B, 3, H, W) → (B, VIS_DIM)."""

    def __init__(self, out_dim: int = VIS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),  # 64
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            _ResBlock(32, 64, stride=2),   # 32
            _ResBlock(64, 128, stride=2),  # 16
            _ResBlock(128, 256, stride=2), # 8
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _StateEncoder(nn.Module):
    """MLP: (B, STATE_DIM) → (B, STATE_ENC_DIM)."""

    def __init__(self, in_dim: int = STATE_DIM, out_dim: int = STATE_ENC_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FiLMBlock(nn.Module):
    """One residual MLP block conditioned via FiLM."""

    def __init__(self, hidden: int, cond_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.film = nn.Linear(cond_dim, 2 * hidden)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = self.norm(self.fc(x))
        return F.relu(gamma * h + beta + x, inplace=True)


class _Denoiser(nn.Module):
    """FiLM-conditioned MLP denoiser.

    Input:  noisy actions (B, H*action_dim) + time embedding
    Cond:   visual + state features
    Output: predicted noise (B, H*action_dim)
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        horizon: int = HORIZON,
        hidden: int = HIDDEN_DIM,
        cond_dim: int = VIS_DIM + STATE_ENC_DIM,
        time_emb_dim: int = 64,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.flat_dim = action_dim * horizon
        total_cond = cond_dim + time_emb_dim

        self.time_emb = _SinusoidalTimeEmb(time_emb_dim)
        self.in_proj = nn.Linear(self.flat_dim, hidden)
        self.blocks = nn.ModuleList([
            _FiLMBlock(hidden, total_cond) for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(hidden, self.flat_dim)

    def forward(
        self,
        x_noisy: torch.Tensor,   # (B, H, action_dim)
        t: torch.Tensor,          # (B,) int timestep
        cond: torch.Tensor,       # (B, cond_dim)
    ) -> torch.Tensor:
        B = x_noisy.shape[0]
        x_flat = x_noisy.reshape(B, -1)
        t_emb = self.time_emb(t)
        full_cond = torch.cat([cond, t_emb], dim=-1)
        h = F.relu(self.in_proj(x_flat), inplace=True)
        for block in self.blocks:
            h = block(h, full_cond)
        return self.out_proj(h).reshape(B, -1, x_noisy.shape[-1])


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class _BridgeDPModel(nn.Module):
    """Diffusion policy model for bridge: encode → denoise → action chunk."""

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        state_dim: int = STATE_DIM,
        horizon: int = HORIZON,
        ddpm_T: int = DDPM_TRAIN_STEPS,
        ddim_steps: int = DDIM_INFER_STEPS,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.ddpm_T = ddpm_T
        self.ddim_steps = ddim_steps

        self.vis_enc = _VisualEncoder(VIS_DIM)
        self.state_enc = _StateEncoder(state_dim, STATE_ENC_DIM)
        cond_dim = VIS_DIM + STATE_ENC_DIM
        self.denoiser = _Denoiser(action_dim, horizon, HIDDEN_DIM, cond_dim)

        sched = _make_schedule(ddpm_T, BETA_START, BETA_END)
        for k, v in sched.items():
            self.register_buffer(k, v)

    # ------------------------------------------------------------------
    # Training: predict noise
    # ------------------------------------------------------------------

    def loss(
        self,
        images: torch.Tensor,      # (B, 3, H, W)
        states: torch.Tensor,      # (B, state_dim)
        actions: torch.Tensor,     # (B, horizon, action_dim) — normalised
    ) -> torch.Tensor:
        B = actions.shape[0]
        cond = self._encode(images, states)
        t = torch.randint(0, self.ddpm_T, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        sqrt_a = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        x_noisy = sqrt_a * actions + sqrt_1ma * noise
        pred_noise = self.denoiser(x_noisy, t, cond)
        return F.mse_loss(pred_noise, noise)

    # ------------------------------------------------------------------
    # Inference: DDIM sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(
        self,
        observation: dict[str, Any],
        init_noise: Optional[torch.Tensor] = None,
    ) -> dict[str, np.ndarray]:
        """Run DDIM sampling; return {"actions": (horizon, action_dim)} numpy."""
        images, states = self._prepare_obs(observation)
        cond = self._encode(images, states)
        actions = self._ddim_sample(cond, init_noise=init_noise)
        return {"actions": actions.squeeze(0).cpu().numpy()}

    def _ddim_sample(
        self,
        cond: torch.Tensor,
        init_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = cond.shape[0]
        x = (
            init_noise.to(cond)
            if init_noise is not None
            else torch.randn(B, self.horizon, self.action_dim, device=cond.device)
        )
        # evenly-spaced DDIM timesteps
        step_size = self.ddpm_T // self.ddim_steps
        timesteps = list(reversed(range(0, self.ddpm_T, step_size)))

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=cond.device, dtype=torch.long)
            pred_noise = self.denoiser(x, t, cond)
            ac = self.alphas_cumprod[t].view(B, 1, 1)
            x0_pred = (x - (1 - ac).sqrt() * pred_noise) / ac.sqrt()
            x0_pred = x0_pred.clamp(-3, 3)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                ac_prev = self.alphas_cumprod[t_prev].view(B, 1, 1)
                x = ac_prev.sqrt() * x0_pred + (1 - ac_prev).sqrt() * pred_noise
            else:
                x = x0_pred
        return x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.vis_enc(images), self.state_enc(states)], dim=-1)

    def _prepare_obs(
        self, observation: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        img = _extract_and_preprocess_image(observation, IMAGE_SIZE).to(device)
        state = _extract_state(observation).to(device)
        return img, state


# ---------------------------------------------------------------------------
# Policy class — mirrors dp_policy.py interface exactly
# ---------------------------------------------------------------------------

class BridgeDiffusionPolicy(Policy):
    """Diffusion policy for the bridge dataset.

    Interface mirrors DPPolicy: uses a _pending_actions queue so act() can
    be called once per step while the model runs only every `horizon` steps.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        action_dim: int = ACTION_DIM,
        state_dim: int = STATE_DIM,
        horizon: int = HORIZON,
        ddpm_T: int = DDPM_TRAIN_STEPS,
        ddim_steps: int = DDIM_INFER_STEPS,
        device: str = "cuda",
        **_: Any,
    ):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self.device = device

        self._model = _BridgeDPModel(action_dim, state_dim, horizon, ddpm_T, ddim_steps)
        self._norm_stats: Optional[dict] = None
        self._instruction: Optional[str] = None
        self._pending_actions: list[np.ndarray] = []

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    # ------------------------------------------------------------------
    # Policy interface (mirrors dp_policy.py)
    # ------------------------------------------------------------------

    def reset(self, instruction: Optional[str] = None) -> None:
        self._instruction = instruction
        self._pending_actions = []

    def act(
        self,
        observation: Any,
        state: Any,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        if not self._pending_actions:
            obs_dict = self._build_obs_dict(observation, state)
            result = self._model.to(self.device).eval().infer(obs_dict)
            actions = result["actions"]  # (horizon, action_dim)
            if self._norm_stats is not None:
                actions = self._unnormalise(actions)
            self._pending_actions = [
                actions[i].astype(np.float32) for i in range(len(actions))
            ]
        return self._pending_actions.pop(0)

    def infer_chunk_with_noise(
        self,
        observation: Any,
        state: Any,
        noise: np.ndarray,
        instruction: Optional[str] = None,
    ) -> list[np.ndarray]:
        """DSRL noise injection: use SAC-predicted noise as the DDIM starting point.

        noise: any shape produced by the RL runner — we flatten and take the
        first horizon*action_dim elements, then reshape to (1, horizon, action_dim).
        The runner may send (action_horizon, noise_dim) where noise_dim = horizon*action_dim,
        meaning the same flat vector is tiled; we only need the first row.
        """
        obs_dict = self._build_obs_dict(observation, state)
        device = torch.device(self.device)
        flat = noise.astype(np.float32).reshape(-1)
        needed = self.horizon * self.action_dim
        if flat.size >= needed:
            flat = flat[:needed]
        else:
            flat = np.pad(flat, (0, needed - flat.size))
        init = torch.from_numpy(flat).reshape(1, self.horizon, self.action_dim)
        result = self._model.to(device).eval().infer(obs_dict, init_noise=init.to(device))
        actions = result["actions"]  # (horizon, action_dim)
        if self._norm_stats is not None:
            actions = self._unnormalise(actions)
        return [actions[i].astype(np.float32) for i in range(len(actions))]

    def load_checkpoint(self, checkpoint_path: str) -> None:
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"BridgeDiffusionPolicy checkpoint not found: {p}")
        ckpt = torch.load(str(p), map_location="cpu")
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._norm_stats = ckpt.get("norm_stats")
        logger.info("BridgeDiffusionPolicy loaded from %s", p)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_obs_dict(self, observation: Any, state: Any) -> dict[str, Any]:
        return {"observation": observation, "state": state}

    def _unnormalise(self, actions: np.ndarray) -> np.ndarray:
        mean = self._norm_stats["action_mean"]
        std = self._norm_stats["action_std"]
        return actions * std + mean


# ---------------------------------------------------------------------------
# Observation utilities (used by both model and training script)
# ---------------------------------------------------------------------------

def _extract_and_preprocess_image(
    obs: Any,
    size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Return (1, 3, size, size) float32 tensor in [0, 1]."""
    arr = _extract_raw_image(obs)
    if arr.shape[:2] != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)  # (H, W, 3)
    return t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


def _extract_raw_image(obs: Any) -> np.ndarray:
    """Extract (H, W, 3) uint8 from various observation formats."""
    if isinstance(obs, dict):
        obs_inner = obs.get("observation", obs)
        if isinstance(obs_inner, dict):
            for key in ("exterior_right", "exterior_left", "wrist"):
                if key in obs_inner:
                    return _load_image_value(obs_inner[key])
            for v in obs_inner.values():
                if isinstance(v, (np.ndarray, str)):
                    return _load_image_value(v)
        if "observation" in obs and isinstance(obs["observation"], (np.ndarray, str)):
            return _load_image_value(obs["observation"])
    if isinstance(obs, (np.ndarray, str)):
        return _load_image_value(obs)
    raise ValueError(f"Cannot extract image from observation type {type(obs)}")


def _load_image_value(v: Any) -> np.ndarray:
    if isinstance(v, str):
        import mediapy as _mp
        return np.asarray(_mp.read_image(v), dtype=np.uint8)
    arr = np.asarray(v, dtype=np.uint8)
    if arr.ndim == 3:
        h, w, c = arr.shape
        # stacked views vertically — take top view
        if c >= 3 and h > w and h % 3 == 0:
            arr = arr[: h // 3, :, :3]
        else:
            arr = arr[:, :, :3]
    return arr


def _extract_state(obs: Any) -> torch.Tensor:
    """Return (1, STATE_DIM) float32 tensor from observation/state dict."""
    state_vec = None
    if isinstance(obs, dict):
        inner_state = obs.get("state")
        if inner_state is None:
            inner_state = obs.get("observation", obs)
        if isinstance(inner_state, dict):
            robot = inner_state.get("robot", inner_state)
            if isinstance(robot, dict):
                for key in ("state", "joint_positions", "joint_position"):
                    if key in robot:
                        state_vec = np.asarray(robot[key], dtype=np.float32).reshape(-1)
                        break
        elif inner_state is not None and not isinstance(inner_state, dict):
            state_vec = np.asarray(inner_state, dtype=np.float32).reshape(-1)

    if state_vec is None:
        state_vec = np.zeros(STATE_DIM, dtype=np.float32)
    # pad or truncate to STATE_DIM
    if len(state_vec) < STATE_DIM:
        state_vec = np.pad(state_vec, (0, STATE_DIM - len(state_vec)))
    else:
        state_vec = state_vec[:STATE_DIM]
    return torch.from_numpy(state_vec).unsqueeze(0)  # (1, STATE_DIM)


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

BridgeBCPolicy = BridgeDiffusionPolicy
