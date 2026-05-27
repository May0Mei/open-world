"""Minimal PyTorch PixelSAC for diffusion noise prediction.

Mirrors the interface of jaxrl2.agents.pixel_sac.PixelSACLearner so
the RLFineTuneRunner can swap implementations without changing its logic.

Network layout (mirrors dsrl_pi0):
  - Pixel encoder: small ConvNet or ResNet → latent_dim features
  - Actor:  latent → mean + log_std (Tanh squashed Gaussian)
  - Critic: [latent, action] → two Q values (twin SAC)
  - Target critic: EMA copy of critic

References:
  - SAC: https://arxiv.org/abs/1812.05905
  - dsrl_pi0/jaxrl2/agents/pixel_sac/pixel_sac_learner.py
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class _SmallCNN(nn.Module):
    """Simple 4-layer CNN encoder for pixel observations."""

    def __init__(self, in_channels: int, feature_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.feature_dim = feature_dim
        self._out_size: Optional[int] = None

    def _infer_out_size(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            out = self.net(x)
        return int(out.reshape(x.shape[0], -1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        feat = self.net(x).reshape(x.shape[0], -1)
        return feat

    def build_linear_head(self, sample_input: torch.Tensor) -> nn.Linear:
        out_size = self._infer_out_size(sample_input)
        self._out_size = out_size
        return nn.Linear(out_size, self.feature_dim)


class _Actor(nn.Module):
    """Squashed Gaussian actor: obs → (mean, log_std) over noise action."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        encoder: _SmallCNN,
        encoder_head: nn.Linear,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_head = encoder_head
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = F.relu(self.encoder_head(self.encoder(pixels)))
        h = self.net(feat)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(pixels)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)


class _Critic(nn.Module):
    """Twin Q-critic: (obs, action) → (Q1, Q2)."""

    def __init__(
        self,
        encoder: _SmallCNN,
        encoder_head: nn.Linear,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_head = encoder_head
        in_dim = latent_dim + action_dim

        def _mlp():
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(
        self, pixels: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = F.relu(self.encoder_head(self.encoder(pixels)))
        x = torch.cat([feat, action], dim=-1)
        return self.q1(x), self.q2(x)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TorchPixelSACAgent:
    """PyTorch PixelSAC agent with the same interface as jaxrl2's PixelSACLearner.

    Only supports pixel observations (``{'pixels': uint8 ndarray}``).
    """

    def __init__(
        self,
        seed: int,
        sample_obs: Dict[str, np.ndarray],
        sample_action: np.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        tau: float = 0.005,
        discount: float = 0.99,
        init_temperature: float = 1.0,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        device: Optional[str] = None,
        **_: Any,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.tau = tau
        self.discount = discount
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- infer shapes ---
        pixels = sample_obs["pixels"]  # (1, H, W, C, 1) or (H, W, C, 1)
        pix_t = self._preprocess_pixels(torch.from_numpy(pixels[0] if pixels.ndim == 5 else pixels))
        in_channels = pix_t.shape[0]
        action_np = sample_action.reshape(-1) if sample_action.ndim > 1 else sample_action
        action_np = action_np[0] if action_np.shape == sample_action.shape else action_np
        action_dim = int(np.prod(sample_action.shape[1:]) if sample_action.ndim > 1 else sample_action.shape[-1])

        self.action_dim = action_dim

        # --- shared encoder (separate weights for actor / critic) ---
        enc_actor = _SmallCNN(in_channels, latent_dim).to(self.device)
        enc_critic = _SmallCNN(in_channels, latent_dim).to(self.device)

        dummy = pix_t.unsqueeze(0).to(self.device)
        head_actor = enc_actor.build_linear_head(dummy).to(self.device)
        head_critic = enc_critic.build_linear_head(dummy).to(self.device)

        self.actor = _Actor(enc_actor, head_actor, latent_dim, action_dim, hidden_dim).to(self.device)
        self.critic = _Critic(enc_critic, head_critic, latent_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.requires_grad_(False)

        self.log_alpha = torch.tensor(
            np.log(init_temperature), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.target_entropy = -action_dim / 2.0

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    # ------------------------------------------------------------------
    # jaxrl2-compatible public interface
    # ------------------------------------------------------------------

    def sample_actions(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Return a sampled noise action for the given pixel observation."""
        pixels = self._obs_to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(pixels)
        # Reshape to match the expected (1, action_dim) SAC action layout
        return action.squeeze(0).cpu().numpy().reshape(1, self.action_dim)

    def update(self, batch: Any) -> Dict[str, float]:
        """One SAC gradient step on a sampled batch.

        ``batch`` can be a flax FrozenDict or a plain dict with keys:
        observations / next_observations / actions / rewards / masks / discount.
        """
        # --- unpack ---
        obs_pix = self._batch_pixels(batch, "observations")
        next_pix = self._batch_pixels(batch, "next_observations")
        actions = self._to_tensor(batch["actions"]).reshape(-1, self.action_dim)
        rewards = self._to_tensor(batch["rewards"]).unsqueeze(-1)
        masks = self._to_tensor(batch["masks"]).unsqueeze(-1)
        discount = self._to_tensor(batch.get("discount", self.discount))
        if discount.ndim == 0:
            discount = discount.expand_as(rewards)
        else:
            discount = discount.unsqueeze(-1)

        alpha = self.log_alpha.exp().detach()

        # --- critic update ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_pix)
            q1_t, q2_t = self.target_critic(next_pix, next_action)
            target_q = torch.min(q1_t, q2_t) - alpha * next_log_prob
            target_q = rewards + masks * discount * target_q

        q1, q2 = self.critic(obs_pix, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- actor + alpha update ---
        new_action, log_prob = self.actor.sample(obs_pix)
        q1_a, q2_a = self.critic(obs_pix, new_action)
        min_q = torch.min(q1_a, q2_a)
        actor_loss = (alpha * log_prob - min_q).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- soft target update ---
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.lerp_(p.data, self.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "alpha_loss": float(alpha_loss.item()),
            "entropy": float(-log_prob.mean().item()),
        }

    def save_checkpoint(self, output_dir: str, step: int, keep_every_n_steps: int) -> None:
        path = Path(output_dir) / f"checkpoint_{step}.pt"
        torch.save(
            {
                "step": step,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict(),
            },
            path,
        )

    def restore_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_pixels(pixels: torch.Tensor) -> torch.Tensor:
        """Convert (H, W, C, 1) or (H, W, C) uint8 → float32 CHW in [0, 1]."""
        if pixels.ndim == 4:
            pixels = pixels.squeeze(-1)
        if pixels.ndim == 3:
            pixels = pixels.permute(2, 0, 1)
        return pixels.float() / 255.0

    def _obs_to_tensor(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        pix = torch.from_numpy(np.asarray(obs["pixels"]))
        return self._preprocess_pixels(pix).to(self.device)

    def _batch_pixels(self, batch: Any, key: str) -> torch.Tensor:
        obs = batch[key] if isinstance(batch, dict) else getattr(batch, key, batch[key])
        pix_np = np.asarray(obs["pixels"] if isinstance(obs, dict) else obs)
        pix = torch.from_numpy(pix_np)
        # pix shape: (B, H, W, C, 1)
        B = pix.shape[0]
        if pix.ndim == 5:
            pix = pix.squeeze(-1)
        if pix.ndim == 4:
            pix = pix.permute(0, 3, 1, 2)
        return pix.float().to(self.device) / 255.0

    def _to_tensor(self, x: Any) -> torch.Tensor:
        arr = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(arr).to(self.device)
