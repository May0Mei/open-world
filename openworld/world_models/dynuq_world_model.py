"""DynUQ world model adapter for OpenWorld.

Wraps the ac_video_model_uq DiT+Diffusion model as a WorldModel. This model
predicts future RGB frames given a context frame and action chunk, and
optionally predicts a per-frame confidence score (how reliable the prediction
is). Confidence is used to filter uncertain trajectories during RL finetuning.

World model checkpoint:
  /n/fs/wm-uq/projects/wm_dynamics_uq/ac_video_model_uq/dynuq/scripts/outputs/
    checkpoints/bridge/20251105_044251_train_bridge_256_conf_pred/ckpt_000050000.pt

Repo path: /n/fs/not-fmrl/Projects/apple_project/ac_video_model_uq
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from openworld.world_models.base_world_model import WorldModel

logger = logging.getLogger(__name__)

_DEFAULT_REPO = "/n/fs/not-fmrl/Projects/apple_project/ac_video_model_uq"
_DEFAULT_CKPT = (
    "/n/fs/wm-uq/projects/wm_dynamics_uq/ac_video_model_uq/dynuq/scripts/outputs/"
    "checkpoints/bridge/20251105_044251_train_bridge_256_conf_pred/ckpt_000050000.pt"
)
_DEFAULT_CFG = (
    "/n/fs/wm-uq/projects/wm_dynamics_uq/ac_video_model_uq/dynuq/scripts/outputs/"
    "checkpoints/bridge/20251105_044251_train_bridge_256_conf_pred/config.yaml"
)
_DEFAULT_REF_LATENT_DIR = (
    "/n/fs/wm-uq/projects/wm_dynamics_uq/ac_video_model_uq/dynuq/scripts/gt_latent/single_frame"
)


@dataclass
class DynUQConfig:
    repo_path: str = _DEFAULT_REPO
    checkpoint_path: str = _DEFAULT_CKPT
    model_config_path: str = _DEFAULT_CFG
    ref_latent_dir: str = _DEFAULT_REF_LATENT_DIR

    # model arch (defaults match 20251105 bridge checkpoint config)
    input_h: int = 256
    input_w: int = 256
    action_dim: int = 7
    patch_size: int = 2
    model_dim: int = 512
    layers: int = 49
    heads: int = 4
    vae_type: str = "sd"
    timesteps: int = 1000
    sampling_timesteps: int = 10
    enable_confidence_prediction: bool = True
    conf_err_threshold: float = 0.85

    use_ema: bool = True
    device: Optional[str] = None

    # confidence threshold for filtering trajectories (None = no filtering)
    # mean raw_conf_pred must exceed this to be considered confident
    conf_filter_threshold: Optional[float] = None


def _ensure_dynuq_on_path(repo_path: str) -> None:
    """Add dynuq package directories to sys.path if needed."""
    repo = Path(repo_path)
    for subdir in (repo, repo / "dynuq" / "models", repo / "dynuq" / "tokenizers"):
        p = str(subdir)
        if p not in sys.path:
            sys.path.insert(0, p)
    # install package root so `from dynuq.xxx import yyy` works
    pkg_root = str(repo)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def _load_config(config_path: str):
    try:
        from omegaconf import OmegaConf
        return OmegaConf.load(config_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dynuq config from {config_path}: {e}") from e


class DynUQWorldModel(WorldModel):
    """Action-conditioned video world model with confidence prediction.

    On each rollout the model takes a single RGB context frame + an action
    chunk and generates `len(action_chunk)` predicted RGB frames plus an
    optional per-frame confidence score in [0, 1] (higher = more certain).

    State layout:
        {"context_frame": np.ndarray (H, W, 3) uint8}
    """

    def __init__(self, **kwargs: Any):
        cfg_dict = {k: v for k, v in kwargs.items() if k in DynUQConfig.__dataclass_fields__}
        self.cfg = DynUQConfig(**cfg_dict)
        self._model = None
        self._diffusion = None
        self._vae = None
        self._loaded = False

    # ------------------------------------------------------------------
    # WorldModel interface
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self.cfg.checkpoint_path = checkpoint_path
        self._load_models()

    def rollout(
        self,
        state: Any,
        observation: Any,
        action_chunk: Any,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate future frames given current observation and action chunk.

        Args:
            state: Dict with optional ``context_frame`` key (H×W×3 uint8).
                   Ignored — we use observation directly.
            observation: Current visual observation. Accepted forms:
                - np.ndarray (H, W, 3) uint8
                - np.ndarray (H*N, W, 3) stacked views — top view extracted
                - dict with view keys (exterior_right, etc.)
                - str path to image file
            action_chunk: (T, 7) array of actions.
            instruction: Unused.

        Returns:
            Dict with:
                "frames": list of T decoded RGB np.ndarray (H, W, 3) uint8
                "next_state": {"context_frame": last predicted frame}
                "confidence": float in [0, 1], mean model confidence (1 = certain)
                "confidence_frames": list of T per-frame confidence scalars
        """
        if not self._loaded:
            self._load_models()

        frame = self._extract_frame(observation)
        actions = self._prepare_actions(action_chunk)
        frames, confidence, confidence_frames = self._generate(frame, actions)

        last_frame = frames[-1] if frames else frame
        return {
            "frames": frames,
            "next_state": {"context_frame": last_frame},
            "confidence": confidence,
            "confidence_frames": confidence_frames,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        _ensure_dynuq_on_path(self.cfg.repo_path)

        device_str = self.cfg.device
        if device_str is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_str)

        logger.info("Loading dynuq model on %s ...", self._device)

        # read model config from YAML to get latent_channels
        model_cfg = _load_config(self.cfg.model_config_path)
        latent_channels = 16  # SD3 VAE

        from dynuq.models.model import DiT
        from dynuq.models.diffusion import Diffusion
        from dynuq.tokenizers.vae import StableDiffusionVAE

        self._model = DiT(
            in_channels=latent_channels,
            patch_size=self.cfg.patch_size,
            dim=self.cfg.model_dim,
            num_layers=self.cfg.layers,
            num_heads=self.cfg.heads,
            action_dim=self.cfg.action_dim,
            max_frames=getattr(model_cfg, "n_frames", 10),
            enable_confidence_pred=self.cfg.enable_confidence_prediction,
        ).to(self._device)

        self._diffusion = Diffusion(
            timesteps=self.cfg.timesteps,
            sampling_timesteps=self.cfg.sampling_timesteps,
            enable_confidence_pred=self.cfg.enable_confidence_prediction,
            device=self._device,
            ref_latent_dir=self.cfg.ref_latent_dir,
            default_err_threshold_conf_pred=self.cfg.conf_err_threshold,
        ).to(self._device)

        logger.info("Loading StableDiffusionVAE ...")
        self._vae = StableDiffusionVAE().to(self._device)

        # load checkpoint
        ckpt_path = Path(self.cfg.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"DynUQ checkpoint not found: {ckpt_path}")
        logger.info("Loading checkpoint from %s", ckpt_path)
        ckpt = torch.load(str(ckpt_path), map_location=self._device)

        if self.cfg.use_ema and "ema" in ckpt:
            logger.info("Using EMA weights")
            self._model.load_state_dict(ckpt["ema"])
        else:
            self._model.load_state_dict(ckpt["model"])

        self._model.eval()
        self._loaded = True
        logger.info("DynUQ world model ready")

    def _extract_frame(self, observation: Any) -> np.ndarray:
        """Extract a single (H, W, 3) uint8 frame from observation."""
        if isinstance(observation, dict):
            views = observation.get("views", observation)
            for key in ("exterior_right", "exterior_left", "wrist"):
                if key in views:
                    val = views[key]
                    if isinstance(val, str):
                        import mediapy as _mp
                        return np.asarray(_mp.read_image(val), dtype=np.uint8)
                    return np.asarray(val, dtype=np.uint8)
            # fallback: first value
            for val in views.values():
                return np.asarray(val, dtype=np.uint8)
        if isinstance(observation, str):
            import mediapy as _mp
            return np.asarray(_mp.read_image(observation), dtype=np.uint8)
        arr = np.asarray(observation, dtype=np.uint8)
        if arr.ndim == 3:
            h, w, c = arr.shape
            if c >= 3 and h != w and h % 3 == 0:
                # stacked views — take top view
                arr = arr[: h // 3, :, :3]
            else:
                arr = arr[:, :, :3]
        return arr

    def _prepare_actions(self, action_chunk: Any) -> torch.Tensor:
        """Convert action chunk to (1, T, action_dim) float32 tensor."""
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        # ensure shape (T, action_dim)
        if a.shape[-1] != self.cfg.action_dim:
            logger.warning(
                "Action dim mismatch: got %d expected %d. Padding/truncating.",
                a.shape[-1], self.cfg.action_dim,
            )
            padded = np.zeros((a.shape[0], self.cfg.action_dim), dtype=np.float32)
            d = min(a.shape[-1], self.cfg.action_dim)
            padded[:, :d] = a[:, :d]
            a = padded
        return torch.from_numpy(a).unsqueeze(0).to(self._device)  # (1, T, 7)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to model input size (H, W)."""
        import cv2
        h, w = self.cfg.input_h, self.cfg.input_w
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        return frame

    def _generate(
        self,
        frame: np.ndarray,
        actions: torch.Tensor,
    ) -> tuple[list[np.ndarray], float, list[float]]:
        """Run diffusion generation; return (decoded_frames, mean_conf, per_frame_confs)."""
        import einops

        resized = self._resize_frame(frame).astype(np.float32) / 255.0
        # (H, W, 3) → (1, 1, H, W, 3) for VAE
        frame_t = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(self._device)

        T = actions.shape[1]
        n_frames = T + 1  # 1 context + T predicted

        # pad actions with zeros for the context frame slot
        zero_action = torch.zeros(1, 1, self.cfg.action_dim, device=self._device)
        full_actions = torch.cat([zero_action, actions], dim=1)  # (1, T+1, 7)

        with torch.no_grad():
            with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16):
                latent = self._vae.encode(frame_t)

                gen_out = self._diffusion.generate(
                    self._model,
                    latent,
                    full_actions,
                    n_context_frames=1,
                    n_frames=n_frames,
                    window_len=None,
                    horizon=1,
                    err_threshold_conf_pred=float(self.cfg.conf_err_threshold),
                )

                generated_latents = gen_out["x_pred"]   # (1, T+1, h, w, C)
                raw_conf = gen_out.get("raw_conf_pred")  # (1, T+1, h, w, C) or None

                # decode predicted frames (skip context frame at index 0)
                pred_latents = generated_latents[:, 1:]   # (1, T, h, w, C)
                decoded = self._vae.decode(pred_latents)  # (1, T, H, W, 3)

        # convert to numpy uint8 frames
        decoded_np = (decoded[0].float().clamp(0, 1) * 255).byte().cpu().numpy()
        frames: list[np.ndarray] = []
        for t in range(decoded_np.shape[0]):
            frames.append(decoded_np[t])  # (H, W, 3)

        # compute trajectory-level confidence
        mean_conf = 0.5
        per_frame_confs: list[float] = []
        if raw_conf is not None:
            # raw_conf: (1, T+1, H, W, C) with values in [0,1] after sigmoid
            # skip context frame (index 0), keep predicted frames
            pred_conf = raw_conf[:, 1:].float().cpu()  # (1, T, H, W, C)
            per_frame_confs = pred_conf[0].mean(dim=(1, 2, 3)).tolist()
            mean_conf = float(pred_conf.mean().item())

        return frames, mean_conf, per_frame_confs
