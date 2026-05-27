"""Dummy random policy for testing RL training loops without real policy weights."""

from typing import Any, Optional

import numpy as np

from openworld.policies.base_policy import Policy


class DummyPolicy(Policy):
    """Returns random 7-D cartesian actions. No JAX or model weights needed."""

    def __init__(self, action_dim: int = 7, seed: int = 42):
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)

    def reset(self, instruction: Optional[str] = None) -> None:
        pass

    def act(self, observation: Any, state: Any, instruction: Optional[str] = None) -> Any:
        return self._rng.uniform(-1.0, 1.0, (self.action_dim,)).astype(np.float32)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        pass

    def infer_chunk_with_noise(
        self,
        observation: Any,
        state: Any,
        noise: np.ndarray,
        instruction: Optional[str] = None,
    ):
        """Noise-aware inference: return chunk of random actions (noise-shaped)."""
        chunk_size = noise.shape[0] if noise.ndim >= 1 else 15
        return [
            self._rng.uniform(-1.0, 1.0, (self.action_dim,)).astype(np.float32)
            for _ in range(chunk_size)
        ]
