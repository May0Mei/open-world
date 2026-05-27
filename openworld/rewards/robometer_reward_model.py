"""Robometer reward model for inline RL finetuning.

Robometer runs in its own uv project at external/robometer/ due to
dependency conflicts with the main venv. This class:
  1. Saves the predicted frames from a world-model rollout to a temp mp4.
  2. Calls `uv run python scripts/score_videos_robometer.py` as a subprocess
     from external/robometer/, exactly as run_evaluation.py does.
  3. Parses the returned JSON and returns a scalar reward.

Reward signal: progress_improvement = mean(last-third progress) - mean(first-third progress).
This is dense (non-zero for any motion) and scale-invariant across chunk lengths.

Setup (once):
    git clone https://github.com/robometer/robometer.git external/robometer
    cd external/robometer && uv sync
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from openworld.rewards.base_reward_model import RewardModel

logger = logging.getLogger(__name__)

# Absolute path to the robometer uv project (clone of robometer repo).
_DEFAULT_ROBOMETER_DIR = Path(__file__).resolve().parents[2] / "external" / "robometer"
_SCORE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "score_videos_robometer.py"


class RobometerRewardModel(RewardModel):
    """Inline robometer reward via subprocess.

    Args:
        model_path: HuggingFace model id or local checkpoint path.
        robometer_dir: Path to the robometer uv project (contains pyproject.toml).
        fps: Frame-sampling rate passed to the scoring script.
        num_views: 1 for single-view videos (bridge); 3 for DROID-style 3-view.
        timeout_s: Subprocess timeout in seconds.
    """

    def __init__(
        self,
        model_path: str = "robometer/Robometer-4B",
        robometer_dir: Optional[str] = None,
        fps: float = 2.0,
        num_views: int = 1,
        timeout_s: float = 120.0,
        **_: Any,
    ) -> None:
        self.model_path = model_path
        self.robometer_dir = Path(robometer_dir or _DEFAULT_ROBOMETER_DIR).resolve()
        self.fps = fps
        self.num_views = num_views
        self.timeout_s = timeout_s
        self._warned_missing = False

    def compute(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Score a rollout chunk with Robometer.

        Args:
            trajectory: dict with:
                "frames": list of (H, W, 3) uint8 numpy arrays.
                "instruction": str task description.

        Returns:
            dict with "reward" (float), "per_frame_progress" (list),
            "success_probs" (list).
        """
        frames: list[np.ndarray] = trajectory.get("frames", [])
        instruction: str = trajectory.get("instruction", "")

        if not frames:
            return {"reward": 0.0, "per_frame_progress": [], "success_probs": []}

        if not self.robometer_dir.exists():
            if not self._warned_missing:
                logger.warning(
                    "Robometer not found at %s. Returning zero reward. "
                    "Clone it with: git clone https://github.com/robometer/robometer.git %s",
                    self.robometer_dir, self.robometer_dir,
                )
                self._warned_missing = True
            return {"reward": 0.0, "per_frame_progress": [], "success_probs": []}

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video_path = tmp / "chunk.mp4"
            manifest_path = tmp / "manifest.json"
            rewards_path = tmp / "rewards.json"

            self._write_video(frames, video_path)
            manifest_path.write_text(json.dumps({
                "episodes": [{"id": "chunk_0", "video_path": str(video_path),
                               "instruction": instruction}]
            }))

            cmd = [
                "uv", "run", "python", str(_SCORE_SCRIPT),
                "--manifest", str(manifest_path),
                "--model-path", self.model_path,
                "--output", str(rewards_path),
                "--fps", str(self.fps),
                "--num-views", str(self.num_views),
            ]

            try:
                subprocess.run(
                    cmd,
                    cwd=str(self.robometer_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=True,
                )
            except subprocess.TimeoutExpired:
                logger.warning("Robometer scoring timed out (%.0fs). Returning zero.", self.timeout_s)
                return {"reward": 0.0, "per_frame_progress": [], "success_probs": []}
            except subprocess.CalledProcessError as e:
                logger.warning("Robometer subprocess failed:\n%s", e.stderr[-500:])
                return {"reward": 0.0, "per_frame_progress": [], "success_probs": []}

            result = json.loads(rewards_path.read_text())

        ep = result.get("episodes", [{}])[0]
        if "error" in ep:
            logger.warning("Robometer error: %s", ep["error"])
            return {"reward": 0.0, "per_frame_progress": [], "success_probs": []}

        progress = ep.get("per_frame_progress", [])
        success = ep.get("success_probs", [])
        reward = self._progress_improvement(progress)

        return {"reward": reward, "per_frame_progress": progress, "success_probs": success}

    # ------------------------------------------------------------------

    @staticmethod
    def _progress_improvement(progress: list[float]) -> float:
        """Dense reward: mean progress over last third minus mean over first third."""
        if not progress:
            return 0.0
        n = len(progress)
        third = max(1, n // 3)
        first = float(np.mean(progress[:third]))
        last = float(np.mean(progress[-third:]))
        return float(np.clip(last - first, -1.0, 1.0))

    @staticmethod
    def _write_video(frames: list[np.ndarray], path: Path, fps: int = 8) -> None:
        import cv2
        if not frames:
            return
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        writer.release()
