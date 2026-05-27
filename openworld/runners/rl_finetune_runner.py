"""DSRL-style RL fine-tuning for world-model environments.

Algorithm (mirrors /n/fs/not-fmrl/Projects/apple_project/dsrl_pi0):
  * A PixelSAC agent predicts diffusion *noise* for the frozen base policy
    (OpenPI / pi05_droid).  Controlling the initial noise steers the
    flow-matching denoiser toward high-reward action trajectories without
    touching the backbone weights.
  * Rollouts are collected inside WorldModelEnv (virtual robot rollouts) instead
    of on a real robot.
  * Transitions (obs, noise_action, reward, next_obs) go into a replay buffer
    and are used to update the SAC agent after each trajectory.

The SAC agent is implemented in PyTorch (see ``torch_pixel_sac.py``) to avoid
JAX version conflicts with the rest of the stack.

Observation fed to SAC:  {'pixels': (H, W, 3, 1)  uint8}  — one camera view.
SAC action:              (1, noise_dim)  float32  — one row of diffusion noise.
Full noise for OpenPI:   (action_horizon, noise_dim)  — SAC row tiled.

Configurable via train_params in the YAML config; defaults match dsrl_pi0.
"""

from __future__ import annotations

import collections
import logging
import random
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional

import cv2
import numpy as np
import wandb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from openworld.datasets.initialization import Initialization
from openworld.envs.world_model_env import WorldModelEnv
from openworld.policies.base_policy import Policy
from openworld.rewards.base_reward_model import RewardModel

logger = logging.getLogger(__name__)
_console = Console()

def _conf_style(conf: float, threshold: Optional[float]) -> str:
    if threshold is None:
        return "cyan"
    if conf >= threshold:
        return "green"
    if conf >= threshold * 0.75:
        return "yellow"
    return "red"


def _print_traj_table(
    traj_idx: int,
    reward: float,
    mean_conf: float,
    conf_filtered: bool,
    traj_confidences: List[float],
    chunk_frames_list: List[List[Any]],
    conf_filter_threshold: Optional[float],
    use_random: bool,
) -> None:
    table = Table(show_header=True, header_style="bold dim", box=None, pad_edge=False)
    table.add_column("chunk", style="dim", width=6)
    table.add_column("frames", width=7)
    table.add_column("confidence", width=12)

    for i, (conf, frames) in enumerate(zip(traj_confidences, chunk_frames_list)):
        style = _conf_style(conf, conf_filter_threshold)
        table.add_row(str(i), str(len(frames)), f"[{style}]{conf:.3f}[/{style}]")

    status_style = "red" if conf_filtered else "green"
    status = "FILTERED" if conf_filtered else "kept"
    mode = "[dim]random[/dim]" if use_random else "agent"
    title = (
        f"traj [bold]{traj_idx:04d}[/bold]  "
        f"reward=[yellow]{reward:+.3f}[/yellow]  "
        f"mean_conf=[{_conf_style(mean_conf, conf_filter_threshold)}]{mean_conf:.3f}[/{_conf_style(mean_conf, conf_filter_threshold)}]  "
        f"[{status_style}]{status}[/{status_style}]  {mode}"
    )
    _console.print(Panel(table, title=title, expand=False, border_style="dim"))


# Franka Panda home-ish joint angles (radians)
_FRANKA_HOME_JOINTS = np.array(
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32
)


# ---------------------------------------------------------------------------
# Minimal replay buffer (no JAX dependency)
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    """Ring-buffer replay buffer for (obs, action, reward, next_obs, mask)."""

    def __init__(self, capacity: int, seed: int = 42):
        self._capacity = capacity
        self._buf: Deque[Dict[str, Any]] = collections.deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def insert(self, transition: Dict[str, Any]) -> None:
        self._buf.append(transition)

    def sample(self, batch_size: int) -> Dict[str, Any]:
        batch = self._rng.sample(list(self._buf), min(batch_size, len(self._buf)))
        stacked: Dict[str, Any] = {}
        for key in batch[0]:
            vals = [t[key] for t in batch]
            if isinstance(vals[0], dict):
                stacked[key] = {
                    sub_key: np.stack([v[sub_key] for v in vals])
                    for sub_key in vals[0]
                }
            else:
                stacked[key] = np.stack(vals)
        return stacked

    def get_iterator(self, batch_size: int) -> Iterator[Dict[str, Any]]:
        while True:
            yield self.sample(batch_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_to_sac_pixels(
    observation: Any,
    resize: int = 128,
    exterior_view_name: str = "exterior_right",
) -> np.ndarray:
    """Extract one camera frame from a world-model observation and resize for SAC.

    Returns shape (resize, resize, 3, 1) — matching jaxrl2 / TorchPixelSAC layout.
    """
    if isinstance(observation, dict):
        views = observation.get("views", observation)
        img_array = None
        for key in (exterior_view_name, "exterior_left", "exterior_right", "wrist"):
            if key in views:
                val = views[key]
                if isinstance(val, str):
                    from PIL import Image
                    with Image.open(val) as im:
                        img_array = np.asarray(im.convert("RGB"), dtype=np.uint8)
                else:
                    img_array = np.asarray(val, dtype=np.uint8)
                break
        if img_array is None:
            img_array = np.zeros((resize, resize, 3), dtype=np.uint8)
    elif isinstance(observation, np.ndarray):
        img_array = observation.astype(np.uint8)
    else:
        img_array = np.zeros((resize, resize, 3), dtype=np.uint8)

    if img_array.ndim == 3:
        h, w, c = img_array.shape
        if c >= 3 and h != w and h % 3 == 0:
            # Three views stacked vertically — take the top (exterior) view
            img_array = img_array[: h // 3, :, :3]
        else:
            img_array = img_array[:, :, :3]
    else:
        img_array = np.zeros((resize, resize, 3), dtype=np.uint8)

    if img_array.shape[:2] != (resize, resize):
        img_array = cv2.resize(img_array, (resize, resize), interpolation=cv2.INTER_LINEAR)

    return img_array[..., np.newaxis].astype(np.uint8)  # (H, W, 3, 1)


_FALLBACK_IMAGE_PATHS = [
    # Real robot frames from the benchmark dataset — used as conditioning image
    # so VidWM generates visually meaningful rollouts instead of all-black frames.
    "data/benchmark/irom_test_carrot/irom_test_carrot/init_1/exterior_right.png",
    "data/benchmark/irom_test_carrot/irom_test_carrot/init_1/exterior_left.png",
    "data/benchmark/irom_test_carrot/irom_test_carrot/init_1/wrist.png",
]


def _load_fallback_observation() -> np.ndarray:
    """Load real robot views stacked vertically (576, 320, 3) for VidWM conditioning."""
    import mediapy as _mp

    views = []
    for path in _FALLBACK_IMAGE_PATHS:
        p = Path(path)
        if p.exists():
            img = np.asarray(_mp.read_image(str(p)), dtype=np.uint8)
            views.append(img)

    if len(views) == 3:
        return np.concatenate(views, axis=0)  # (576, 320, 3)

    # No real images found — use mid-gray placeholder so VidWM at least has a
    # non-zero conditioning frame.
    return np.full((576, 320, 3), 128, dtype=np.uint8)


def _make_dummy_initialization(instruction: str = "pick up the object") -> Initialization:
    """Minimal initialization for RL training without a real dataset."""
    # VidWM expects 3 views stacked vertically: 3 × 192px = 576px total height.
    dummy_obs = _load_fallback_observation()
    initial_state: Dict[str, Any] = {
        "robot": {
            "joint_position": _FRANKA_HOME_JOINTS.copy(),
            "joint_positions": _FRANKA_HOME_JOINTS.copy(),
            "gripper_position": np.array([0.05], dtype=np.float32),
            "state": np.concatenate([_FRANKA_HOME_JOINTS, [0.05]]),
        }
    }
    return Initialization(
        id="dummy_rl_init",
        initial_state=initial_state,
        initial_observation=dummy_obs,
        instruction=instruction,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class RLFineTuneRunner:
    """DSRL-style RL fine-tuning of a frozen policy inside a world-model env.

    A TorchPixelSAC agent learns to predict diffusion noise that steers the
    base policy toward high-reward rollouts.  Base-policy weights are never
    updated.
    """

    def __init__(
        self,
        env: WorldModelEnv,
        policy: Policy,
        reward_model: Optional[RewardModel] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.env = env
        self.policy = policy
        self.reward_model = reward_model
        self.config = config or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.config

        # --- hyperparameters ---
        seed: int = cfg.get("seed", 42)
        max_steps: int = cfg.get("max_steps", 10000)
        batch_size: int = cfg.get("batch_size", 256)
        resize_image: int = cfg.get("resize_image", 128)
        action_horizon: int = cfg.get("action_horizon", 15)
        noise_dim: int = cfg.get("noise_dim", 32)
        max_chunks_per_episode: int = cfg.get("max_chunks_per_episode", 5)
        num_initial_trajs: int = cfg.get("num_initial_trajs", 3)
        multi_grad_step: int = cfg.get("multi_grad_step", 1)
        discount: float = cfg.get("discount", 0.99)
        log_interval: int = cfg.get("log_interval", 50)
        checkpoint_interval: int = cfg.get("checkpoint_interval", 500)
        video_save_interval: int = cfg.get("video_save_interval", 10)
        output_dir: str = cfg.get("output_dir", "outputs/rl_openpi")
        dataset_path: Optional[str] = cfg.get("dataset_path", None)
        instruction: str = cfg.get("instruction", "pick up the object")
        exterior_view: str = cfg.get("exterior_view_name", "exterior_right")
        actor_lr: float = cfg.get("actor_lr", 3e-4)
        critic_lr: float = cfg.get("critic_lr", 3e-4)
        pytorch_device: Optional[str] = cfg.get("pytorch_device", None)
        # Confidence-based trajectory filtering (set to None to disable).
        # Trajectories where the world model's mean confidence falls below
        # this threshold are not added to the replay buffer.
        conf_filter_threshold: Optional[float] = cfg.get("conf_filter_threshold", None)
        wandb_project: Optional[str] = cfg.get("wandb_project", None)
        wandb_run: Optional[str] = cfg.get("wandb_run", None)
        wandb_entity: Optional[str] = cfg.get("wandb_entity", None)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        wandb.init(
            project=wandb_project,
            name=wandb_run,
            entity=wandb_entity,
            config=cfg,
            mode="disabled" if wandb_project is None else "online",
        )

        # --- build SAC agent ---
        from openworld.runners.torch_pixel_sac import TorchPixelSACAgent

        image_shape = (resize_image, resize_image, 3, 1)
        sample_obs = {"pixels": np.zeros(image_shape, dtype=np.uint8)[np.newaxis]}
        sample_action = np.zeros((1, noise_dim), dtype=np.float32)

        agent = TorchPixelSACAgent(
            seed=seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            discount=discount,
            device=pytorch_device,
        )
        logger.info("TorchPixelSAC initialised on device=%s", agent.device)

        # --- replay buffer ---
        capacity = max(max_steps * max_chunks_per_episode * 2, 10_000)
        replay_buffer = _ReplayBuffer(capacity=capacity, seed=seed)

        # --- initializations ---
        initializations = self._load_initializations(dataset_path, instruction)

        # log a grid of sample initial frames to wandb
        sample_frames = []
        for init in initializations[:16]:
            obs = init.initial_observation
            if isinstance(obs, np.ndarray) and obs.ndim == 3:
                sample_frames.append(obs)
        if sample_frames:
            wandb.log({"rl/initial_frames": [wandb.Image(f, caption=initializations[i].id) for i, f in enumerate(sample_frames)]})

        total_grad_steps = 0
        total_trajs = 0

        _console.print(
            f"[bold]DSRL training[/bold]  max_steps={max_steps}  batch={batch_size}  "
            f"horizon={action_horizon}  noise_dim={noise_dim}  "
            f"conf_threshold={conf_filter_threshold}"
        )

        while total_grad_steps < max_steps:
            init = initializations[total_trajs % len(initializations)]
            use_random = total_trajs < num_initial_trajs

            traj, chunk_frames_list, traj_confidences = self._collect_trajectory(
                agent=agent,
                initialization=init,
                action_horizon=action_horizon,
                noise_dim=noise_dim,
                max_chunks=max_chunks_per_episode,
                resize_image=resize_image,
                exterior_view=exterior_view,
                use_random_noise=use_random,
                discount=discount,
            )

            # confidence-based filtering: skip low-confidence trajectories
            mean_traj_conf = (
                float(np.mean(traj_confidences)) if traj_confidences else 1.0
            )
            conf_filtered = (
                conf_filter_threshold is not None
                and mean_traj_conf < conf_filter_threshold
            )
            if not conf_filtered:
                for transition in traj:
                    replay_buffer.insert(transition)

            total_trajs += 1
            episode_reward = float(sum(t["rewards"] for t in traj))

            _print_traj_table(
                traj_idx=total_trajs,
                reward=episode_reward,
                mean_conf=mean_traj_conf,
                conf_filtered=conf_filtered,
                traj_confidences=traj_confidences,
                chunk_frames_list=chunk_frames_list,
                conf_filter_threshold=conf_filter_threshold,
                use_random=use_random,
            )
            _console.print(
                f"  [dim]buffer={len(replay_buffer)}  steps={total_grad_steps}[/dim]"
            )

            wandb.log({
                "rl/reward": episode_reward,
                "rl/mean_confidence": mean_traj_conf,
                "rl/filtered": float(conf_filtered),
                "rl/buffer_size": len(replay_buffer),
                "rl/grad_steps": total_grad_steps,
                **{f"rl/chunk_conf/{i}": c for i, c in enumerate(traj_confidences)},
            }, step=total_trajs)

            if total_trajs % video_save_interval == 0:
                video_path = self._save_trajectory_video(
                    chunk_frames_list, traj_confidences, output_dir, total_trajs,
                    conf_filter_threshold,
                    initial_obs=init.initial_observation,
                )
                if video_path is not None:
                    wandb.log({
                        "rl/trajectory_video": wandb.Video(str(video_path), fps=4, format="mp4"),
                    }, step=total_trajs)

            if use_random:
                continue

            # --- gradient updates ---
            num_grads = max(len(traj) * multi_grad_step, 1)
            if total_trajs == num_initial_trajs:
                num_grads = max(num_grads, min(1000, max_steps // 2))

            buf_iter = replay_buffer.get_iterator(batch_size)
            for _ in range(num_grads):
                if len(replay_buffer) < batch_size:
                    break
                batch = next(buf_iter)
                update_info = agent.update(batch)
                total_grad_steps += 1

                wandb.log({f"sac/{k}": v for k, v in update_info.items()}, step=total_grad_steps)

                if total_grad_steps % log_interval == 0:
                    parts = "  ".join(
                        f"[dim]{k}[/dim]=[cyan]{v:.4f}[/cyan]"
                        for k, v in update_info.items()
                    )
                    _console.log(f"[bold]step={total_grad_steps}[/bold]  {parts}")

                if (
                    checkpoint_interval > 0
                    and total_grad_steps % checkpoint_interval == 0
                ):
                    agent.save_checkpoint(output_dir, total_grad_steps, checkpoint_interval)
                    _console.print(f"  [dim]checkpoint saved at step {total_grad_steps}[/dim]")

                if total_grad_steps >= max_steps:
                    break

        _console.print(f"\n[bold green]Training done.[/bold green]  Total gradient steps: {total_grad_steps}")
        wandb.finish()

    # ------------------------------------------------------------------
    # Trajectory collection
    # ------------------------------------------------------------------

    def _collect_trajectory(
        self,
        agent: Any,
        initialization: Initialization,
        action_horizon: int,
        noise_dim: int,
        max_chunks: int,
        resize_image: int,
        exterior_view: str,
        use_random_noise: bool,
        discount: float,
    ) -> tuple:
        """Collect one episode; return (transitions, chunk_frames_list, confidences).

        chunk_frames_list[i] is the list of predicted RGB frames for chunk i.
        confidences[i] is the mean world-model confidence for chunk i (or 1.0
        if the world model does not return confidence).
        """
        env_info = self.env.reset(initialization)
        obs = env_info["observation"]
        state = env_info["state"]
        self.policy.reset(instruction=initialization.instruction)

        transitions: List[Dict[str, Any]] = []
        chunk_frames_list: List[List[Any]] = []
        confidences: List[float] = []

        for chunk_idx in range(max_chunks):
            sac_pixels = _frame_to_sac_pixels(obs, resize_image, exterior_view)
            sac_obs = {"pixels": sac_pixels}

            # --- sample noise ---
            if use_random_noise:
                noise_1d = np.random.uniform(-1.0, 1.0, (1, noise_dim)).astype(np.float32)
            else:
                noise_1d = agent.sample_actions(sac_obs)
                noise_1d = np.asarray(noise_1d, dtype=np.float32).reshape(1, noise_dim)

            # --- tile noise to (action_horizon, noise_dim) for OpenPI ---
            full_noise = np.repeat(noise_1d, action_horizon, axis=0)

            # --- run frozen policy with injected noise ---
            adapted_actions = self._run_policy_with_noise(
                obs, state, full_noise, initialization.instruction
            )

            # --- step through env action-by-action ---
            predicted_frames: List[Any] = []
            chunk_confidence: float = 1.0
            for action in adapted_actions:
                step_result = self.env.step(action)
                if step_result["did_rollout"]:
                    predicted_frames = step_result["predicted_frames"]
                    obs = step_result["observation"]
                    state = step_result["state"]
                    if step_result.get("confidence") is not None:
                        chunk_confidence = float(step_result["confidence"])

            chunk_frames_list.append(predicted_frames)
            confidences.append(chunk_confidence)

            # --- reward ---
            if self.reward_model is not None:
                reward_info = self.reward_model.compute(
                    {"frames": predicted_frames, "instruction": initialization.instruction}
                )
                reward = float(reward_info.get("reward", 0.0))
            else:
                reward = 0.0

            # --- store transition ---
            next_pixels = _frame_to_sac_pixels(obs, resize_image, exterior_view)
            is_last = chunk_idx == max_chunks - 1
            transitions.append(
                {
                    "observations": sac_obs,
                    "next_observations": {"pixels": next_pixels},
                    "actions": noise_1d,
                    "rewards": np.float32(reward),
                    "masks": np.float32(0.0 if is_last else 1.0),
                    "discount": np.float32(discount),
                }
            )

        return transitions, chunk_frames_list, confidences

    def _run_policy_with_noise(
        self,
        observation: Any,
        state: Any,
        full_noise: np.ndarray,
        instruction: Optional[str],
    ) -> List[Any]:
        """Call base policy with injected diffusion noise; return adapted actions."""
        if hasattr(self.policy, "infer_chunk_with_noise"):
            return self.policy.infer_chunk_with_noise(
                observation=observation,
                state=state,
                noise=full_noise,
                instruction=instruction,
            )

        # Fallback for policies without noise injection (e.g. during testing)
        logger.debug(
            "%s does not support noise injection; using standard act()",
            type(self.policy).__name__,
        )
        chunk_size = self.env.scheduler.chunk_size
        actions: List[Any] = []
        for _ in range(chunk_size):
            action = self.policy.act(
                observation=observation, state=state, instruction=instruction
            )
            actions.append(action)
        return actions

    def _load_initializations(
        self,
        dataset_path: Optional[str],
        instruction: str,
        max_inits: int = 200,
        split: str = "train",
    ) -> List[Initialization]:
        if dataset_path is None:
            _console.print("[yellow]No dataset_path — using dummy initialization.[/yellow]")
            return [_make_dummy_initialization(instruction)]

        dataset_path = Path(dataset_path)
        bridge_video_root = dataset_path / "videos" / split

        if bridge_video_root.exists():
            return self._load_bridge_initializations(
                dataset_path, instruction, split=split, max_inits=max_inits
            )

        from openworld.datasets.initialization_dataset import InitializationDataset

        _console.print(f"Loading initializations from [cyan]{dataset_path}[/cyan]")
        dataset = InitializationDataset.from_yaml(str(dataset_path))
        if len(dataset) == 0:
            _console.print("[yellow]Dataset empty — falling back to dummy initialization.[/yellow]")
            return [_make_dummy_initialization(instruction)]
        return list(dataset)

    @staticmethod
    def _load_bridge_initializations(
        dataset_path: Path,
        default_instruction: str,
        split: str = "train",
        max_inits: int = 200,
    ) -> List[Initialization]:
        """Build Initialization objects from the bridge dataset layout.

        Reads the first frame of each trajectory video as the initial observation
        and the first proprioceptive state row as initial_state.
        """
        import json

        video_root = dataset_path / "videos" / split
        anno_root = dataset_path / "annotation" / split

        traj_dirs = sorted(d for d in video_root.iterdir() if d.is_dir())
        rng = random.Random(42)
        rng.shuffle(traj_dirs)
        traj_dirs = traj_dirs[:max_inits]

        inits: List[Initialization] = []
        for traj_dir in traj_dirs:
            traj_id = traj_dir.name
            video_path = traj_dir / "rgb.mp4"
            state_path = traj_dir / "state.npy"
            anno_path = anno_root / f"{traj_id}.json"

            if not video_path.exists():
                continue

            # first frame as initial observation
            import mediapy as _mp
            try:
                frames = _mp.read_video(str(video_path))
            except Exception:
                continue
            if len(frames) == 0:
                continue
            obs = np.asarray(frames[0], dtype=np.uint8)

            # first state row
            if state_path.exists():
                state_arr = np.load(str(state_path)).astype(np.float32)
                state = {"robot": {"state": state_arr[0] if state_arr.ndim > 1 else state_arr}}
            else:
                state = {"robot": {"state": np.zeros(8, dtype=np.float32)}}

            # instruction from annotation
            instr = default_instruction
            if anno_path.exists():
                try:
                    with open(anno_path) as f:
                        data = json.load(f)
                    instr = data.get("instruction", default_instruction)
                except Exception:
                    pass

            inits.append(Initialization(
                id=traj_id,
                initial_state=state,
                initial_observation=obs,
                instruction=instr,
            ))

        _console.print(f"Loaded [green]{len(inits)}[/green] bridge initializations from [cyan]{video_root}[/cyan]")
        if not inits:
            _console.print("[yellow]No valid trajectories found — falling back to dummy.[/yellow]")
            return [_make_dummy_initialization(default_instruction)]
        return inits

    def _save_trajectory_video(
        self,
        chunk_frames_list: List[List[Any]],
        traj_confidences: List[float],
        output_dir: str,
        traj_idx: int,
        conf_filter_threshold: Optional[float] = None,
        fps: int = 4,
        initial_obs: Optional[Any] = None,
    ) -> None:
        """Concatenate all chunks into one annotated MP4, overlaying chunk index and confidence."""
        import imageio

        all_frames = []

        # Prepend the raw initial observation so the video shows where the rollout started.
        if initial_obs is not None and len(chunk_frames_list) > 0 and len(chunk_frames_list[0]) > 0:
            ref_h, ref_w = np.asarray(chunk_frames_list[0][0]).shape[:2]
            init_frame = np.asarray(initial_obs, dtype=np.uint8)
            if init_frame.ndim == 3 and init_frame.shape[2] >= 3:
                init_frame = init_frame[:, :, :3]
            init_frame = cv2.resize(init_frame, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
            cv2.putText(init_frame, "init", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(init_frame, "init", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            all_frames.append(init_frame)

        for chunk_idx, (frames, conf) in enumerate(zip(chunk_frames_list, traj_confidences)):
            color = (0, 220, 0) if (conf_filter_threshold is None or conf >= conf_filter_threshold) else (0, 0, 220)
            for frame in frames:
                f = np.asarray(frame, dtype=np.uint8).copy()
                label = f"chunk={chunk_idx}  conf={conf:.3f}"
                cv2.putText(f, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(f, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                all_frames.append(f)

        if not all_frames:
            return None

        video_dir = Path(output_dir) / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        path = video_dir / f"traj_{traj_idx:05d}.mp4"
        imageio.mimwrite(str(path), all_frames, fps=fps)
        _console.print(f"  [dim]saved video ({len(all_frames)} frames): {path}[/dim]")
        return path
