"""Train a diffusion policy on the bridge dataset (BC pretraining).

Architecture: ResNet image encoder + MLP state encoder + FiLM-conditioned
DDPM denoiser predicting action chunks of length HORIZON.

Action space:  7-D  [dx, dy, dz, droll, dpitch, dyaw, gripper]
State space:   8-D  proprioceptive

Usage:
    python scripts/train_bridge_bc.py \\
        --dataset_path /n/fs/not-fmrl/Projects/wm_alignment/cosmos-predict2/datasets/bridge \\
        --output_dir outputs/bridge_bc \\
        --wandb_project bridge_bc \\
        --num_epochs 100 \\
        --batch_size 256 \\
        --lr 1e-4
"""

import argparse
import json
import logging
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BridgeChunkDataset(Dataset):
    """Bridge dataset: (frame_t, state_t, action_chunk[t:t+H]) triples.

    Fast loading strategy:
    1. Index cache — the (video_path, frame_idx, action_chunk, state) index is
       built once from JSON files and saved to `<data_root>/cache_<split>_h<H>.pkl`.
       Subsequent runs load the cache in ~1 second instead of re-reading 25k JSONs.
    2. JPEG frames — if `preprocess_bridge_frames.py` has been run, frames are
       read from `videos/<split>/<id>/frames/<XXXXXX>.jpg` (fast random access).
       Falls back to cv2 video seeking when JPEGs are absent.
    """

    _CACHE_VERSION = 2  # bump to invalidate old caches

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        horizon: int = 8,
        image_size: int = 128,
        max_trajs: int = -1,
        norm_stats: dict | None = None,
    ):
        from openworld.policies.bridge_bc_policy import ACTION_DIM, STATE_DIM
        self.horizon = horizon
        self.image_size = image_size
        self.action_dim = ACTION_DIM
        self.state_dim = STATE_DIM
        self._data_root = Path(data_root)
        self._split = split

        # ---- load or build index ----
        cache_tag = f"h{horizon}" + (f"_top{max_trajs}" if max_trajs > 0 else "")
        cache_path = self._data_root / f"cache_{split}_{cache_tag}.pkl"
        index = self._load_or_build_index(cache_path, max_trajs)

        self._video_paths: list[Path] = index["video_paths"]
        self._frame_indices: np.ndarray = index["frame_indices"]     # (N,) int32
        self._action_chunks: np.ndarray = index["action_chunks"]     # (N, H, 7)
        self._states: np.ndarray = index["states"]                   # (N, 8)

        logger.info(
            "BridgeChunkDataset [%s]: %d samples (horizon=%d)",
            split, len(self._video_paths), horizon,
        )

        self._norm_stats = norm_stats if norm_stats is not None else self._compute_norm_stats()

    # ------------------------------------------------------------------
    # Index cache
    # ------------------------------------------------------------------

    def _load_or_build_index(self, cache_path: Path, max_trajs: int) -> dict:
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("version") == self._CACHE_VERSION:
                    logger.info("Loaded dataset index from cache: %s", cache_path)
                    return cached
                logger.info("Cache version mismatch — rebuilding.")
            except Exception as e:
                logger.warning("Failed to load cache (%s) — rebuilding.", e)

        index = self._build_index(max_trajs)
        index["version"] = self._CACHE_VERSION
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(index, f, protocol=4)
            logger.info("Saved dataset index cache: %s", cache_path)
        except Exception as e:
            logger.warning("Could not save cache: %s", e)
        return index

    def _build_index(self, max_trajs: int) -> dict:
        video_root = self._data_root / "videos" / self._split
        anno_root = self._data_root / "annotation" / self._split

        anno_files = sorted(anno_root.glob("*.json"))
        if max_trajs > 0:
            anno_files = anno_files[:max_trajs]

        video_paths: list[Path] = []
        frame_indices: list[int] = []
        action_chunks: list[np.ndarray] = []
        states: list[np.ndarray] = []

        for anno_path in tqdm(anno_files, desc=f"Indexing {self._split}", leave=False):
            traj_id = anno_path.stem
            video_path = video_root / traj_id / "rgb.mp4"
            state_path = video_root / traj_id / "state.npy"
            if not video_path.exists():
                continue
            try:
                with open(anno_path) as f:
                    data = json.load(f)
                actions = np.array(data["action"], dtype=np.float32)
                traj_states = (
                    np.load(str(state_path)).astype(np.float32)
                    if state_path.exists()
                    else np.zeros((len(actions) + 1, self.state_dim), dtype=np.float32)
                )
            except Exception as e:
                logger.debug("Skipping %s: %s", anno_path, e)
                continue

            T = len(actions)
            for t in range(T - self.horizon + 1):
                video_paths.append(video_path)
                frame_indices.append(t)
                action_chunks.append(actions[t : t + self.horizon])
                states.append(traj_states[t, : self.state_dim])

        return {
            "video_paths": video_paths,
            "frame_indices": np.array(frame_indices, dtype=np.int32),
            "action_chunks": np.stack(action_chunks).astype(np.float32),
            "states": np.stack(states).astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Norm stats
    # ------------------------------------------------------------------

    def _compute_norm_stats(self) -> dict:
        if len(self._action_chunks) == 0:
            return {"action_mean": np.zeros(self.action_dim), "action_std": np.ones(self.action_dim)}
        flat = self._action_chunks.reshape(-1, self.action_dim)
        return {"action_mean": flat.mean(0), "action_std": flat.std(0) + 1e-6}

    @property
    def norm_stats(self) -> dict:
        return self._norm_stats

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._video_paths)

    def __getitem__(self, idx: int):
        video_path = self._video_paths[idx]
        frame_idx = int(self._frame_indices[idx])
        raw_chunk = self._action_chunks[idx]
        state = self._states[idx]

        frame = self._load_frame(video_path, frame_idx)
        chunk_norm = (raw_chunk - self._norm_stats["action_mean"]) / self._norm_stats["action_std"]

        img_t = torch.from_numpy(
            cv2.resize(frame, (self.image_size, self.image_size)).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        return (
            img_t,
            torch.from_numpy(state),
            torch.from_numpy(chunk_norm.astype(np.float32)),
        )

    @staticmethod
    def _load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
        import mediapy as _mp

        # Fast path: pre-extracted JPEG (run preprocess_bridge_frames.py once)
        jpeg_path = video_path.parent / "frames" / f"{frame_idx:06d}.jpg"
        if jpeg_path.exists():
            try:
                return np.asarray(_mp.read_image(str(jpeg_path)), dtype=np.uint8)
            except Exception:
                pass

        # Slow fallback: seek inside the video
        try:
            frames = _mp.read_video(str(video_path))
            if frame_idx < len(frames):
                return np.asarray(frames[frame_idx], dtype=np.uint8)
        except Exception:
            pass
        return np.zeros((128, 128, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from openworld.policies.bridge_bc_policy import (
        _BridgeDPModel, ACTION_DIM, STATE_DIM,
        DDPM_TRAIN_STEPS, DDIM_INFER_STEPS, HORIZON,
    )

    import wandb
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config=vars(args),
        mode="disabled" if args.wandb_project is None else "online",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    train_set = BridgeChunkDataset(
        args.dataset_path, split="train",
        horizon=args.horizon, image_size=args.image_size,
        max_trajs=args.max_trajs,
    )
    val_set = BridgeChunkDataset(
        args.dataset_path, split="val",
        horizon=args.horizon, image_size=args.image_size,
        max_trajs=min(200, args.max_trajs) if args.max_trajs > 0 else 200,
        norm_stats=train_set.norm_stats,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=2)

    model = _BridgeDPModel(
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        horizon=args.horizon,
        ddpm_T=DDPM_TRAIN_STEPS,
        ddim_steps=DDIM_INFER_STEPS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %d", n_params)
    wandb.config.update({"n_params": n_params})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val = float("inf")

    epoch_bar = tqdm(range(args.num_epochs), desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        total = 0.0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for imgs, states, actions in batch_bar:
            imgs = imgs.to(device)
            states = states.to(device)
            actions = actions.to(device)

            loss = model.loss(imgs, states, actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += loss.item()
            global_step += 1
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            wandb.log({"train/loss_step": loss.item(), "step": global_step})

        scheduler.step()
        avg_train = total / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, states, actions in tqdm(val_loader, desc="Val", leave=False):
                val_loss += model.loss(
                    imgs.to(device), states.to(device), actions.to(device)
                ).item()
        avg_val = val_loss / max(len(val_loader), 1)

        lr = scheduler.get_last_lr()[0]
        epoch_bar.set_postfix(train=f"{avg_train:.4f}", val=f"{avg_val:.4f}")
        wandb.log({
            "train/loss_epoch": avg_train,
            "val/loss": avg_val,
            "lr": lr,
            "epoch": epoch,
        })

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "norm_stats": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in train_set.norm_stats.items()
            },
            "val_loss": avg_val,
        }
        torch.save(ckpt, output_dir / "last.pt")
        if avg_val < best_val:
            best_val = avg_val
            torch.save(ckpt, output_dir / "best.pt")
            wandb.run.summary["best_val_loss"] = best_val
            wandb.run.summary["best_epoch"] = epoch

    logger.info("Done. best val=%.4f  checkpoint: %s", best_val, output_dir / "best.pt")
    run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",
                   default="/n/fs/not-fmrl/Projects/wm_alignment/cosmos-predict2/datasets/bridge")
    p.add_argument("--output_dir", default="outputs/bridge_bc")
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--max_trajs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="bridge_bc",
                   help="W&B project name. Pass --wandb_project '' to disable.")
    p.add_argument("--wandb_run", default=None, help="W&B run name (auto-generated if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # treat empty string as disabled
    if args.wandb_project == "":
        args.wandb_project = None
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train(args)
