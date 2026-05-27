"""Pre-extract bridge video frames to JPEG for fast DataLoader reads.

Reads each trajectory video once (sequential, fast) and saves every frame as
a JPEG next to the video:
    videos/train/0/rgb.mp4  →  videos/train/0/frames/000000.jpg
                                                      000001.jpg ...

Run once before training:
    python scripts/preprocess_bridge_frames.py --split train --num_workers 16
    python scripts/preprocess_bridge_frames.py --split val   --num_workers 8

Training will automatically use the JPEGs when present; falls back to video
seeking if a JPEG is missing (slower but safe).
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JPEG_QUALITY = 95


def _extract_traj(video_path: Path) -> tuple[str, int]:
    """Extract all frames from one trajectory video. Returns (path_str, n_frames)."""
    out_dir = video_path.parent / "frames"
    out_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = out_dir / f"{n:06d}.jpg"
        if not out_path.exists():
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        n += 1
    cap.release()
    return str(video_path), n


def main(args: argparse.Namespace) -> None:
    video_root = Path(args.dataset_path) / "videos" / args.split
    video_paths = sorted(video_root.rglob("rgb.mp4"))

    if args.overwrite:
        logger.info("--overwrite set: re-extracting all frames")
    else:
        # skip trajectories that already have a frames/ dir
        before = len(video_paths)
        video_paths = [p for p in video_paths if not (p.parent / "frames").exists()]
        logger.info("Skipping %d already-extracted trajectories", before - len(video_paths))

    if not video_paths:
        logger.info("All frames already extracted. Nothing to do.")
        return

    logger.info("Extracting frames from %d trajectories with %d workers ...",
                len(video_paths), args.num_workers)

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_extract_traj, p): p for p in video_paths}
        total_frames = 0
        with tqdm(total=len(futures), desc=f"Extracting [{args.split}]") as bar:
            for fut in as_completed(futures):
                try:
                    _, n = fut.result()
                    total_frames += n
                except Exception as e:
                    logger.warning("Failed %s: %s", futures[fut], e)
                bar.update(1)

    logger.info("Done. Extracted ~%d frames total.", total_frames)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",
                   default="/n/fs/not-fmrl/Projects/wm_alignment/cosmos-predict2/datasets/bridge")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
