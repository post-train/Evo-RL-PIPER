#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import write_info, write_stats, write_tasks  # noqa: E402


@dataclass
class TrimDecision:
    source_episode_index: int
    output_episode_index: int
    original_length: int
    trim_start: int
    trim_end: int
    trimmed_length: int
    head_static_frames: int
    tail_static_frames: int
    motion_threshold: float
    applied: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim leading/trailing static segments from a local LeRobot dataset."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--buffer-frames", type=int, default=5)
    parser.add_argument("--head-buffer-frames", type=int, default=None)
    parser.add_argument("--tail-buffer-frames", type=int, default=None)
    parser.add_argument("--motion-threshold", type=float, default=0.2)
    parser.add_argument("--sustain-window", type=int, default=5)
    parser.add_argument("--active-count", type=int, default=3)
    parser.add_argument("--min-static-frames", type=int, default=15)
    parser.add_argument("--min-episode-length", type=int, default=32)
    parser.add_argument("--video-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_all_data(data_root: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(path) for path in sorted(data_root.glob("chunk-*/*.parquet"))]
    if not parts:
        raise FileNotFoundError(f"No parquet data files found under {data_root}")
    return pd.concat(parts, ignore_index=True)


def load_all_episode_meta(episodes_root: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(path) for path in sorted(episodes_root.glob("chunk-*/*.parquet"))]
    if not parts:
        raise FileNotFoundError(f"No episode metadata files found under {episodes_root}")
    return pd.concat(parts, ignore_index=True)


def compute_motion_score(ep_df: pd.DataFrame) -> np.ndarray:
    action = np.stack(ep_df["action"].to_numpy())
    state = np.stack(ep_df["observation.state"].to_numpy())
    action_delta = np.zeros(len(ep_df), dtype=np.float32)
    state_delta = np.zeros(len(ep_df), dtype=np.float32)
    if len(ep_df) > 1:
        action_delta[1:] = np.linalg.norm(action[1:] - action[:-1], axis=1)
        state_delta[1:] = np.linalg.norm(state[1:] - state[:-1], axis=1)
    return np.maximum(action_delta, state_delta)


def find_active_start(score: np.ndarray, threshold: float, sustain_window: int, active_count: int) -> int:
    for i in range(len(score)):
        if int((score[i : i + sustain_window] >= threshold).sum()) >= active_count:
            return i
    return 0


def find_active_end(score: np.ndarray, threshold: float, sustain_window: int, active_count: int) -> int:
    for i in range(len(score) - 1, -1, -1):
        if int((score[max(0, i - sustain_window + 1) : i + 1] >= threshold).sum()) >= active_count:
            return i
    return len(score) - 1


def decide_trim(
    ep_df: pd.DataFrame,
    src_episode_index: int,
    dst_episode_index: int,
    threshold: float,
    sustain_window: int,
    active_count: int,
    min_static_frames: int,
    head_buffer_frames: int,
    tail_buffer_frames: int,
    min_episode_length: int,
) -> TrimDecision:
    score = compute_motion_score(ep_df)
    active_start = find_active_start(score, threshold, sustain_window, active_count)
    active_end = find_active_end(score, threshold, sustain_window, active_count)
    head_static = active_start
    tail_static = len(score) - 1 - active_end

    trim_start = max(0, active_start - head_buffer_frames) if head_static >= min_static_frames else 0
    trim_end = (
        min(len(score) - 1, active_end + tail_buffer_frames)
        if tail_static >= min_static_frames
        else len(score) - 1
    )

    if trim_end - trim_start + 1 < min_episode_length:
        trim_start = 0
        trim_end = len(score) - 1

    trimmed_length = trim_end - trim_start + 1
    applied = trim_start > 0 or trim_end < len(score) - 1

    return TrimDecision(
        source_episode_index=src_episode_index,
        output_episode_index=dst_episode_index,
        original_length=len(ep_df),
        trim_start=trim_start,
        trim_end=trim_end,
        trimmed_length=trimmed_length,
        head_static_frames=head_static,
        tail_static_frames=tail_static,
        motion_threshold=threshold,
        applied=applied,
    )


def compute_numeric_stats(values: np.ndarray) -> dict[str, list | float | int]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def recompute_stats(df: pd.DataFrame, src_stats: dict, feature_names: dict) -> dict:
    stats = {}
    for key, feature in feature_names.items():
        if key.startswith("observation.images."):
            stats[key] = src_stats[key]
            continue
        series = df[key]
        if feature["dtype"] in {"float32", "float64", "int64", "int32"}:
            if feature["shape"] == [1] or feature["shape"] == (1,):
                values = series.to_numpy()
            else:
                values = np.stack(series.to_numpy())
            stats[key] = compute_numeric_stats(values)
        else:
            stats[key] = src_stats[key]
    return stats


def link_or_copy_videos(src_root: Path, dst_root: Path, mode: str) -> None:
    src_videos = src_root / "videos"
    dst_videos = dst_root / "videos"
    if dst_videos.exists() or dst_videos.is_symlink():
        if dst_videos.is_symlink() or dst_videos.is_file():
            dst_videos.unlink()
        else:
            shutil.rmtree(dst_videos)
    if mode == "symlink":
        dst_videos.symlink_to(src_videos.resolve(), target_is_directory=True)
    else:
        shutil.copytree(src_videos, dst_videos)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    repo_id = args.repo_id or output_root.name
    head_buffer = args.buffer_frames if args.head_buffer_frames is None else args.head_buffer_frames
    tail_buffer = args.buffer_frames if args.tail_buffer_frames is None else args.tail_buffer_frames

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    info = json.loads((input_root / "meta" / "info.json").read_text())
    src_stats = json.loads((input_root / "meta" / "stats.json").read_text())
    tasks_df = pd.read_parquet(input_root / "meta" / "tasks.parquet")
    data_df = load_all_data(input_root / "data")
    episodes_df = load_all_episode_meta(input_root / "meta" / "episodes")
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)

    source_episode_indices = episodes_df["episode_index"].tolist()
    if args.max_episodes is not None:
        source_episode_indices = source_episode_indices[: args.max_episodes]
        episodes_df = episodes_df[episodes_df["episode_index"].isin(source_episode_indices)].reset_index(drop=True)
        data_df = data_df[data_df["episode_index"].isin(source_episode_indices)].reset_index(drop=True)

    new_rows: list[pd.DataFrame] = []
    new_episode_rows: list[dict] = []
    manifest: list[dict] = []
    global_index = 0
    fps = float(info["fps"])

    for dst_ep_idx, src_ep_idx in enumerate(source_episode_indices):
        ep_df = data_df[data_df["episode_index"] == src_ep_idx].copy().reset_index(drop=True)
        ep_meta = episodes_df[episodes_df["episode_index"] == src_ep_idx].iloc[0].to_dict()
        decision = decide_trim(
            ep_df=ep_df,
            src_episode_index=src_ep_idx,
            dst_episode_index=dst_ep_idx,
            threshold=args.motion_threshold,
            sustain_window=args.sustain_window,
            active_count=args.active_count,
            min_static_frames=args.min_static_frames,
            head_buffer_frames=head_buffer,
            tail_buffer_frames=tail_buffer,
            min_episode_length=args.min_episode_length,
        )

        kept = ep_df.iloc[decision.trim_start : decision.trim_end + 1].copy().reset_index(drop=True)
        kept["episode_index"] = dst_ep_idx
        kept["frame_index"] = np.arange(len(kept), dtype=np.int64)
        kept["timestamp"] = kept["frame_index"].to_numpy(dtype=np.float32) / fps
        kept["index"] = np.arange(global_index, global_index + len(kept), dtype=np.int64)
        global_index += len(kept)
        new_rows.append(kept)

        new_ep = dict(ep_meta)
        new_ep["episode_index"] = dst_ep_idx
        new_ep["length"] = len(kept)
        new_ep["data/chunk_index"] = 0
        new_ep["data/file_index"] = 0
        new_ep["dataset_from_index"] = int(kept["index"].iloc[0])
        new_ep["dataset_to_index"] = int(kept["index"].iloc[-1]) + 1
        new_ep["meta/episodes/chunk_index"] = 0
        new_ep["meta/episodes/file_index"] = 0

        for video_key in [k for k in info["features"] if k.startswith("observation.images.")]:
            from_key = f"videos/{video_key}/from_timestamp"
            to_key = f"videos/{video_key}/to_timestamp"
            original_from_ts = float(ep_meta[from_key])
            new_ep[from_key] = original_from_ts + decision.trim_start / fps
            new_ep[to_key] = original_from_ts + (decision.trim_end + 1) / fps

        new_episode_rows.append(new_ep)
        manifest.append(decision.__dict__)

    trimmed_df = pd.concat(new_rows, ignore_index=True)
    trimmed_episodes_df = pd.DataFrame(new_episode_rows)
    trimmed_episodes_df = trimmed_episodes_df.sort_values("episode_index").reset_index(drop=True)

    (output_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    trimmed_df.to_parquet(output_root / "data" / "chunk-000" / "file-000.parquet", index=False)
    trimmed_episodes_df.to_parquet(
        output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False
    )
    write_tasks(tasks_df, output_root)
    link_or_copy_videos(input_root, output_root, args.video_mode)

    info["total_episodes"] = int(trimmed_episodes_df["episode_index"].nunique())
    info["total_frames"] = int(len(trimmed_df))
    info["total_tasks"] = int(len(tasks_df))
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    info["data_files_size_in_mb"] = round(
        sum(p.stat().st_size for p in (output_root / "data").glob("chunk-*/*.parquet")) / (1024**2)
    )
    if args.video_mode == "copy":
        info["video_files_size_in_mb"] = round(
            sum(p.stat().st_size for p in (output_root / "videos").glob("**/*.mp4")) / (1024**2)
        )
    write_info(info, output_root)

    stats = recompute_stats(trimmed_df, src_stats, info["features"])
    write_stats(stats, output_root)

    trim_summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "repo_id": repo_id,
        "fps": info["fps"],
        "motion_threshold": args.motion_threshold,
        "sustain_window": args.sustain_window,
        "active_count": args.active_count,
        "min_static_frames": args.min_static_frames,
        "head_buffer_frames": head_buffer,
        "tail_buffer_frames": tail_buffer,
        "min_episode_length": args.min_episode_length,
        "video_mode": args.video_mode,
        "episodes": manifest,
        "total_frames_before": int(data_df.shape[0]),
        "total_frames_after": int(trimmed_df.shape[0]),
    }
    (output_root / "meta" / "trim_manifest.json").write_text(
        json.dumps(trim_summary, ensure_ascii=False, indent=2)
    )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "episodes": info["total_episodes"],
                "frames_before": int(data_df.shape[0]),
                "frames_after": int(trimmed_df.shape[0]),
                "trimmed_frames": int(data_df.shape[0] - trimmed_df.shape[0]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
