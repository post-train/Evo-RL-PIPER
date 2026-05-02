#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

HF_CACHE_ROOT = REPO_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_ROOT / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_ROOT / "hub"))
os.environ.setdefault("DATASETS_CACHE", str(HF_CACHE_ROOT / "datasets"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import write_info, write_stats, write_tasks  # noqa: E402


VIDEO_PREFIX = "videos/"
IMAGE_PREFIX = "observation.images."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample a local LeRobot dataset to a lower FPS and lower video resolution."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--target-width", type=int, default=320)
    parser.add_argument("--target-height", type=int, default=240)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg-preset", type=str, default="veryfast")
    parser.add_argument("--ffmpeg-crf", type=int, default=18)
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


def compute_numeric_stats(values: np.ndarray) -> dict[str, list | int]:
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
        if key.startswith(IMAGE_PREFIX):
            stats[key] = src_stats[key]
            continue
        series = df[key]
        if feature["dtype"] in {"float32", "float64", "int64", "int32", "bool"}:
            if feature["shape"] == [1] or feature["shape"] == (1,):
                values = series.to_numpy()
            else:
                values = np.stack(series.to_numpy())
            stats[key] = compute_numeric_stats(values)
        else:
            stats[key] = src_stats[key]
    return stats


def ffmpeg_reencode_segment(
    src_video: Path,
    dst_video: Path,
    from_timestamp: float,
    to_timestamp: float,
    target_fps: float,
    target_width: int,
    target_height: int,
    preset: str,
    crf: int,
) -> None:
    duration = max(0.0, to_timestamp - from_timestamp + (1.0 / 30.0))
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{from_timestamp:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(src_video),
        "-vf",
        (
            f"fps={target_fps:g},"
            f"scale={target_width}:{target_height}:flags=lanczos"
        ),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(dst_video),
    ]
    subprocess.run(cmd, check=True)


def select_episode_rows(ep_df: pd.DataFrame, original_fps: float, target_fps: float) -> pd.DataFrame:
    stride = original_fps / target_fps
    selected_positions = []
    next_pos = 0.0
    while round(next_pos) < len(ep_df):
        selected_positions.append(int(round(next_pos)))
        next_pos += stride
    selected_positions = np.unique(np.clip(selected_positions, 0, len(ep_df) - 1))
    selected = ep_df.iloc[selected_positions].copy().reset_index(drop=True)
    return selected


def update_info(info: dict, target_fps: float, target_width: int, target_height: int, total_episodes: int, total_frames: int) -> dict:
    out = json.loads(json.dumps(info))
    out["fps"] = target_fps
    out["total_episodes"] = total_episodes
    out["total_frames"] = total_frames
    out["video_files_size_in_mb"] = None
    out["data_files_size_in_mb"] = None
    for key, feature in out["features"].items():
        if feature["dtype"] != "video":
            continue
        feature["shape"] = [target_height, target_width, 3]
        if "info" in feature:
            feature["info"]["video.height"] = target_height
            feature["info"]["video.width"] = target_width
            feature["info"]["video.fps"] = target_fps
            feature["info"]["video.codec"] = "h264"
            feature["info"]["video.pix_fmt"] = "yuv420p"
    return out


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    repo_id = args.repo_id or output_root.name

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    info = json.loads((input_root / "meta" / "info.json").read_text())
    src_stats = json.loads((input_root / "meta" / "stats.json").read_text())
    tasks_df = pd.read_parquet(input_root / "meta" / "tasks.parquet")
    data_df = load_all_data(input_root / "data")
    episodes_df = load_all_episode_meta(input_root / "meta" / "episodes").sort_values("episode_index").reset_index(drop=True)

    original_fps = float(info["fps"])
    if args.target_fps <= 0 or args.target_fps > original_fps:
        raise ValueError(f"--target-fps must be in (0, {original_fps}]")

    video_feature_keys = [key for key, value in info["features"].items() if value["dtype"] == "video"]
    source_episode_indices = episodes_df["episode_index"].tolist()
    if args.max_episodes is not None:
        source_episode_indices = source_episode_indices[: args.max_episodes]
        episodes_df = episodes_df[episodes_df["episode_index"].isin(source_episode_indices)].reset_index(drop=True)
        data_df = data_df[data_df["episode_index"].isin(source_episode_indices)].reset_index(drop=True)

    new_rows: list[pd.DataFrame] = []
    new_episode_rows: list[dict] = []
    manifest: list[dict] = []
    global_index = 0

    for dst_ep_idx, src_ep_idx in enumerate(source_episode_indices):
        ep_df = data_df[data_df["episode_index"] == src_ep_idx].copy().reset_index(drop=True)
        ep_meta = episodes_df[episodes_df["episode_index"] == src_ep_idx].iloc[0].to_dict()
        kept = select_episode_rows(ep_df, original_fps=original_fps, target_fps=args.target_fps)
        kept["episode_index"] = dst_ep_idx
        kept["frame_index"] = np.arange(len(kept), dtype=np.int64)
        kept["timestamp"] = kept["frame_index"].to_numpy(dtype=np.float32) / args.target_fps
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

        for feature_key in video_feature_keys:
            prefix = f"{VIDEO_PREFIX}{feature_key}/"
            src_chunk = int(ep_meta[f"{prefix}chunk_index"])
            src_file = int(ep_meta[f"{prefix}file_index"])
            src_from = float(ep_meta[f"{prefix}from_timestamp"])
            src_to = float(ep_meta[f"{prefix}to_timestamp"])
            src_video = input_root / "videos" / feature_key / f"chunk-{src_chunk:03d}" / f"file-{src_file:03d}.mp4"
            dst_video = output_root / "videos" / feature_key / "chunk-000" / f"file-{dst_ep_idx:03d}.mp4"
            ffmpeg_reencode_segment(
                src_video=src_video,
                dst_video=dst_video,
                from_timestamp=src_from,
                to_timestamp=src_to,
                target_fps=args.target_fps,
                target_width=args.target_width,
                target_height=args.target_height,
                preset=args.ffmpeg_preset,
                crf=args.ffmpeg_crf,
            )
            new_ep[f"{prefix}chunk_index"] = 0
            new_ep[f"{prefix}file_index"] = dst_ep_idx
            new_ep[f"{prefix}from_timestamp"] = 0.0
            new_ep[f"{prefix}to_timestamp"] = float(max(0, len(kept) - 1)) / args.target_fps

        new_episode_rows.append(new_ep)
        manifest.append(
            {
                "source_episode_index": int(src_ep_idx),
                "output_episode_index": dst_ep_idx,
                "original_length": int(len(ep_df)),
                "downsampled_length": int(len(kept)),
                "source_fps": original_fps,
                "target_fps": args.target_fps,
            }
        )

    output_df = pd.concat(new_rows, ignore_index=True)
    output_df = output_df[data_df.columns.tolist()]
    out_data_dir = output_root / "data" / "chunk-000"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(out_data_dir / "file-000.parquet", index=False)

    output_episodes_df = pd.DataFrame(new_episode_rows)
    out_episode_dir = output_root / "meta" / "episodes" / "chunk-000"
    out_episode_dir.mkdir(parents=True, exist_ok=True)
    output_episodes_df.to_parquet(out_episode_dir / "file-000.parquet", index=False)

    out_stats = recompute_stats(output_df, src_stats=src_stats, feature_names=info["features"])
    out_info = update_info(
        info=info,
        target_fps=args.target_fps,
        target_width=args.target_width,
        target_height=args.target_height,
        total_episodes=len(output_episodes_df),
        total_frames=len(output_df),
    )

    write_tasks(tasks_df, output_root)
    write_stats(out_stats, output_root)
    write_info(out_info, output_root)
    (output_root / "meta" / "downsample_manifest.json").write_text(
        json.dumps(
            {
                "input_root": str(input_root),
                "output_root": str(output_root),
                "repo_id": repo_id,
                "source_fps": original_fps,
                "target_fps": args.target_fps,
                "target_width": args.target_width,
                "target_height": args.target_height,
                "episodes": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"downsampled_dataset: episodes={len(output_episodes_df)} "
        f"frames={len(output_df)} fps={args.target_fps} "
        f"resolution={args.target_width}x{args.target_height}"
    )


if __name__ == "__main__":
    main()
