#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import unflatten_dict, write_info, write_stats


def _unwrap_numeric(value):
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        if value.ndim == 0:
            return _unwrap_numeric(value.item())
        return np.asarray([_unwrap_numeric(item) for item in value.tolist()])
    if isinstance(value, list):
        return np.asarray([_unwrap_numeric(item) for item in value])
    if hasattr(value, "to_pylist"):
        return _unwrap_numeric(value.to_pylist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _unwrap_numeric(value.item())
        except Exception:
            pass
    return value


def _normalize_stat_value(value):
    normalized = _unwrap_numeric(value)
    if np.isscalar(normalized):
        return np.asarray([normalized], dtype=np.float64)
    arr = np.asarray(normalized)
    if arr.dtype == object:
        arr = arr.astype(np.float64)
    return arr


def _reshape_feature_stats(stats: dict, features: dict) -> dict:
    reshaped = {}
    for feature_key, feature_stats in stats.items():
        feature_info = features.get(feature_key, {})
        is_image_like = feature_info.get("dtype") == "video"
        reshaped[feature_key] = {}
        for stat_key, value in feature_stats.items():
            arr = np.asarray(value)
            if stat_key != "count" and is_image_like and arr.ndim == 1:
                arr = arr.reshape(-1, 1, 1)
            reshaped[feature_key][stat_key] = arr
    return reshaped


def _backup_file(src: Path, backup_root: Path) -> None:
    dst = backup_root / src.relative_to(src.anchor)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_episode_metadata(dataset_root: Path) -> tuple[dict[int, dict], list[Path]]:
    episodes_dir = dataset_root / "meta" / "episodes"
    episode_rows: dict[int, dict] = {}
    episode_files: list[Path] = []
    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        episode_files.append(parquet_path)
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            ep_idx = int(row["episode_index"])
            if ep_idx in episode_rows:
                raise ValueError(f"Duplicate episode metadata rows found for episode_index={ep_idx}")
            episode_rows[ep_idx] = {
                "episode_file": parquet_path,
                "meta_file_index": int(row["meta/episodes/file_index"]),
                "meta_chunk_index": int(row["meta/episodes/chunk_index"]),
                "data_file_index": int(row["data/file_index"]),
                "data_chunk_index": int(row["data/chunk_index"]),
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
                "length": int(row["length"]),
                "stats_row": row.to_dict(),
            }
    return episode_rows, episode_files


def _rewrite_duplicate_episode_rows(dataset_root: Path, episode_index: int, backup_root: Path) -> list[str]:
    episode_rows, _ = _load_episode_metadata(dataset_root)
    if episode_index not in episode_rows:
        raise ValueError(f"episode_index={episode_index} not found in meta/episodes")

    canonical = episode_rows[episode_index]
    canonical_chunk = canonical["data_chunk_index"]
    canonical_file = canonical["data_file_index"]
    canonical_from = canonical["dataset_from_index"]
    canonical_to = canonical["dataset_to_index"]

    actions: list[str] = []
    data_dir = dataset_root / "data"
    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        if "episode_index" not in df.columns:
            continue

        rel = parquet_path.relative_to(dataset_root)
        chunk_idx = int(rel.parts[1].split("-")[1])
        file_idx = int(rel.stem.split("-")[1])
        mask = df["episode_index"] == episode_index
        if not mask.any():
            continue

        keep_mask = pd.Series(True, index=df.index)
        if chunk_idx == canonical_chunk and file_idx == canonical_file:
            canonical_rows = mask & df["index"].between(canonical_from, canonical_to - 1)
            dropped = int((mask & ~canonical_rows).sum())
            keep_mask &= ~mask | canonical_rows
            action = (
                f"kept canonical rows in {rel} for episode {episode_index}, "
                f"dropped {dropped} non-canonical row(s)"
            )
        else:
            dropped = int(mask.sum())
            keep_mask &= ~mask
            action = f"removed {dropped} duplicate row(s) for episode {episode_index} from {rel}"

        if dropped > 0:
            _backup_file(parquet_path, backup_root)
            cleaned = df.loc[keep_mask].reset_index(drop=True)
            cleaned.to_parquet(parquet_path, index=False)
            actions.append(action)

    return actions


def _recompute_global_indices(dataset_root: Path, backup_root: Path) -> dict[int, dict[str, int]]:
    episode_ranges: dict[int, dict[str, int]] = {}
    next_index = 0

    for parquet_path in sorted((dataset_root / "data").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        if len(df) == 0:
            continue

        expected_index = np.arange(next_index, next_index + len(df), dtype=np.int64)
        needs_rewrite = "index" not in df.columns or not np.array_equal(df["index"].to_numpy(), expected_index)
        if needs_rewrite:
            _backup_file(parquet_path, backup_root)
            df["index"] = expected_index
            df.to_parquet(parquet_path, index=False)

        for ep_idx, group in df.groupby("episode_index", sort=False):
            ep_idx = int(ep_idx)
            episode_ranges[ep_idx] = {
                "dataset_from_index": int(group["index"].iloc[0]),
                "dataset_to_index": int(group["index"].iloc[-1]) + 1,
                "length": int(len(group)),
            }
        next_index += len(df)

    return episode_ranges


def _rewrite_episode_metadata(dataset_root: Path, episode_ranges: dict[int, dict[str, int]], backup_root: Path) -> None:
    episodes_dir = dataset_root / "meta" / "episodes"
    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        updated = False
        for row_idx in range(len(df)):
            ep_idx = int(df.at[row_idx, "episode_index"])
            if ep_idx not in episode_ranges:
                raise ValueError(f"Episode metadata references missing episode_index={ep_idx}")
            payload = episode_ranges[ep_idx]
            for key in ("dataset_from_index", "dataset_to_index", "length"):
                if int(df.at[row_idx, key]) != int(payload[key]):
                    df.at[row_idx, key] = int(payload[key])
                    updated = True
        if updated:
            _backup_file(parquet_path, backup_root)
            df.to_parquet(parquet_path, index=False)


def _rebuild_stats_from_episode_metadata(dataset_root: Path) -> dict:
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    per_episode_stats = []
    for parquet_path in sorted((dataset_root / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            flat_stats = {}
            for column, value in row.items():
                if not column.startswith("stats/"):
                    continue
                flat_stats[column.removeprefix("stats/")] = _normalize_stat_value(value)
            per_episode_stats.append(_reshape_feature_stats(unflatten_dict(flat_stats), features))

    if not per_episode_stats:
        raise ValueError("No per-episode stats found in meta/episodes")
    return aggregate_stats(per_episode_stats)


def _rewrite_info(dataset_root: Path, backup_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_frames = 0
    for parquet_path in sorted((dataset_root / "data").rglob("*.parquet")):
        total_frames += pq.read_metadata(parquet_path).num_rows

    total_episodes = 0
    for parquet_path in sorted((dataset_root / "meta" / "episodes").rglob("*.parquet")):
        total_episodes += pq.read_metadata(parquet_path).num_rows

    info["total_frames"] = int(total_frames)
    info["total_episodes"] = int(total_episodes)
    if "splits" in info and isinstance(info["splits"], dict) and "train" in info["splits"]:
        info["splits"]["train"] = f"0:{total_episodes}"

    _backup_file(info_path, backup_root)
    write_info(info, dataset_root)
    return info


def _remove_orphan_tmp_videos(dataset_root: Path, episode_index: int) -> list[str]:
    removed: list[str] = []
    pattern = f"*_{episode_index:03d}.mp4"
    for path in sorted(dataset_root.glob(f"tmp*/{pattern}")):
        path.unlink()
        removed.append(str(path.relative_to(dataset_root)))

    for tmp_dir in sorted(dataset_root.glob("tmp*")):
        if tmp_dir.is_dir() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean duplicate episode rows from datasets2 and rewrite metadata.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/szk/szk/Evo-RL/datasets2"),
        help="Path to the LeRobot dataset root.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=26,
        help="Episode index whose duplicate rows should be removed.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    backup_root = dataset_root / "meta" / "cleanup_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root.mkdir(parents=True, exist_ok=True)

    actions = _rewrite_duplicate_episode_rows(dataset_root, args.episode_index, backup_root)
    episode_ranges = _recompute_global_indices(dataset_root, backup_root)
    _rewrite_episode_metadata(dataset_root, episode_ranges, backup_root)
    stats = _rebuild_stats_from_episode_metadata(dataset_root)
    stats_path = dataset_root / "meta" / "stats.json"
    _backup_file(stats_path, backup_root)
    write_stats(stats, dataset_root)
    info = _rewrite_info(dataset_root, backup_root)
    removed_tmp = _remove_orphan_tmp_videos(dataset_root, args.episode_index)

    print("Cleanup completed.")
    print(f"Dataset root: {dataset_root}")
    print(f"Backup root: {backup_root}")
    print(f"total_episodes={info['total_episodes']}")
    print(f"total_frames={info['total_frames']}")
    print("Data edits:")
    for item in actions:
        print(f"  - {item}")
    print("Removed tmp files:")
    for item in removed_tmp:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
