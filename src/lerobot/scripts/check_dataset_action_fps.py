#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def iter_parquet_files(dataset_root: Path) -> list[Path]:
    data_root = dataset_root / "data"
    return sorted(data_root.glob("chunk-*/file-*.parquet"))


def load_dataset_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return {}
    return json.loads(info_path.read_text())


def load_episode_timestamps(dataset_root: Path) -> dict[int, list[np.ndarray]]:
    episode_to_chunks: dict[int, list[np.ndarray]] = {}

    for parquet_path in iter_parquet_files(dataset_root):
        table = pq.read_table(parquet_path, columns=["episode_index", "timestamp"])
        episode_indices = table["episode_index"].to_numpy()
        timestamps = table["timestamp"].to_numpy()

        unique_eps = np.unique(episode_indices)
        for ep_idx in unique_eps:
            mask = episode_indices == ep_idx
            episode_to_chunks.setdefault(int(ep_idx), []).append(np.asarray(timestamps[mask], dtype=np.float64))

    return episode_to_chunks


def summarize_episode(ep_idx: int, timestamps: np.ndarray) -> dict[str, float | int]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    timestamps = timestamps[np.isfinite(timestamps)]
    timestamps.sort()

    frame_count = int(timestamps.size)
    if frame_count == 0:
        return {
            "episode_index": ep_idx,
            "frames": 0,
            "duration_s": 0.0,
            "mean_dt_ms": float("nan"),
            "median_dt_ms": float("nan"),
            "p95_dt_ms": float("nan"),
            "effective_fps": float("nan"),
            "median_inst_fps": float("nan"),
        }

    if frame_count == 1:
        return {
            "episode_index": ep_idx,
            "frames": 1,
            "duration_s": 0.0,
            "mean_dt_ms": float("nan"),
            "median_dt_ms": float("nan"),
            "p95_dt_ms": float("nan"),
            "effective_fps": float("nan"),
            "median_inst_fps": float("nan"),
        }

    dt = np.diff(timestamps)
    positive_dt = dt[dt > 0]
    duration_s = float(timestamps[-1] - timestamps[0])

    if positive_dt.size == 0 or duration_s <= 0:
        return {
            "episode_index": ep_idx,
            "frames": frame_count,
            "duration_s": duration_s,
            "mean_dt_ms": float("nan"),
            "median_dt_ms": float("nan"),
            "p95_dt_ms": float("nan"),
            "effective_fps": float("nan"),
            "median_inst_fps": float("nan"),
        }

    inst_fps = 1.0 / positive_dt
    return {
        "episode_index": ep_idx,
        "frames": frame_count,
        "duration_s": duration_s,
        "mean_dt_ms": float(np.mean(positive_dt) * 1e3),
        "median_dt_ms": float(np.median(positive_dt) * 1e3),
        "p95_dt_ms": float(np.percentile(positive_dt, 95) * 1e3),
        "effective_fps": float((frame_count - 1) / duration_s),
        "median_inst_fps": float(np.median(inst_fps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check per-episode action frequency from a LeRobot dataset.")
    parser.add_argument("dataset_root", type=Path, help="Path to dataset root, e.g. /path/to/datasets3")
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print all episode rows instead of only the first/last few plus summary.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    info = load_dataset_info(dataset_root)
    nominal_fps = info.get("fps")

    episode_to_chunks = load_episode_timestamps(dataset_root)
    episode_stats = []
    for ep_idx in sorted(episode_to_chunks):
        timestamps = np.concatenate(episode_to_chunks[ep_idx])
        episode_stats.append(summarize_episode(ep_idx, timestamps))

    print(f"dataset: {dataset_root}")
    if nominal_fps is not None:
        print(f"nominal_fps(meta): {nominal_fps}")
    print(f"episodes_found: {len(episode_stats)}")

    if not episode_stats:
        return

    rows = episode_stats
    if not args.show_all and len(rows) > 12:
        rows_to_print = rows[:6] + rows[-6:]
    else:
        rows_to_print = rows

    print(
        "episode  frames  duration_s  mean_dt_ms  median_dt_ms  p95_dt_ms  effective_fps  median_inst_fps"
    )
    for row in rows_to_print:
        print(
            f"{row['episode_index']:>7}  "
            f"{row['frames']:>6}  "
            f"{row['duration_s']:>10.3f}  "
            f"{row['mean_dt_ms']:>10.3f}  "
            f"{row['median_dt_ms']:>12.3f}  "
            f"{row['p95_dt_ms']:>9.3f}  "
            f"{row['effective_fps']:>13.3f}  "
            f"{row['median_inst_fps']:>15.3f}"
        )

    effective_fps = np.array([row["effective_fps"] for row in rows], dtype=np.float64)
    mean_dt_ms = np.array([row["mean_dt_ms"] for row in rows], dtype=np.float64)
    median_dt_ms = np.array([row["median_dt_ms"] for row in rows], dtype=np.float64)

    valid_effective = effective_fps[np.isfinite(effective_fps)]
    valid_mean_dt = mean_dt_ms[np.isfinite(mean_dt_ms)]
    valid_median_dt = median_dt_ms[np.isfinite(median_dt_ms)]

    print("\nsummary:")
    if valid_effective.size:
        print(f"  effective_fps mean={valid_effective.mean():.3f} median={np.median(valid_effective):.3f} min={valid_effective.min():.3f} max={valid_effective.max():.3f}")
    if valid_mean_dt.size:
        print(f"  mean_dt_ms    mean={valid_mean_dt.mean():.3f} median={np.median(valid_mean_dt):.3f} min={valid_mean_dt.min():.3f} max={valid_mean_dt.max():.3f}")
    if valid_median_dt.size:
        print(f"  median_dt_ms  mean={valid_median_dt.mean():.3f} median={np.median(valid_median_dt):.3f} min={valid_median_dt.min():.3f} max={valid_median_dt.max():.3f}")


if __name__ == "__main__":
    main()
