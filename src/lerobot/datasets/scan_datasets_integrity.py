#!/usr/bin/env python3

# python scan_datasets2_integrity.py \
#     --dataset-root /home/szk/szk/Evo-RL/datasets3 \
#     --output-prefix /home/szk/szk/Evo-RL/datasets3_report

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq


@dataclass
class Issue:
    severity: str
    category: str
    path: str
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "severity": self.severity,
            "category": self.category,
            "path": self.path,
            "message": self.message,
        }
        if self.details:
            data["details"] = self.details
        return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_numpy_2d(column: list[Any], expected_name: str, issues: list[Issue], path: Path) -> np.ndarray | None:
    try:
        arr = np.asarray(column, dtype=np.float64)
    except Exception as err:
        issues.append(
            Issue(
                severity="error",
                category="parquet_decode",
                path=str(path),
                message=f"Failed to convert column `{expected_name}` to numpy array.",
                details={"error": str(err)},
            )
        )
        return None

    if arr.ndim != 2:
        issues.append(
            Issue(
                severity="error",
                category="shape",
                path=str(path),
                message=f"Column `{expected_name}` is expected to be 2D, got ndim={arr.ndim}.",
                details={"shape": list(arr.shape)},
            )
        )
        return None
    return arr


def inspect_data_parquet(path: Path, fps: float, issues: list[Issue]) -> dict[int, dict[str, Any]]:
    required_columns = [
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
    ]
    per_episode_ranges: dict[int, dict[str, Any]] = {}

    try:
        table = pq.read_table(path)
    except Exception as err:
        issues.append(
            Issue(
                severity="error",
                category="parquet_read",
                path=str(path),
                message="Failed to read parquet file.",
                details={"error": str(err)},
            )
        )
        return per_episode_ranges

    columns = set(table.column_names)
    for name in required_columns:
        if name not in columns:
            issues.append(
                Issue(
                    severity="error",
                    category="missing_column",
                    path=str(path),
                    message=f"Missing required column `{name}`.",
                )
            )

    if not set(required_columns).issubset(columns):
        return per_episode_ranges

    data = table.select(required_columns).to_pydict()

    action = _as_numpy_2d(data["action"], "action", issues, path)
    state = _as_numpy_2d(data["observation.state"], "observation.state", issues, path)
    if action is not None:
        if not np.isfinite(action).all():
            issues.append(
                Issue(
                    severity="error",
                    category="nan_inf",
                    path=str(path),
                    message="`action` contains NaN or Inf.",
                )
            )
    if state is not None:
        if not np.isfinite(state).all():
            issues.append(
                Issue(
                    severity="error",
                    category="nan_inf",
                    path=str(path),
                    message="`observation.state` contains NaN or Inf.",
                )
            )

    timestamps = np.asarray(data["timestamp"], dtype=np.float64)
    frame_index = np.asarray(data["frame_index"], dtype=np.int64)
    episode_index = np.asarray(data["episode_index"], dtype=np.int64)
    dataset_index = np.asarray(data["index"], dtype=np.int64)

    if not np.isfinite(timestamps).all():
        issues.append(
            Issue(
                severity="error",
                category="nan_inf",
                path=str(path),
                message="`timestamp` contains NaN or Inf.",
            )
        )
        return per_episode_ranges

    if len(dataset_index) > 1 and not np.all(np.diff(dataset_index) == 1):
        issues.append(
            Issue(
                severity="warning",
                category="index_gap",
                path=str(path),
                message="Global `index` is not strictly consecutive inside parquet file.",
                details={
                    "first_bad_position": int(np.where(np.diff(dataset_index) != 1)[0][0]),
                },
            )
        )

    unique_eps = np.unique(episode_index)
    expected_dt = 1.0 / fps if fps > 0 else None
    for ep in unique_eps:
        mask = episode_index == ep
        ep_ts = timestamps[mask]
        ep_frame_idx = frame_index[mask]
        ep_global_idx = dataset_index[mask]
        per_episode_ranges[int(ep)] = {
            "dataset_from_index": int(ep_global_idx[0]),
            "dataset_to_index": int(ep_global_idx[-1]) + 1,
            "length": int(mask.sum()),
            "from_timestamp": float(ep_ts[0]),
            "to_timestamp": float(ep_ts[-1]),
        }

        if len(ep_frame_idx) > 1 and not np.all(np.diff(ep_frame_idx) == 1):
            issues.append(
                Issue(
                    severity="warning",
                    category="frame_index_gap",
                    path=str(path),
                    message=f"`frame_index` is not consecutive for episode {int(ep)}.",
                    details={"episode_index": int(ep)},
                )
            )

        if len(ep_ts) > 1:
            dt = np.diff(ep_ts)
            if np.any(dt <= 0):
                issues.append(
                    Issue(
                        severity="error",
                        category="timestamp_order",
                        path=str(path),
                        message=f"Timestamps are not strictly increasing for episode {int(ep)}.",
                        details={"episode_index": int(ep)},
                    )
                )
            if expected_dt is not None:
                max_abs_err = float(np.max(np.abs(dt - expected_dt)))
                if max_abs_err > max(1e-3, expected_dt * 0.25):
                    issues.append(
                        Issue(
                            severity="warning",
                            category="timestamp_spacing",
                            path=str(path),
                            message=f"Timestamp spacing deviates from expected fps for episode {int(ep)}.",
                            details={
                                "episode_index": int(ep),
                                "expected_dt": expected_dt,
                                "max_abs_error": max_abs_err,
                            },
                        )
                    )

    return per_episode_ranges


def inspect_episode_metadata(
    path: Path,
    expected_video_keys: list[str],
    issues: list[Issue],
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    try:
        table = pq.read_table(path)
    except Exception as err:
        issues.append(
            Issue(
                severity="error",
                category="parquet_read",
                path=str(path),
                message="Failed to read episode metadata parquet.",
                details={"error": str(err)},
            )
        )
        return result

    data = table.to_pydict()
    rows = table.num_rows
    for row_idx in range(rows):
        ep_idx = int(data["episode_index"][row_idx])
        result[ep_idx] = {
            "path": str(path),
            "length": int(data["length"][row_idx]),
            "dataset_from_index": int(data["dataset_from_index"][row_idx]),
            "dataset_to_index": int(data["dataset_to_index"][row_idx]),
            "videos": {},
        }
        for video_key in expected_video_keys:
            prefix = f"videos/{video_key}"
            chunk_col = f"{prefix}/chunk_index"
            file_col = f"{prefix}/file_index"
            from_col = f"{prefix}/from_timestamp"
            to_col = f"{prefix}/to_timestamp"
            if chunk_col not in data or file_col not in data or from_col not in data or to_col not in data:
                issues.append(
                    Issue(
                        severity="error",
                        category="missing_video_metadata",
                        path=str(path),
                        message=f"Episode metadata missing columns for video key `{video_key}`.",
                    )
                )
                continue
            result[ep_idx]["videos"][video_key] = {
                "chunk_index": int(data[chunk_col][row_idx]),
                "file_index": int(data[file_col][row_idx]),
                "from_timestamp": _safe_float(data[from_col][row_idx]),
                "to_timestamp": _safe_float(data[to_col][row_idx]),
            }
    return result


def inspect_video_file(
    dataset_root: Path,
    video_key: str,
    chunk_index: int,
    file_index: int,
    from_timestamp: float | None,
    to_timestamp: float | None,
    fps: float,
    issues: list[Issue],
) -> None:
    video_path = dataset_root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
    if not video_path.exists():
        issues.append(
            Issue(
                severity="error",
                category="missing_video",
                path=str(video_path),
                message="Referenced video file does not exist.",
            )
        )
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        issues.append(
            Issue(
                severity="error",
                category="video_open",
                path=str(video_path),
                message="OpenCV failed to open video file.",
            )
        )
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / video_fps if frame_count > 0 and video_fps > 0 else None
    cap.release()

    if frame_count <= 0:
        issues.append(
            Issue(
                severity="error",
                category="video_frames",
                path=str(video_path),
                message="Video has zero readable frames.",
            )
        )
        return

    if width <= 0 or height <= 0:
        issues.append(
            Issue(
                severity="error",
                category="video_shape",
                path=str(video_path),
                message="Video reports invalid width/height.",
                details={"width": width, "height": height},
            )
        )

    if fps > 0 and video_fps > 0 and abs(video_fps - fps) > 1.0:
        issues.append(
            Issue(
                severity="warning",
                category="video_fps",
                path=str(video_path),
                message="Video fps deviates from dataset fps.",
                details={"dataset_fps": fps, "video_fps": video_fps},
            )
        )

    if from_timestamp is not None and to_timestamp is not None and duration is not None:
        expected_duration = max(0.0, to_timestamp - from_timestamp)
        # Allow one second slack plus two frames.
        slack = max(1.0, 2.0 / fps if fps > 0 else 1.0)
        if duration + slack < to_timestamp:
            issues.append(
                Issue(
                    severity="error",
                    category="video_duration",
                    path=str(video_path),
                    message="Video appears too short for referenced episode timestamps.",
                    details={
                        "from_timestamp": from_timestamp,
                        "to_timestamp": to_timestamp,
                        "video_duration": duration,
                        "expected_episode_span": expected_duration,
                    },
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan datasets2 for broken videos and invalid joint/action data.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/szk/szk/Evo-RL/datasets2"),
        help="Path to LeRobot dataset root.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/home/szk/szk/Evo-RL/datasets2_integrity_report"),
        help="Prefix for output files (.json and .txt will be appended).",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json at {info_path}")

    info = _load_json(info_path)
    fps = float(info.get("fps", 0))
    features = info.get("features", {})
    video_keys = sorted([key for key in features if key.startswith("observation.images.")])

    issues: list[Issue] = []

    data_ranges: dict[int, dict[str, Any]] = {}
    for parquet_path in sorted((dataset_root / "data").rglob("*.parquet")):
        per_file = inspect_data_parquet(parquet_path, fps, issues)
        for ep_idx, payload in per_file.items():
            if ep_idx in data_ranges:
                issues.append(
                    Issue(
                        severity="warning",
                        category="duplicate_episode_data",
                        path=str(parquet_path),
                        message=f"Episode {ep_idx} appears in multiple data parquet files.",
                    )
                )
            data_ranges[ep_idx] = payload

    episode_meta: dict[int, dict[str, Any]] = {}
    for parquet_path in sorted((dataset_root / "meta" / "episodes").rglob("*.parquet")):
        per_file = inspect_episode_metadata(parquet_path, video_keys, issues)
        for ep_idx, payload in per_file.items():
            if ep_idx in episode_meta:
                issues.append(
                    Issue(
                        severity="warning",
                        category="duplicate_episode_meta",
                        path=str(parquet_path),
                        message=f"Episode {ep_idx} appears in multiple episode metadata files.",
                    )
                )
            episode_meta[ep_idx] = payload

    for ep_idx, meta in sorted(episode_meta.items()):
        if ep_idx not in data_ranges:
            issues.append(
                Issue(
                    severity="error",
                    category="missing_episode_data",
                    path=meta["path"],
                    message=f"Episode metadata exists but data rows are missing for episode {ep_idx}.",
                )
            )
            continue

        data_info = data_ranges[ep_idx]
        if data_info["length"] != meta["length"]:
            issues.append(
                Issue(
                    severity="warning",
                    category="episode_length",
                    path=meta["path"],
                    message=f"Episode {ep_idx} length mismatch between data and metadata.",
                    details={"data_length": data_info["length"], "meta_length": meta["length"]},
                )
            )
        if data_info["dataset_from_index"] != meta["dataset_from_index"] or data_info["dataset_to_index"] != meta["dataset_to_index"]:
            issues.append(
                Issue(
                    severity="warning",
                    category="episode_index_range",
                    path=meta["path"],
                    message=f"Episode {ep_idx} dataset index range mismatch.",
                    details={
                        "data": {
                            "from": data_info["dataset_from_index"],
                            "to": data_info["dataset_to_index"],
                        },
                        "meta": {
                            "from": meta["dataset_from_index"],
                            "to": meta["dataset_to_index"],
                        },
                    },
                )
            )

        for video_key, video_info in meta["videos"].items():
            inspect_video_file(
                dataset_root=dataset_root,
                video_key=video_key,
                chunk_index=video_info["chunk_index"],
                file_index=video_info["file_index"],
                from_timestamp=video_info["from_timestamp"],
                to_timestamp=video_info["to_timestamp"],
                fps=fps,
                issues=issues,
            )

    orphan_tmp_files = sorted(dataset_root.glob("tmp*/**/*"))
    for path in orphan_tmp_files:
        if path.is_file():
            issues.append(
                Issue(
                    severity="warning",
                    category="orphan_tmp",
                    path=str(path),
                    message="Temporary file exists inside dataset root; this may indicate interrupted recording/encoding.",
                )
            )

    issues_by_path: dict[str, list[Issue]] = defaultdict(list)
    for issue in issues:
        issues_by_path[issue.path].append(issue)

    output_json = args.output_prefix.with_suffix(".json")
    output_txt = args.output_prefix.with_suffix(".txt")

    report = {
        "dataset_root": str(dataset_root),
        "fps": fps,
        "video_keys": video_keys,
        "num_issues": len(issues),
        "issues": [issue.to_dict() for issue in issues],
        "problem_files": sorted(issues_by_path),
    }
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"Dataset root: {dataset_root}",
        f"Dataset fps: {fps}",
        f"Video keys: {video_keys}",
        f"Total issues: {len(issues)}",
        "",
        "Problem files:",
    ]
    lines.extend(sorted(issues_by_path))
    lines.extend(["", "Detailed issues:"])
    for issue in issues:
        detail_str = f" | details={json.dumps(issue.details, ensure_ascii=False)}" if issue.details else ""
        lines.append(f"[{issue.severity}] {issue.category} | {issue.path} | {issue.message}{detail_str}")
    output_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote JSON report: {output_json}")
    print(f"Wrote text report: {output_txt}")
    print(f"Found {len(issues)} issue(s) across {len(issues_by_path)} file(s).")


if __name__ == "__main__":
    main()
