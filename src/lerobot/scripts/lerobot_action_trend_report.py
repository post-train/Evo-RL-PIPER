#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generate a detailed action-trend report for a local LeRobot v3 dataset.

Examples:

```bash
python src/lerobot/scripts/lerobot_action_trend_report.py --dataset /path/to/dataset
python src/lerobot/scripts/lerobot_action_trend_report.py --dataset datasets3 --json
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.scripts.lerobot_dataset_report import resolve_dataset_root


def _to_numpy_matrix(series: pd.Series) -> np.ndarray:
    values = series.to_list()
    if len(values) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_median(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _feature_names(info: dict[str, Any], key: str) -> list[str]:
    feature = info.get("features", {}).get(key, {})
    names = feature.get("names")
    if names:
        return list(names)
    shape = feature.get("shape", [])
    if len(shape) == 1:
        return [f"{key}[{i}]" for i in range(int(shape[0]))]
    return []


def _find_parquet_files(dataset_root: Path) -> list[Path]:
    data_root = dataset_root / "data"
    return sorted(data_root.rglob("*.parquet"))


def _load_dataframe(dataset_root: Path) -> pd.DataFrame:
    parquet_files = _find_parquet_files(dataset_root)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet data files found under {dataset_root / 'data'}")
    return pd.concat((pd.read_parquet(path) for path in parquet_files), ignore_index=True)


def _compute_episode_lengths(episode_index: np.ndarray) -> dict[int, int]:
    unique_ids, counts = np.unique(episode_index, return_counts=True)
    return {int(ep): int(count) for ep, count in zip(unique_ids, counts, strict=True)}


def _adjacent_same_episode_mask(episode_index: np.ndarray, step_ahead: int = 1) -> np.ndarray:
    if step_ahead <= 0:
        raise ValueError(f"step_ahead must be positive, got {step_ahead}")
    if len(episode_index) <= step_ahead:
        return np.zeros((0,), dtype=bool)
    return episode_index[:-step_ahead] == episode_index[step_ahead:]


def _hold_current_state_baseline(
    actions: np.ndarray,
    states: np.ndarray,
    episode_index: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
    n_obs_steps: int,
    n_action_steps: int,
) -> dict[str, float]:
    if len(actions) < n_obs_steps + n_action_steps:
        return {"raw_l1_mean": 0.0, "raw_l1_median": 0.0, "norm_l1_mean": 0.0, "norm_l1_median": 0.0}

    norm_scale = np.where((action_max - action_min) == 0, 1.0, action_max - action_min)

    raw_losses: list[float] = []
    norm_losses: list[float] = []

    max_start = len(actions) - (n_obs_steps - 1 + n_action_steps) + 1
    for start in range(max_start):
        end = start + n_obs_steps - 1 + n_action_steps
        if not np.all(episode_index[start:end] == episode_index[start]):
            continue

        current_state = states[start + n_obs_steps - 1]
        future_actions = actions[start + n_obs_steps - 1 : start + n_obs_steps - 1 + n_action_steps]
        baseline = np.repeat(current_state[None, :], n_action_steps, axis=0)

        raw_losses.append(float(np.mean(np.abs(baseline - future_actions))))

        future_actions_norm = (future_actions - action_min) / norm_scale
        baseline_norm = (baseline - action_min) / norm_scale
        norm_losses.append(float(np.mean(np.abs(baseline_norm - future_actions_norm))))

    raw_arr = np.asarray(raw_losses, dtype=np.float32)
    norm_arr = np.asarray(norm_losses, dtype=np.float32)
    return {
        "raw_l1_mean": _safe_mean(raw_arr),
        "raw_l1_median": _safe_median(raw_arr),
        "norm_l1_mean": _safe_mean(norm_arr),
        "norm_l1_median": _safe_median(norm_arr),
    }


def build_report(dataset_root: Path, n_obs_steps: int, n_action_steps: int, max_ahead: int) -> dict[str, Any]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
    df = _load_dataframe(dataset_root)

    required_columns = {"action", "observation.state", "episode_index"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    actions = _to_numpy_matrix(df["action"])
    states = _to_numpy_matrix(df["observation.state"])
    episode_index = df["episode_index"].to_numpy(dtype=np.int64)

    if actions.shape != states.shape:
        raise ValueError(
            f"Shape mismatch between action {tuple(actions.shape)} and observation.state {tuple(states.shape)}"
        )

    action_names = _feature_names(info, "action")
    fps = float(info.get("fps", 0) or 0)
    episode_lengths = _compute_episode_lengths(episode_index)

    adjacent_mask = _adjacent_same_episode_mask(episode_index, step_ahead=1)
    delta_actions = actions[1:] - actions[:-1]
    delta_actions_same_ep = delta_actions[adjacent_mask]
    delta_norms = np.linalg.norm(delta_actions_same_ep, axis=1) if delta_actions_same_ep.size else np.zeros((0,))

    same_prev = np.all(np.isclose(actions[1:], actions[:-1], atol=1e-6), axis=1) if len(actions) > 1 else np.zeros((0,))
    same_prev_same_ep = same_prev[adjacent_mask]

    action_state_diff = np.abs(actions - states)
    action_state_l2 = np.linalg.norm(actions - states, axis=1) if len(actions) else np.zeros((0,))

    lookahead_stats: list[dict[str, float | int]] = []
    for step_ahead in range(1, max_ahead + 1):
        mask = _adjacent_same_episode_mask(episode_index, step_ahead=step_ahead)
        if mask.size == 0:
            deltas = np.zeros((0,), dtype=np.float32)
        else:
            deltas = np.linalg.norm(actions[step_ahead:] - states[:-step_ahead], axis=1)[mask]
        lookahead_stats.append(
            {
                "step_ahead": step_ahead,
                "seconds_ahead": float(step_ahead / fps) if fps > 0 else 0.0,
                "mean_l2": _safe_mean(deltas),
                "median_l2": _safe_median(deltas),
                "p90_l2": _safe_quantile(deltas, 0.9),
            }
        )

    valid_window_scores: list[float] = []
    max_start = len(actions) - (n_obs_steps - 1 + n_action_steps) + 1
    for start in range(max_start):
        end = start + n_obs_steps - 1 + n_action_steps
        if not np.all(episode_index[start:end] == episode_index[start]):
            continue
        future = actions[start + n_obs_steps - 1 : start + n_obs_steps - 1 + n_action_steps]
        valid_window_scores.append(float(np.mean(np.std(future, axis=0))))
    valid_window_scores_arr = np.asarray(valid_window_scores, dtype=np.float32)

    action_min = np.asarray(stats["action"]["min"], dtype=np.float32)
    action_max = np.asarray(stats["action"]["max"], dtype=np.float32)
    baseline = _hold_current_state_baseline(
        actions=actions,
        states=states,
        episode_index=episode_index,
        action_min=action_min,
        action_max=action_max,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )

    policy_action_nonzero_ratio = None
    if "complementary_info.policy_action" in df.columns:
        policy_action = _to_numpy_matrix(df["complementary_info.policy_action"])
        policy_action_nonzero_ratio = float(np.mean(np.abs(policy_action) > 1e-8)) if policy_action.size else 0.0

    intervention_ratio = None
    if "complementary_info.is_intervention" in df.columns:
        intervention = np.asarray(df["complementary_info.is_intervention"].to_list(), dtype=np.float32).reshape(-1)
        intervention_ratio = float(np.mean(intervention > 0.0)) if intervention.size else 0.0

    per_joint_report: list[dict[str, float | str]] = []
    mean_abs_delta = np.mean(np.abs(delta_actions_same_ep), axis=0) if delta_actions_same_ep.size else np.zeros(actions.shape[1])
    mean_abs_action_state = np.mean(action_state_diff, axis=0) if action_state_diff.size else np.zeros(actions.shape[1])
    action_std = np.std(actions, axis=0) if actions.size else np.zeros((0,), dtype=np.float32)
    stationary_ratio = (
        np.mean(np.abs(delta_actions_same_ep) < 1e-6, axis=0) if delta_actions_same_ep.size else np.zeros(actions.shape[1])
    )
    for idx in range(actions.shape[1]):
        per_joint_report.append(
            {
                "joint": action_names[idx] if idx < len(action_names) else f"joint_{idx}",
                "action_std": float(action_std[idx]),
                "mean_abs_delta": float(mean_abs_delta[idx]),
                "mean_abs_action_state_gap": float(mean_abs_action_state[idx]),
                "adjacent_zero_delta_ratio": float(stationary_ratio[idx]),
            }
        )

    episode_summaries: list[dict[str, float | int]] = []
    for ep in sorted(episode_lengths):
        ep_mask = episode_index == ep
        ep_actions = actions[ep_mask]
        ep_states = states[ep_mask]
        unique_actions = np.unique(ep_actions, axis=0).shape[0] if len(ep_actions) else 0
        ep_delta = ep_actions[1:] - ep_actions[:-1] if len(ep_actions) > 1 else np.zeros((0, actions.shape[1]), dtype=np.float32)
        ep_delta_norm = np.linalg.norm(ep_delta, axis=1) if ep_delta.size else np.zeros((0,), dtype=np.float32)
        episode_summaries.append(
            {
                "episode_index": ep,
                "length_frames": int(ep_actions.shape[0]),
                "length_seconds": float(ep_actions.shape[0] / fps) if fps > 0 else 0.0,
                "unique_action_count": int(unique_actions),
                "unique_action_ratio": _safe_ratio(unique_actions, max(1, ep_actions.shape[0])),
                "mean_adjacent_action_l2": _safe_mean(ep_delta_norm),
                "median_adjacent_action_l2": _safe_median(ep_delta_norm),
                "mean_action_state_l2": float(np.mean(np.linalg.norm(ep_actions - ep_states, axis=1))) if len(ep_actions) else 0.0,
            }
        )

    episode_summaries_sorted = sorted(episode_summaries, key=lambda item: item["mean_adjacent_action_l2"])

    return {
        "dataset_root": str(dataset_root),
        "meta": {
            "robot_type": info.get("robot_type"),
            "fps": fps,
            "total_frames": int(len(df)),
            "total_episodes": int(len(episode_lengths)),
            "action_dim": int(actions.shape[1]) if actions.ndim == 2 else 0,
            "action_names": action_names,
            "n_obs_steps": n_obs_steps,
            "n_action_steps": n_action_steps,
            "max_ahead": max_ahead,
        },
        "summary": {
            "same_action_as_previous_frame_ratio": float(np.mean(same_prev_same_ep)) if same_prev_same_ep.size else 0.0,
            "adjacent_action_l2_mean": _safe_mean(delta_norms),
            "adjacent_action_l2_median": _safe_median(delta_norms),
            "adjacent_action_l2_p90": _safe_quantile(delta_norms, 0.9),
            "action_state_l2_mean": _safe_mean(action_state_l2),
            "action_state_l2_median": _safe_median(action_state_l2),
            "action_state_abs_gap_mean": _safe_mean(action_state_diff),
            "action_state_abs_gap_std": _safe_std(action_state_diff),
            "windows_mean_joint_std_mean": _safe_mean(valid_window_scores_arr),
            "windows_mean_joint_std_median": _safe_median(valid_window_scores_arr),
            "windows_mean_joint_std_p10": _safe_quantile(valid_window_scores_arr, 0.1),
            "windows_mean_joint_std_p90": _safe_quantile(valid_window_scores_arr, 0.9),
            "window_ratio_mean_joint_std_lt_0_2": float(np.mean(valid_window_scores_arr < 0.2)) if valid_window_scores_arr.size else 0.0,
            "window_ratio_mean_joint_std_lt_0_5": float(np.mean(valid_window_scores_arr < 0.5)) if valid_window_scores_arr.size else 0.0,
            "window_ratio_mean_joint_std_lt_1_0": float(np.mean(valid_window_scores_arr < 1.0)) if valid_window_scores_arr.size else 0.0,
            "policy_action_nonzero_ratio": policy_action_nonzero_ratio,
            "intervention_ratio": intervention_ratio,
        },
        "hold_current_state_baseline": baseline,
        "lookahead_state_to_future_action_l2": lookahead_stats,
        "per_joint": per_joint_report,
        "episodes": {
            "most_static": episode_summaries_sorted[:10],
            "most_dynamic": list(reversed(episode_summaries_sorted[-10:])),
        },
    }


def format_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = []

    meta = report["meta"]
    summary = report["summary"]
    baseline = report["hold_current_state_baseline"]

    lines.append("=== Action Trend Report ===")
    lines.append(f"Root: {report['dataset_root']}")
    lines.append("")

    lines.append("[Meta]")
    lines.append(f"- robot_type: {meta['robot_type']}")
    lines.append(f"- fps: {meta['fps']}")
    lines.append(f"- total_frames: {meta['total_frames']}")
    lines.append(f"- total_episodes: {meta['total_episodes']}")
    lines.append(f"- action_dim: {meta['action_dim']}")
    lines.append(
        f"- report horizon: n_obs_steps={meta['n_obs_steps']}, n_action_steps={meta['n_action_steps']}, max_ahead={meta['max_ahead']}"
    )
    lines.append("")

    lines.append("[Global Trend Summary]")
    lines.append(f"- same_action_as_previous_frame_ratio: {summary['same_action_as_previous_frame_ratio']:.6f}")
    lines.append(
        f"- adjacent_action_l2: mean={summary['adjacent_action_l2_mean']:.6f}, "
        f"median={summary['adjacent_action_l2_median']:.6f}, p90={summary['adjacent_action_l2_p90']:.6f}"
    )
    lines.append(
        f"- action_state_l2: mean={summary['action_state_l2_mean']:.6f}, "
        f"median={summary['action_state_l2_median']:.6f}"
    )
    lines.append(
        f"- action_state_abs_gap: mean={summary['action_state_abs_gap_mean']:.6f}, "
        f"std={summary['action_state_abs_gap_std']:.6f}"
    )
    lines.append(
        f"- future_window_mean_joint_std: mean={summary['windows_mean_joint_std_mean']:.6f}, "
        f"median={summary['windows_mean_joint_std_median']:.6f}, "
        f"p10={summary['windows_mean_joint_std_p10']:.6f}, p90={summary['windows_mean_joint_std_p90']:.6f}"
    )
    lines.append(
        f"- low-trend window ratios: <0.2={summary['window_ratio_mean_joint_std_lt_0_2']:.6f}, "
        f"<0.5={summary['window_ratio_mean_joint_std_lt_0_5']:.6f}, "
        f"<1.0={summary['window_ratio_mean_joint_std_lt_1_0']:.6f}"
    )
    if summary["policy_action_nonzero_ratio"] is not None:
        lines.append(f"- policy_action_nonzero_ratio: {summary['policy_action_nonzero_ratio']:.6f}")
    if summary["intervention_ratio"] is not None:
        lines.append(f"- intervention_ratio: {summary['intervention_ratio']:.6f}")
    lines.append("")

    lines.append("[Hold-Current-State Baseline]")
    lines.append(
        f"- raw_l1: mean={baseline['raw_l1_mean']:.6f}, median={baseline['raw_l1_median']:.6f}"
    )
    lines.append(
        f"- normalized_l1: mean={baseline['norm_l1_mean']:.6f}, median={baseline['norm_l1_median']:.6f}"
    )
    lines.append("")

    lines.append("[Lookahead: state(t) -> action(t+k)]")
    for item in report["lookahead_state_to_future_action_l2"]:
        lines.append(
            f"- k={item['step_ahead']:>2} ({item['seconds_ahead']:.3f}s): "
            f"mean={item['mean_l2']:.6f}, median={item['median_l2']:.6f}, p90={item['p90_l2']:.6f}"
        )
    lines.append("")

    lines.append("[Per-Joint]")
    for item in report["per_joint"]:
        lines.append(
            f"- {item['joint']}: action_std={item['action_std']:.6f}, mean_abs_delta={item['mean_abs_delta']:.6f}, "
            f"mean_abs_action_state_gap={item['mean_abs_action_state_gap']:.6f}, "
            f"adjacent_zero_delta_ratio={item['adjacent_zero_delta_ratio']:.6f}"
        )
    lines.append("")

    lines.append("[Most Static Episodes]")
    for item in report["episodes"]["most_static"]:
        lines.append(
            f"- ep={item['episode_index']}: frames={item['length_frames']}, seconds={item['length_seconds']:.3f}, "
            f"unique_action_ratio={item['unique_action_ratio']:.6f}, "
            f"mean_adjacent_action_l2={item['mean_adjacent_action_l2']:.6f}, "
            f"mean_action_state_l2={item['mean_action_state_l2']:.6f}"
        )
    lines.append("")

    lines.append("[Most Dynamic Episodes]")
    for item in report["episodes"]["most_dynamic"]:
        lines.append(
            f"- ep={item['episode_index']}: frames={item['length_frames']}, seconds={item['length_seconds']:.3f}, "
            f"unique_action_ratio={item['unique_action_ratio']:.6f}, "
            f"mean_adjacent_action_l2={item['mean_adjacent_action_l2']:.6f}, "
            f"mean_action_state_l2={item['mean_action_state_l2']:.6f}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a detailed action-trend report for a LeRobot dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset repo id or local filesystem path.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional root directory containing datasets. Defaults to HF_LEROBOT_HOME behavior.",
    )
    parser.add_argument("--n-obs-steps", type=int, default=8, help="Observation history used for baseline windowing.")
    parser.add_argument("--n-action-steps", type=int, default=8, help="Future action chunk length for baseline windowing.")
    parser.add_argument("--max-ahead", type=int, default=8, help="Maximum step-ahead distance for state-to-future-action analysis.")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text.")
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset, args.root)
    report = build_report(
        dataset_root=dataset_root,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        max_ahead=args.max_ahead,
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_text_report(report))


if __name__ == "__main__":
    main()
