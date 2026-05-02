#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
HF_CACHE_ROOT = REPO_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_ROOT / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_ROOT / "hub"))
os.environ.setdefault("DATASETS_CACHE", str(HF_CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ABPolicy rollout visualization on dataset episodes.")
    parser.add_argument("--checkpoint-path", type=Path, action="append", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-repo-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--episode-indices", type=int, nargs="*", default=[0, 10, 20])
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def load_cfg(checkpoint_path: Path, args: argparse.Namespace) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig.from_pretrained(checkpoint_path)
    cfg = copy.deepcopy(cfg)
    cfg.policy.device = resolve_device(args.device)
    cfg.policy.pretrained_path = checkpoint_path
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    if args.dataset_repo_id is not None:
        cfg.dataset.repo_id = args.dataset_repo_id
    return cfg


def load_policy_and_processors(cfg: TrainPipelineConfig, dataset):
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    policy.eval()
    return policy, preprocessor


def predict_ab_chunk(policy, processed_batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model_batch = policy._prepare_model_batch(processed_batch, for_training=False)

    state = model_batch[OBS_STATE]
    if state.ndim == 2:
        state = state.unsqueeze(1)

    images = model_batch["observation.images"]
    if images.ndim == 5:
        images = images.unsqueeze(1)

    ctrl_points = policy.model.generate_ctrl_points(
        {
            OBS_STATE: state,
            "observation.images": images,
        }
    )
    if policy.config.use_bspline:
        ctrl_points = policy._denormalize_action_like(ctrl_points)
        full_actions = policy.model.projector.rebuild_batch(ctrl_points)
    else:
        full_actions = policy._denormalize_action_like(ctrl_points)
    start = policy.config.action_history_horizon
    end = start + policy.config.n_action_steps
    return full_actions[:, start:end]


def batchify_processed_item(processed_item: dict) -> dict:
    batched = {}
    for key, value in processed_item.items():
        if isinstance(value, torch.Tensor):
            batched[key] = value.unsqueeze(0)
        else:
            batched[key] = value
    return batched


def select_windows(dataset, episode_indices: list[int], windows_per_episode: int, drop_n_last: int) -> list[dict]:
    episode_ids = np.asarray(dataset.hf_dataset["episode_index"])
    frame_ids = np.asarray(dataset.hf_dataset["frame_index"])
    windows = []
    for episode_index in episode_indices:
        positions = np.where(episode_ids == episode_index)[0].tolist()
        if not positions:
            continue
        valid = positions[:-drop_n_last] if drop_n_last > 0 and len(positions) > drop_n_last else positions
        if not valid:
            continue
        if len(valid) <= windows_per_episode:
            selected = valid
        else:
            picks = np.linspace(0, len(valid) - 1, windows_per_episode, dtype=int).tolist()
            selected = [valid[i] for i in picks]
        for dataset_index in selected:
            windows.append(
                {
                    "episode_index": int(episode_index),
                    "dataset_index": int(dataset_index),
                    "frame_index": int(frame_ids[dataset_index]),
                }
            )
    return windows


def ensure_chunk_2d(chunk: np.ndarray | torch.Tensor, action_dim: int, label: str) -> np.ndarray:
    if isinstance(chunk, torch.Tensor):
        arr = chunk.detach().cpu().numpy()
    else:
        arr = np.asarray(chunk)

    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.shape[0] == action_dim:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{label} must be 2D after squeeze, got shape {arr.shape}.")
    if arr.shape[-1] != action_dim:
        raise ValueError(f"{label} last dim must be {action_dim}, got shape {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def ensure_anchor_1d(anchor: np.ndarray | torch.Tensor, action_dim: int) -> np.ndarray:
    if isinstance(anchor, torch.Tensor):
        arr = anchor.detach().cpu().numpy()
    else:
        arr = np.asarray(anchor)

    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"anchor must be 1D after squeeze, got shape {arr.shape}.")
    if arr.shape[0] != action_dim:
        raise ValueError(f"anchor dim must be {action_dim}, got shape {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def tensor_to_rows(chunk: np.ndarray, action_names: list[str]) -> list[dict[str, float]]:
    rows = []
    for step_idx, action in enumerate(chunk):
        row = {"step": step_idx}
        for joint_name, value in zip(action_names, action, strict=True):
            row[joint_name] = float(value)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_window(
    output_path: Path,
    action_names: list[str],
    gt_chunk: np.ndarray,
    pred_by_ckpt: dict[str, np.ndarray],
    anchor: np.ndarray,
    title: str,
) -> None:
    num_actions = len(action_names)
    fig, axes = plt.subplots(num_actions, 1, figsize=(14, 2.2 * num_actions), sharex=True)
    if num_actions == 1:
        axes = [axes]

    x = np.arange(gt_chunk.shape[0])
    for idx, joint_name in enumerate(action_names):
        ax = axes[idx]
        ax.plot(x, gt_chunk[:, idx], label="gt", linewidth=2.0, color="black")
        for ckpt_name, pred_chunk in pred_by_ckpt.items():
            ax.plot(x, pred_chunk[:, idx], label=ckpt_name, linewidth=1.6)
        ax.axhline(anchor[idx], linestyle="--", linewidth=1.0, color="gray", alpha=0.7, label="anchor" if idx == 0 else None)
        ax.set_ylabel(joint_name)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    axes[-1].set_xlabel("chunk step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_paths = [path.expanduser().resolve() for path in args.checkpoint_path]
    cfg = load_cfg(checkpoint_paths[0], args)
    dataset = make_dataset(cfg)
    action_names = dataset.meta.features["action"]["names"]
    action_dim = len(action_names)
    drop_n_last = getattr(cfg.policy, "drop_n_last_frames", 0)
    windows = select_windows(dataset, args.episode_indices, args.windows_per_episode, drop_n_last)
    if not windows:
        raise RuntimeError("No valid episode windows were selected.")

    loaded = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_cfg = load_cfg(checkpoint_path, args)
        policy, preprocessor = load_policy_and_processors(checkpoint_cfg, dataset)
        loaded.append(
            {
                "checkpoint_path": checkpoint_path,
                "checkpoint_name": checkpoint_path.parent.name,
                "policy": policy,
                "preprocessor": preprocessor,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for window in windows:
        raw_item = dataset[window["dataset_index"]]
        gt_action = raw_item[ACTION]
        gt_chunk = gt_action[cfg.policy.action_history_horizon : cfg.policy.action_history_horizon + cfg.policy.n_action_steps]
        gt_chunk = ensure_chunk_2d(gt_chunk, action_dim, "gt_chunk")

        state = raw_item[OBS_STATE]
        anchor = state[cfg.policy.n_obs_steps - 1]
        anchor = ensure_anchor_1d(anchor, action_dim)

        episode_dir = args.output_dir / f"episode_{window['episode_index']:04d}" / f"frame_{window['frame_index']:05d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        write_csv(episode_dir / "gt_chunk.csv", tensor_to_rows(gt_chunk, action_names))
        np.save(episode_dir / "gt_chunk.npy", gt_chunk)
        np.save(episode_dir / "anchor.npy", anchor)

        pred_by_ckpt = {}
        window_record = {
            **window,
            "files": {
                "gt_csv": str(episode_dir / "gt_chunk.csv"),
                "gt_npy": str(episode_dir / "gt_chunk.npy"),
                "anchor_npy": str(episode_dir / "anchor.npy"),
            },
            "checkpoints": [],
        }

        for entry in loaded:
            processed = batchify_processed_item(entry["preprocessor"](raw_item))
            with torch.no_grad():
                pred_chunk = predict_ab_chunk(entry["policy"], processed)
            pred_chunk = ensure_chunk_2d(pred_chunk, action_dim, f"{entry['checkpoint_name']}_pred")

            pred_by_ckpt[entry["checkpoint_name"]] = pred_chunk
            pred_csv = episode_dir / f"{entry['checkpoint_name']}_pred.csv"
            pred_npy = episode_dir / f"{entry['checkpoint_name']}_pred.npy"
            write_csv(pred_csv, tensor_to_rows(pred_chunk, action_names))
            np.save(pred_npy, pred_chunk)

            window_record["checkpoints"].append(
                {
                    "checkpoint_name": entry["checkpoint_name"],
                    "checkpoint_path": str(entry["checkpoint_path"]),
                    "pred_csv": str(pred_csv),
                    "pred_npy": str(pred_npy),
                    "chunk_mae": float(np.mean(np.abs(pred_chunk - gt_chunk))),
                    "first_mae": float(np.mean(np.abs(pred_chunk[0] - gt_chunk[0]))),
                }
            )

        plot_path = episode_dir / "action_plot.png"
        plot_window(
            plot_path,
            action_names,
            gt_chunk,
            pred_by_ckpt,
            anchor,
            title=f"episode={window['episode_index']} frame={window['frame_index']}",
        )
        window_record["files"]["plot_png"] = str(plot_path)
        manifest.append(window_record)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"saved_manifest": str(manifest_path), "num_windows": len(manifest)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
