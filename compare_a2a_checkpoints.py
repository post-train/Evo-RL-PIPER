#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
HF_CACHE_ROOT = REPO_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_ROOT / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_ROOT / "hub"))
os.environ.setdefault("DATASETS_CACHE", str(HF_CACHE_ROOT / "datasets"))

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple A2A checkpoints offline on the same dataset samples. "
            "Focuses on action amplitude / conservatism rather than only fit loss."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        action="append",
        required=True,
        help="Path to a checkpoint pretrained_model directory. Repeat this flag for multiple checkpoints.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override dataset root from train_config.json.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="Override dataset repo_id from train_config.json.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=20,
        help="Number of dataloader batches to compare. Use a larger value for a more stable estimate.",
    )
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit episode subset.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def ensure_local_hf_cache(repo_root: Path) -> None:
    cache_root = repo_root / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def denormalize_action(action: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    action_min = torch.as_tensor(stats["min"], dtype=torch.float32, device=action.device)
    action_max = torch.as_tensor(stats["max"], dtype=torch.float32, device=action.device)
    denom = torch.where((action_max - action_min) == 0, torch.full_like(action_max, 1e-8), action_max - action_min)
    return (action + 1.0) / 2.0 * denom + action_min


def l2_mean(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=-1).mean()


def load_cfg(checkpoint_path: Path, args: argparse.Namespace) -> TrainPipelineConfig:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    cfg = TrainPipelineConfig.from_pretrained(checkpoint_path)
    cfg = copy.deepcopy(cfg)
    cfg.policy.device = resolve_device(args.device)
    cfg.policy.pretrained_path = checkpoint_path
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    if args.dataset_repo_id is not None:
        cfg.dataset.repo_id = args.dataset_repo_id
    return cfg


def validate_compatible(reference_cfg: TrainPipelineConfig, other_cfg: TrainPipelineConfig, checkpoint_path: Path) -> None:
    keys = ("type", "n_obs_steps", "horizon", "n_action_steps")
    mismatches = []
    for key in keys:
        ref_value = getattr(reference_cfg.policy, key)
        other_value = getattr(other_cfg.policy, key)
        if ref_value != other_value:
            mismatches.append(f"{key}: ref={ref_value}, other={other_value}")

    ref_inputs = reference_cfg.policy.input_features
    other_inputs = other_cfg.policy.input_features
    if set(ref_inputs.keys()) != set(other_inputs.keys()):
        mismatches.append(
            f"input_features keys differ: ref={sorted(ref_inputs.keys())}, other={sorted(other_inputs.keys())}"
        )
    else:
        for key in sorted(ref_inputs.keys()):
            ref_shape = tuple(ref_inputs[key].shape)
            other_shape = tuple(other_inputs[key].shape)
            if ref_shape != other_shape:
                mismatches.append(f"{key}.shape: ref={ref_shape}, other={other_shape}")

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(f"Incompatible checkpoint {checkpoint_path}: {joined}")


def make_loader(dataset: Any, batch_size: int, num_workers: int, episode_indices: list[int] | None, drop_n_last: int):
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=episode_indices,
        drop_n_last_frames=drop_n_last,
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader, len(sampler)


def collect_raw_batches(loader: DataLoader, max_batches: int) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batches.append(raw_batch)
    if not batches:
        raise RuntimeError("No batches were collected for comparison.")
    return batches


def load_policy_and_preprocessor(cfg: TrainPipelineConfig, dataset: Any):
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
    return policy, preprocessor


def evaluate_checkpoint(
    checkpoint_path: Path,
    cfg: TrainPipelineConfig,
    dataset: Any,
    raw_batches: list[dict[str, Any]],
) -> dict[str, Any]:
    policy, preprocessor = load_policy_and_preprocessor(cfg, dataset)
    policy.eval()

    future_start = policy.config.n_obs_steps - 1
    future_end = future_start + policy.config.n_action_steps
    action_stats = dataset.meta.stats[ACTION]
    image_keys = list(policy.config.image_features.keys())

    totals = {
        "loss": 0.0,
        "chunk_mae_raw": 0.0,
        "first_mae_raw": 0.0,
        "pred_gap_raw": 0.0,
        "target_gap_raw": 0.0,
        "pred_first_gap_raw": 0.0,
        "target_first_gap_raw": 0.0,
        "pred_adjacent_l2_raw": 0.0,
        "target_adjacent_l2_raw": 0.0,
    }
    total_samples = 0

    with torch.no_grad():
        for raw_batch in raw_batches:
            batch = preprocessor(raw_batch)
            loss, _ = policy.forward(batch)

            model_batch = {OBS_STATE: batch[OBS_STATE]}
            if image_keys:
                model_batch[OBS_IMAGES] = torch.stack([batch[key] for key in image_keys], dim=-4)

            pred_actions = policy.model.generate_actions(model_batch)
            target_actions = batch[ACTION][:, future_start:future_end, :]

            anchor = batch[OBS_STATE][:, future_start, :]
            pred_actions_raw = denormalize_action(pred_actions, action_stats)
            target_actions_raw = denormalize_action(target_actions, action_stats)
            anchor_raw = denormalize_action(anchor, action_stats)
            anchor_raw = anchor_raw.unsqueeze(1)

            pred_gap_raw = (pred_actions_raw - anchor_raw).abs()
            target_gap_raw = (target_actions_raw - anchor_raw).abs()

            batch_size = target_actions.shape[0]
            totals["loss"] += float(loss.item()) * batch_size
            totals["chunk_mae_raw"] += float((pred_actions_raw - target_actions_raw).abs().mean().item()) * batch_size
            totals["first_mae_raw"] += float((pred_actions_raw[:, 0] - target_actions_raw[:, 0]).abs().mean().item()) * batch_size
            totals["pred_gap_raw"] += float(pred_gap_raw.mean().item()) * batch_size
            totals["target_gap_raw"] += float(target_gap_raw.mean().item()) * batch_size
            totals["pred_first_gap_raw"] += float(pred_gap_raw[:, 0].mean().item()) * batch_size
            totals["target_first_gap_raw"] += float(target_gap_raw[:, 0].mean().item()) * batch_size

            if pred_actions_raw.shape[1] > 1:
                totals["pred_adjacent_l2_raw"] += float(l2_mean(pred_actions_raw[:, 1:] - pred_actions_raw[:, :-1]).item()) * batch_size
                totals["target_adjacent_l2_raw"] += float(l2_mean(target_actions_raw[:, 1:] - target_actions_raw[:, :-1]).item()) * batch_size

            total_samples += batch_size

    means = {key: value / total_samples for key, value in totals.items()}
    pred_target_gap_ratio = means["pred_gap_raw"] / means["target_gap_raw"] if means["target_gap_raw"] != 0 else None
    pred_target_first_gap_ratio = (
        means["pred_first_gap_raw"] / means["target_first_gap_raw"] if means["target_first_gap_raw"] != 0 else None
    )
    pred_target_adjacent_ratio = (
        means["pred_adjacent_l2_raw"] / means["target_adjacent_l2_raw"] if means["target_adjacent_l2_raw"] != 0 else None
    )

    return {
        "checkpoint_path": str(checkpoint_path),
        "num_samples": total_samples,
        **means,
        "pred_target_gap_ratio": pred_target_gap_ratio,
        "pred_target_first_gap_ratio": pred_target_first_gap_ratio,
        "pred_target_adjacent_ratio": pred_target_adjacent_ratio,
    }


def print_summary(results: dict[str, Any]) -> None:
    print("=== A2A Checkpoint Comparison ===")
    print(f"dataset_repo_id: {results['dataset']['repo_id']}")
    print(f"dataset_root: {results['dataset']['root']}")
    print(f"device: {results['device']}")
    print(f"policy_type: {results['policy_type']}")
    print(f"samples: {results['num_samples']}")
    print()
    for row in results["checkpoints"]:
        print(row["checkpoint_path"])
        print(
            f"  loss={row['loss']:.6f} "
            f"chunk_mae_raw={row['chunk_mae_raw']:.6f} "
            f"first_mae_raw={row['first_mae_raw']:.6f}"
        )
        print(
            f"  pred_gap_raw={row['pred_gap_raw']:.6f} "
            f"target_gap_raw={row['target_gap_raw']:.6f} "
            f"pred/target={row['pred_target_gap_ratio']:.6f}"
        )
        print(
            f"  pred_first_gap_raw={row['pred_first_gap_raw']:.6f} "
            f"target_first_gap_raw={row['target_first_gap_raw']:.6f} "
            f"pred/target_first={row['pred_target_first_gap_ratio']:.6f}"
        )
        print(
            f"  pred_adjacent_l2_raw={row['pred_adjacent_l2_raw']:.6f} "
            f"target_adjacent_l2_raw={row['target_adjacent_l2_raw']:.6f} "
            f"pred/target_adjacent={row['pred_target_adjacent_ratio']:.6f}"
        )
        print()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    ensure_local_hf_cache(repo_root)

    checkpoint_paths = [path.resolve() for path in args.checkpoint_path]
    reference_cfg = load_cfg(checkpoint_paths[0], args)
    if reference_cfg.policy.type != "a2a":
        raise ValueError(f"Expected A2A policy, got {reference_cfg.policy.type}")

    for checkpoint_path in checkpoint_paths[1:]:
        other_cfg = load_cfg(checkpoint_path, args)
        validate_compatible(reference_cfg, other_cfg, checkpoint_path)

    dataset = make_dataset(reference_cfg)
    drop_n_last = getattr(reference_cfg.policy, "drop_n_last_frames", 0)
    loader, _ = make_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        episode_indices=args.episode_indices,
        drop_n_last=drop_n_last,
    )
    raw_batches = collect_raw_batches(loader, args.max_batches)

    rows = []
    for checkpoint_path in checkpoint_paths:
        cfg = load_cfg(checkpoint_path, args)
        row = evaluate_checkpoint(checkpoint_path, cfg, dataset, raw_batches)
        rows.append(row)

    results = {
        "device": reference_cfg.policy.device,
        "policy_type": reference_cfg.policy.type,
        "dataset": {
            "repo_id": reference_cfg.dataset.repo_id,
            "root": str(reference_cfg.dataset.root),
            "num_episodes": dataset.num_episodes,
            "num_frames": dataset.num_frames,
        },
        "num_samples": rows[0]["num_samples"],
        "max_batches": args.max_batches,
        "checkpoints": rows,
    }

    print_summary(results)
    print(json.dumps(results, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"saved_json: {args.output_json}")


if __name__ == "__main__":
    main()
