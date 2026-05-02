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
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ABPolicy checkpoints offline on the same dataset windows.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        action="append",
        required=True,
        help="Path to a checkpoint pretrained_model directory. Repeat for multiple checkpoints.",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-repo-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--episode-indices", type=int, nargs="*", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested_device


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
    keys = ("type", "n_obs_steps", "n_action_steps", "action_history_horizon", "action_horizon")
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

    if mismatches:
        raise ValueError(f"Incompatible checkpoint {checkpoint_path}: {'; '.join(mismatches)}")


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
    return loader


def collect_raw_batches(loader: DataLoader, max_batches: int) -> list[dict[str, Any]]:
    batches = []
    for idx, raw_batch in enumerate(loader):
        if idx >= max_batches:
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


def build_prediction_batch(policy, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prepared = policy._prepare_model_batch(batch, for_training=False)
    return {
        OBS_STATE: prepared[OBS_STATE],
        OBS_IMAGES: prepared[OBS_IMAGES],
    }


def predict_ab_chunk(policy, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model_batch = build_prediction_batch(policy, batch)
    ctrl_points = policy.model.generate_ctrl_points(model_batch)
    if policy.config.use_bspline:
        ctrl_points = policy._denormalize_action_like(ctrl_points)
        full_actions = policy.model.projector.rebuild_batch(ctrl_points)
    else:
        full_actions = policy._denormalize_action_like(ctrl_points)
    start = policy.config.action_history_horizon
    end = start + policy.config.n_action_steps
    return full_actions[:, start:end]


def l2_mean(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=-1).mean()


def evaluate_checkpoint(checkpoint_path: Path, cfg: TrainPipelineConfig, dataset: Any, raw_batches: list[dict[str, Any]]) -> dict[str, Any]:
    policy, preprocessor = load_policy_and_preprocessor(cfg, dataset)
    policy.eval()

    totals = {
        "loss": 0.0,
        "chunk_mae_raw": 0.0,
        "first_mae_raw": 0.0,
        "pred_adjacent_l2_raw": 0.0,
        "target_adjacent_l2_raw": 0.0,
        "pred_first_delta_raw": 0.0,
        "target_first_delta_raw": 0.0,
    }
    total_samples = 0

    with torch.no_grad():
        for raw_batch in raw_batches:
            batch = preprocessor(raw_batch)
            loss, _ = policy.forward(batch)

            pred_actions = predict_ab_chunk(policy, batch)
            if pred_actions.device != batch[ACTION].device:
                pred_actions = pred_actions.to(batch[ACTION].device)

            start = policy.config.action_history_horizon
            end = start + policy.config.n_action_steps
            target_actions = batch[ACTION][:, start:end, :]

            anchor = batch[OBS_STATE][:, policy.config.n_obs_steps - 1, :].unsqueeze(1)
            pred_delta = (pred_actions[:, :1] - anchor).abs()
            target_delta = (target_actions[:, :1] - anchor).abs()

            batch_size = target_actions.shape[0]
            totals["loss"] += float(loss.item()) * batch_size
            totals["chunk_mae_raw"] += float((pred_actions - target_actions).abs().mean().item()) * batch_size
            totals["first_mae_raw"] += float((pred_actions[:, 0] - target_actions[:, 0]).abs().mean().item()) * batch_size
            totals["pred_first_delta_raw"] += float(pred_delta.mean().item()) * batch_size
            totals["target_first_delta_raw"] += float(target_delta.mean().item()) * batch_size

            if pred_actions.shape[1] > 1:
                totals["pred_adjacent_l2_raw"] += float(l2_mean(pred_actions[:, 1:] - pred_actions[:, :-1]).item()) * batch_size
                totals["target_adjacent_l2_raw"] += float(l2_mean(target_actions[:, 1:] - target_actions[:, :-1]).item()) * batch_size

            total_samples += batch_size

    means = {key: value / total_samples for key, value in totals.items()}
    means["pred_target_adjacent_ratio"] = (
        means["pred_adjacent_l2_raw"] / means["target_adjacent_l2_raw"] if means["target_adjacent_l2_raw"] != 0 else None
    )
    means["pred_target_first_delta_ratio"] = (
        means["pred_first_delta_raw"] / means["target_first_delta_raw"] if means["target_first_delta_raw"] != 0 else None
    )

    return {
        "checkpoint_path": str(checkpoint_path),
        "num_samples": total_samples,
        **means,
    }


def print_summary(results: dict[str, Any]) -> None:
    print("=== ABPolicy Checkpoint Comparison ===")
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
            f"  pred_adjacent_l2_raw={row['pred_adjacent_l2_raw']:.6f} "
            f"target_adjacent_l2_raw={row['target_adjacent_l2_raw']:.6f} "
            f"pred/target_adjacent={row['pred_target_adjacent_ratio']:.6f}"
        )
        print(
            f"  pred_first_delta_raw={row['pred_first_delta_raw']:.6f} "
            f"target_first_delta_raw={row['target_first_delta_raw']:.6f} "
            f"pred/target_first_delta={row['pred_target_first_delta_ratio']:.6f}"
        )
        print()


def main() -> None:
    args = parse_args()
    checkpoint_paths = [path.resolve() for path in args.checkpoint_path]
    reference_cfg = load_cfg(checkpoint_paths[0], args)
    if reference_cfg.policy.type not in {"abpolicy", "cage"}:
        raise ValueError(f"Expected abpolicy/cage policy, got {reference_cfg.policy.type}")

    for checkpoint_path in checkpoint_paths[1:]:
        validate_compatible(reference_cfg, load_cfg(checkpoint_path, args), checkpoint_path)

    dataset = make_dataset(reference_cfg)
    drop_n_last = getattr(reference_cfg.policy, "drop_n_last_frames", 0)
    loader = make_loader(
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
        rows.append(evaluate_checkpoint(checkpoint_path, cfg, dataset, raw_batches))

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
