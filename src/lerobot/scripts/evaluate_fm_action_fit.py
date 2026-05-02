#!/usr/bin/env python

import os
from pathlib import Path

_CACHE_ROOT = Path(__file__).resolve().parent / ".cache" / "hf_eval"
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_ROOT / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_CACHE_ROOT / "hub"))

import argparse
import json
import random

import numpy as np
import torch

import lerobot.policies.fm.configuration_fm  # noqa: F401
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Flow Matching action fitting, with emphasis on episode starts.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to pretrained_model directory.")
    parser.add_argument("--output", type=Path, default=None, help="Where to save the JSON report.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference.",
    )
    parser.add_argument("--max-start-windows", type=int, default=None)
    parser.add_argument("--max-early-windows", type=int, default=None)
    parser.add_argument("--max-overall-windows", type=int, default=256)
    parser.add_argument("--early-frame-threshold", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available.")
    return device_arg


def to_batched_policy_input(raw: dict, preprocessor, image_features: list[str]) -> dict[str, torch.Tensor]:
    proc = preprocessor(raw)
    batch = {OBS_STATE: proc[OBS_STATE].unsqueeze(0)}
    if image_features:
        batch[OBS_IMAGES] = torch.stack([proc[key].unsqueeze(0) for key in image_features], dim=-4)
    return batch


def generate_actions(policy, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(policy, "flow_matching"):
        return policy.flow_matching.generate_actions(batch)
    raise TypeError(f"Unsupported policy object for FM evaluation: {type(policy).__name__}")


def summarize_split(
    name: str,
    indices: list[int],
    dataset,
    policy,
    preprocessor,
    postprocessor,
    future_start: int,
    future_end: int,
    n_obs_steps: int,
    image_features: list[str],
    action_names: list[str],
) -> dict:
    chunk_maes = []
    first_maes = []
    first_pred = []
    first_gt = []
    first_pred_delta = []
    first_gt_delta = []

    for idx in indices:
        raw = dataset[idx]
        batch = to_batched_policy_input(raw, preprocessor, image_features)
        pred = generate_actions(policy, batch)
        pred = postprocessor(pred)[0].detach().cpu()
        gt = raw["action"][future_start:future_end].detach().cpu()
        anchor = raw["observation.state"][n_obs_steps - 1].detach().cpu()

        chunk_maes.append(torch.mean(torch.abs(pred - gt)).item())
        first_maes.append(torch.mean(torch.abs(pred[0] - gt[0])).item())
        first_pred.append(pred[0].numpy())
        first_gt.append(gt[0].numpy())
        first_pred_delta.append((pred[0] - anchor).numpy())
        first_gt_delta.append((gt[0] - anchor).numpy())

    first_pred = np.stack(first_pred)
    first_gt = np.stack(first_gt)
    first_pred_delta = np.stack(first_pred_delta)
    first_gt_delta = np.stack(first_gt_delta)

    pred_delta_abs = np.abs(first_pred_delta).mean(axis=0)
    gt_delta_abs = np.abs(first_gt_delta).mean(axis=0)
    amp_ratio = pred_delta_abs / np.maximum(gt_delta_abs, 1e-8)
    sign_match = (np.sign(first_pred_delta) == np.sign(first_gt_delta)).mean(axis=0)
    action_mae_per_joint = np.mean(np.abs(first_pred - first_gt), axis=0)
    delta_mae_per_joint = np.mean(np.abs(first_pred_delta - first_gt_delta), axis=0)

    def keyed(values):
        return {joint_name: float(value) for joint_name, value in zip(action_names, values)}

    return {
        "split": name,
        "count": len(indices),
        "chunk_mae_mean": float(np.mean(chunk_maes)),
        "chunk_mae_std": float(np.std(chunk_maes)),
        "first_action_mae_mean": float(np.mean(first_maes)),
        "first_action_mae_std": float(np.std(first_maes)),
        "pred_first_action_mean": keyed(first_pred.mean(axis=0)),
        "gt_first_action_mean": keyed(first_gt.mean(axis=0)),
        "pred_first_action_std": keyed(first_pred.std(axis=0)),
        "gt_first_action_std": keyed(first_gt.std(axis=0)),
        "pred_first_delta_abs_mean": keyed(pred_delta_abs),
        "gt_first_delta_abs_mean": keyed(gt_delta_abs),
        "delta_abs_amplitude_ratio_pred_over_gt": keyed(amp_ratio),
        "delta_sign_match_rate": keyed(sign_match),
        "first_action_mae_per_joint": keyed(action_mae_per_joint),
        "first_delta_mae_per_joint": keyed(delta_mae_per_joint),
    }


def sample_indices(indices: list[int], max_count: int | None, rng: random.Random) -> list[int]:
    if max_count is None or len(indices) <= max_count:
        return indices
    return sorted(rng.sample(indices, max_count))


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    device = resolve_device(args.device)
    output_path = args.output or checkpoint.parent.parent / "fm_action_fit_report.json"

    cfg = TrainPipelineConfig.from_pretrained(checkpoint)
    cfg.policy.device = device
    cfg.dataset.video_backend = "pyav"

    dataset = make_dataset(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg.policy, dataset_stats=dataset.meta.stats)
    policy_cls = get_policy_class(cfg.policy.type)
    policy = policy_cls.from_pretrained(checkpoint, config=cfg.policy)
    policy.eval()

    frame_index = np.asarray(dataset.hf_dataset["frame_index"])
    rng = random.Random(args.seed)
    start_indices = np.where(frame_index == 0)[0].tolist()
    early_indices = np.where(frame_index < args.early_frame_threshold)[0].tolist()
    overall_indices = list(range(len(dataset)))

    start_indices = sample_indices(start_indices, args.max_start_windows, rng)
    early_indices = sample_indices(early_indices, args.max_early_windows, rng)
    overall_indices = sample_indices(overall_indices, args.max_overall_windows, rng)

    future_start = cfg.policy.n_obs_steps - 1
    future_end = future_start + cfg.policy.n_action_steps
    action_names = dataset.meta.features["action"]["names"]
    image_features = list(cfg.policy.image_features.keys())

    report = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(cfg.dataset.root),
        "device": device,
        "policy_type": cfg.policy.type,
        "n_obs_steps": cfg.policy.n_obs_steps,
        "n_action_steps": cfg.policy.n_action_steps,
        "horizon": cfg.policy.horizon,
        "future_start_index": future_start,
        "evaluated_windows": {
            "start": len(start_indices),
            "early": len(early_indices),
            "overall_sampled": len(overall_indices),
        },
        "start_windows": summarize_split(
            "start",
            start_indices,
            dataset,
            policy,
            preprocessor,
            postprocessor,
            future_start,
            future_end,
            cfg.policy.n_obs_steps,
            image_features,
            action_names,
        ),
        "early_windows": summarize_split(
            "early",
            early_indices,
            dataset,
            policy,
            preprocessor,
            postprocessor,
            future_start,
            future_end,
            cfg.policy.n_obs_steps,
            image_features,
            action_names,
        ),
        "overall_sampled_windows": summarize_split(
            "overall_sampled",
            overall_indices,
            dataset,
            policy,
            preprocessor,
            postprocessor,
            future_start,
            future_end,
            cfg.policy.n_obs_steps,
            image_features,
            action_names,
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    summary = {
        "saved_to": str(output_path),
        "start_chunk_mae_mean": report["start_windows"]["chunk_mae_mean"],
        "start_first_action_mae_mean": report["start_windows"]["first_action_mae_mean"],
        "early_first_action_mae_mean": report["early_windows"]["first_action_mae_mean"],
        "overall_first_action_mae_mean": report["overall_sampled_windows"]["first_action_mae_mean"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
