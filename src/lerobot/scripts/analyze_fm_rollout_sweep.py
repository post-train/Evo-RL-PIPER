#!/usr/bin/env python

import os
from dataclasses import asdict, dataclass
from pathlib import Path

_CACHE_ROOT = Path(__file__).resolve().parent / ".cache" / "hf_rollout"
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_ROOT / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_CACHE_ROOT / "hub"))

import argparse
import json
import math
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

import lerobot.policies.fm.configuration_fm  # noqa: F401
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@dataclass
class VariantSpec:
    name: str
    num_inference_steps: int
    solver_type: str
    clip_sample: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline FM rollout sweep on the same observation segment to compare policy_action jitter."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to pretrained_model directory.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to replay.")
    parser.add_argument("--start-frame", type=int, default=0, help="Episode-local frame index to start from.")
    parser.add_argument("--length", type=int, default=96, help="How many consecutive timesteps to replay.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--steps-list",
        type=str,
        default="5,10,20",
        help="Comma-separated num_inference_steps values to compare.",
    )
    parser.add_argument(
        "--solver-list",
        type=str,
        default="euler,dopri5",
        help="Comma-separated solver_type values to compare.",
    )
    parser.add_argument(
        "--clip-list",
        type=str,
        default="true,false",
        help="Comma-separated clip_sample values to compare.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed used to generate the fixed chunk noises shared by all variants.",
    )
    parser.add_argument(
        "--joint-indices",
        type=str,
        default="0,1,2",
        help="Comma-separated joint indices to plot raw policy_action traces for.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to checkpoint.parent.parent / analysis / rollout_sweep.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available.")
    return device_arg


def parse_bool_list(raw: str) -> list[bool]:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    values = []
    for item in raw.split(","):
        key = item.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported boolean value: {item}")
        values.append(mapping[key])
    return values


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_variants(cfg_policy, args: argparse.Namespace) -> list[VariantSpec]:
    baseline = VariantSpec(
        name="baseline",
        num_inference_steps=cfg_policy.num_inference_steps,
        solver_type=cfg_policy.solver_type,
        clip_sample=cfg_policy.clip_sample,
    )
    variants: list[VariantSpec] = [baseline]

    for steps in parse_int_list(args.steps_list):
        name = f"steps_{steps}"
        if steps != baseline.num_inference_steps:
            variants.append(
                VariantSpec(
                    name=name,
                    num_inference_steps=steps,
                    solver_type=baseline.solver_type,
                    clip_sample=baseline.clip_sample,
                )
            )

    for solver in parse_str_list(args.solver_list):
        name = f"solver_{solver}"
        if solver != baseline.solver_type:
            variants.append(
                VariantSpec(
                    name=name,
                    num_inference_steps=baseline.num_inference_steps,
                    solver_type=solver,
                    clip_sample=baseline.clip_sample,
                )
            )

    for clip in parse_bool_list(args.clip_list):
        name = f"clip_{str(clip).lower()}"
        if clip != baseline.clip_sample:
            variants.append(
                VariantSpec(
                    name=name,
                    num_inference_steps=baseline.num_inference_steps,
                    solver_type=baseline.solver_type,
                    clip_sample=clip,
                )
            )

    seen = set()
    deduped = []
    for variant in variants:
        key = (variant.num_inference_steps, variant.solver_type, variant.clip_sample)
        if key not in seen:
            seen.add(key)
            deduped.append(variant)
    return deduped


def to_batched_policy_input(raw: dict, preprocessor, image_features: list[str]) -> dict[str, torch.Tensor]:
    proc = preprocessor(raw)
    batch = {OBS_STATE: proc[OBS_STATE].unsqueeze(0)}
    if image_features:
        batch[OBS_IMAGES] = torch.stack([proc[key].unsqueeze(0) for key in image_features], dim=-4)
    return batch


def find_segment_indices(dataset, episode_index: int, start_frame: int, length: int) -> list[int]:
    episode_arr = np.asarray(dataset.hf_dataset["episode_index"])
    frame_arr = np.asarray(dataset.hf_dataset["frame_index"])
    mask = (episode_arr == episode_index) & (frame_arr >= start_frame)
    indices = np.flatnonzero(mask).tolist()
    if len(indices) < length:
        raise ValueError(
            f"Requested episode={episode_index}, start_frame={start_frame}, length={length}, "
            f"but only found {len(indices)} matching windows."
        )
    segment = indices[:length]
    expected = np.arange(start_frame, start_frame + length)
    actual = frame_arr[segment]
    if not np.array_equal(actual, expected):
        raise ValueError(
            "Segment is not contiguous in frame_index. "
            f"Expected {expected[:5]}..., got {actual[:5]}..."
        )
    return segment


def make_fixed_noises(
    num_chunks: int,
    horizon: int,
    action_dim: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return [
        torch.randn((1, horizon, action_dim), generator=generator, device=device, dtype=dtype)
        for _ in range(num_chunks)
    ]


def replay_variant(
    policy,
    dataset,
    segment_indices: list[int],
    preprocessor,
    postprocessor,
    image_features: list[str],
    noises: list[torch.Tensor],
) -> np.ndarray:
    policy.eval()
    policy.reset()

    queued_actions: list[np.ndarray] = []
    chunk_idx = 0
    outputs: list[np.ndarray] = []

    for dataset_idx in segment_indices:
        raw = dataset[dataset_idx]
        batch = to_batched_policy_input(raw, preprocessor, image_features)

        if not queued_actions:
            action_chunk = policy.flow_matching.generate_actions(
                batch,
                noise=noises[chunk_idx],
                rtc_processor=None,
            )
            action_chunk = postprocessor(action_chunk)[0].detach().cpu().numpy()
            queued_actions = [action_chunk[i] for i in range(action_chunk.shape[0])]
            chunk_idx += 1

        outputs.append(queued_actions.pop(0))

    return np.stack(outputs, axis=0)


def compute_metrics(actions: np.ndarray, fps: float, chunk_size: int) -> dict:
    step = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    jerk = np.linalg.norm(np.diff(actions, n=2, axis=0), axis=1)

    centered = actions - actions.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(centered, axis=0)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(actions.shape[0], d=1.0 / fps)
    total_power = power[1:].sum()

    def band_ratio(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(power[mask].sum() / total_power) if total_power > 0 else 0.0

    step_mod_p95 = {}
    for mod in range(chunk_size):
        mod_values = step[np.arange(step.shape[0]) % chunk_size == mod]
        if len(mod_values) == 0:
            continue
        step_mod_p95[str(mod)] = float(np.percentile(mod_values, 95))

    return {
        "num_frames": int(actions.shape[0]),
        "step_mean": float(step.mean()),
        "step_p95": float(np.percentile(step, 95)),
        "step_p99": float(np.percentile(step, 99)),
        "step_max": float(step.max()),
        "jerk_mean": float(jerk.mean()) if len(jerk) else 0.0,
        "jerk_p95": float(np.percentile(jerk, 95)) if len(jerk) else 0.0,
        "jerk_p99": float(np.percentile(jerk, 99)) if len(jerk) else 0.0,
        "jerk_max": float(jerk.max()) if len(jerk) else 0.0,
        "band_ratio_0_1hz": band_ratio(0.0, 1.0),
        "band_ratio_1_3hz": band_ratio(1.0, 3.0),
        "band_ratio_3_6hz": band_ratio(3.0, 6.0),
        "band_ratio_6_12hz": band_ratio(6.0, 12.0),
        "band_ratio_12_15hz": band_ratio(12.0, 15.0),
        "step_mod_p95": step_mod_p95,
    }


def save_plots(
    traces: dict[str, np.ndarray],
    output_path: Path,
    fps: float,
    chunk_size: int,
    joint_indices: list[int],
) -> None:
    num_steps = next(iter(traces.values())).shape[0]
    times = np.arange(num_steps) / fps

    fig, axes = plt.subplots(2 + len(joint_indices), 1, figsize=(16, 4 * (2 + len(joint_indices))), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for name, actions in traces.items():
        step = np.linalg.norm(np.diff(actions, axis=0), axis=1)
        axes[0].plot(times[1:], step, label=name, linewidth=1.5)
    axes[0].set_title("Step Norm")
    axes[0].set_ylabel("L2")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for name, actions in traces.items():
        jerk = np.linalg.norm(np.diff(actions, n=2, axis=0), axis=1)
        axes[1].plot(times[2:], jerk, label=name, linewidth=1.5)
    axes[1].set_title("Jerk Norm")
    axes[1].set_ylabel("L2")
    axes[1].grid(True, alpha=0.3)

    for ax in axes[:2]:
        for boundary in range(chunk_size, num_steps, chunk_size):
            ax.axvline(boundary / fps, color="k", linestyle="--", alpha=0.15)

    for plot_idx, joint_idx in enumerate(joint_indices, start=2):
        ax = axes[plot_idx]
        for name, actions in traces.items():
            ax.plot(times, actions[:, joint_idx], label=name, linewidth=1.2)
        ax.set_title(f"Joint {joint_idx} Policy Action")
        ax.set_ylabel("deg")
        ax.grid(True, alpha=0.3)
        for boundary in range(chunk_size, num_steps, chunk_size):
            ax.axvline(boundary / fps, color="k", linestyle="--", alpha=0.15)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    spectrum_path = output_path.with_name(output_path.stem + "_spectrum.png")
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, actions in traces.items():
        centered = actions - actions.mean(axis=0, keepdims=True)
        fft = np.fft.rfft(centered, axis=0)
        power = np.abs(fft) ** 2
        mean_power = power.mean(axis=1)
        freqs = np.fft.rfftfreq(actions.shape[0], d=1.0 / fps)
        ax.plot(freqs[1:], mean_power[1:], label=name, linewidth=1.5)
    ax.set_title("Mean Joint Power Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(spectrum_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    device = resolve_device(args.device)

    cfg = TrainPipelineConfig.from_pretrained(checkpoint)
    cfg.policy.device = device
    cfg.dataset.video_backend = "pyav"

    dataset = make_dataset(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg.policy, dataset_stats=dataset.meta.stats)
    image_features = list(cfg.policy.image_features.keys())
    action_dim = cfg.policy.action_feature.shape[0]
    chunk_size = cfg.policy.n_action_steps
    horizon = cfg.policy.horizon

    variants = build_variants(cfg.policy, args)
    segment_indices = find_segment_indices(dataset, args.episode_index, args.start_frame, args.length)
    num_chunks = math.ceil(args.length / chunk_size)
    dtype = torch.float32
    noises = make_fixed_noises(num_chunks, horizon, action_dim, device, dtype, args.seed)

    traces: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    skipped: dict[str, str] = {}

    policy_cls = get_policy_class(cfg.policy.type)
    for variant in variants:
        variant_cfg = deepcopy(cfg.policy)
        variant_cfg.device = device
        variant_cfg.num_inference_steps = variant.num_inference_steps
        variant_cfg.solver_type = variant.solver_type
        variant_cfg.clip_sample = variant.clip_sample

        try:
            policy = policy_cls.from_pretrained(checkpoint, config=variant_cfg)
        except ImportError as exc:
            skipped[variant.name] = str(exc)
            continue

        actions = replay_variant(
            policy=policy,
            dataset=dataset,
            segment_indices=segment_indices,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            image_features=image_features,
            noises=noises,
        )
        traces[variant.name] = actions
        metrics[variant.name] = {
            **asdict(variant),
            **compute_metrics(actions, fps=30.0, chunk_size=chunk_size),
        }

    if not traces:
        raise RuntimeError(f"All variants failed to run. Skipped: {skipped}")

    output_dir = args.output_dir or checkpoint.parent.parent / "analysis" / "rollout_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    joint_indices = [int(item.strip()) for item in args.joint_indices.split(",") if item.strip()]
    prefix = f"ep{args.episode_index:04d}_frame{args.start_frame:04d}_len{args.length}"

    report = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(cfg.dataset.root),
        "episode_index": args.episode_index,
        "start_frame": args.start_frame,
        "length": args.length,
        "device": device,
        "seed": args.seed,
        "baseline_config": {
            "n_obs_steps": cfg.policy.n_obs_steps,
            "n_action_steps": cfg.policy.n_action_steps,
            "horizon": cfg.policy.horizon,
            "num_inference_steps": cfg.policy.num_inference_steps,
            "solver_type": cfg.policy.solver_type,
            "clip_sample": cfg.policy.clip_sample,
            "clip_sample_range": cfg.policy.clip_sample_range,
        },
        "segment_dataset_indices": segment_indices,
        "metrics": metrics,
        "skipped": skipped,
    }

    report_path = output_dir / f"{prefix}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    plot_path = output_dir / f"{prefix}_traces.png"
    save_plots(traces, plot_path, fps=30.0, chunk_size=chunk_size, joint_indices=joint_indices)

    summary = {
        "report": str(report_path),
        "trace_plot": str(plot_path),
        "spectrum_plot": str(plot_path.with_name(plot_path.stem + "_spectrum.png")),
        "variants_ran": list(traces.keys()),
        "variants_skipped": skipped,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
