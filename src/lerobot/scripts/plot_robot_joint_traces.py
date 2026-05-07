#!/usr/bin/env python

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_matrix(df: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in df.columns:
        return None
    return np.stack(df[column].to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot recorded robot joint traces from a LeRobot parquet file.")
    parser.add_argument("parquet", type=Path, help="Path to a parquet file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to <parquet_stem>_joint_traces.png",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    policy_action = _load_matrix(df, "action")
    sent_action = _load_matrix(df, "complementary_info.sent_action")
    feedback_action = _load_matrix(df, "complementary_info.feedback_action")
    observation_state = _load_matrix(df, "observation.state")

    if policy_action is None or observation_state is None:
        raise ValueError("Parquet must contain both 'action' and 'observation.state'.")

    if feedback_action is None:
        feedback_action = observation_state

    num_joints = observation_state.shape[1]
    ncols = 2
    nrows = (num_joints + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 3 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)

    for joint_idx in range(num_joints):
        ax = axes[joint_idx]
        ax.plot(policy_action[:, joint_idx], label="policy_action", linewidth=1.0, alpha=0.9)
        if sent_action is not None:
            ax.plot(sent_action[:, joint_idx], label="sent_action", linewidth=1.0, alpha=0.9)
        ax.plot(feedback_action[:, joint_idx], label="feedback_action", linewidth=1.0, alpha=0.9)
        ax.set_title(f"Joint {joint_idx}")
        ax.grid(alpha=0.25)

    for ax in axes[num_joints:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(args.parquet.name)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output = args.output or args.parquet.with_name(f"{args.parquet.stem}_joint_traces.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
