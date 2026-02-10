#!/usr/bin/env python3
"""
Plot NCCL env sensitivity summary produced by run_nccl_env_sensitivity.sh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def _load_profiles(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [p for p in payload.get("profiles", []) if p.get("status") == "ok"]


def main() -> int:
    apply_plot_style()

    parser = argparse.ArgumentParser(description="Plot NCCL env sensitivity peak bus bandwidth")
    parser.add_argument("--input", required=True, help="Input summary JSON")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default="NCCL Env Sensitivity", help="Plot title")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"ERROR: input summary not found: {in_path}")

    profiles = _load_profiles(in_path)
    if not profiles:
        raise SystemExit(f"ERROR: no successful profiles found in {in_path}")

    names = [p.get("profile", "unknown") for p in profiles]
    peaks = [float(p.get("peak_busbw_gbps", 0.0)) for p in profiles]
    speedups = [p.get("speedup_vs_baseline") for p in profiles]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1f77b4" if name != "baseline_auto" else "#2ca02c" for name in names]
    bars = ax.bar(names, peaks, color=colors, alpha=0.9)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        speed = speedups[i]
        label = f"{h:.1f} GB/s"
        if speed is not None:
            label = f"{label}\n{float(speed):.3f}x"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Peak bus bandwidth (GB/s)")
    ax.set_xlabel("NCCL env profile")
    ax.set_title(args.title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
