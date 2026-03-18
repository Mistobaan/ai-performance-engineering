#!/usr/bin/env python3
"""Build and run all Ozaki scheme variants in one place."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.cuda_binary_benchmark import ARCH_SUFFIX, detect_supported_arch

_METRIC_PATTERNS = {
    "variant": re.compile(r"VARIANT:\s*(\S+)"),
    "time_ms": re.compile(r"TIME_MS:\s*([0-9.eE+-]+)"),
    "tflops": re.compile(r"TFLOPS:\s*([0-9.eE+-]+)"),
    "retained_bits": re.compile(r"RETAINED_BITS:\s*(-?\d+)"),
    "emulation_used": re.compile(r"EMULATION_USED:\s*(\d+)"),
    "max_abs_error": re.compile(r"MAX_ABS_ERROR:\s*([0-9.eE+-]+)"),
    "mean_abs_error": re.compile(r"MEAN_ABS_ERROR:\s*([0-9.eE+-]+)"),
}


def parse_metrics(stdout: str) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    for key, pattern in _METRIC_PATTERNS.items():
        match = pattern.search(stdout)
        if not match:
            continue
        value = match.group(1)
        if key == "variant":
            metrics[key] = value
        elif key in {"retained_bits", "emulation_used"}:
            metrics[key] = int(value)
        else:
            metrics[key] = float(value)
    return metrics


def run_binary(binary: str, args: list[str]) -> dict[str, float | str]:
    completed = subprocess.run(
        [str(LAB_DIR / binary), *args],
        cwd=LAB_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{binary} failed with rc={completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    metrics = parse_metrics(completed.stdout)
    metrics["stdout"] = completed.stdout
    return metrics


def format_row(label: str, metrics: dict[str, float | str], baseline_ms: float) -> str:
    time_ms = float(metrics.get("time_ms", 0.0))
    speedup = baseline_ms / time_ms if time_ms > 0 else 0.0
    retained_bits = metrics.get("retained_bits", "-")
    emulation_used = metrics.get("emulation_used", 0)
    return (
        f"| {label} | {time_ms:.3f} | {float(metrics.get('tflops', 0.0)):.3f} | "
        f"{speedup:.2f}x | {retained_bits} | {emulation_used} | "
        f"{float(metrics.get('max_abs_error', 0.0)):.3e} | "
        f"{float(metrics.get('mean_abs_error', 0.0)):.3e} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="Skip `make all` before running")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input-scale", type=float, default=0.001)
    parser.add_argument("--dynamic-max-bits", type=int, default=16)
    parser.add_argument("--dynamic-offset", type=int, default=-56)
    parser.add_argument("--fixed-bits", type=int, default=12)
    args = parser.parse_args()

    if not args.skip_build:
        subprocess.run(["make", "all"], cwd=LAB_DIR, check=True)

    arch = detect_supported_arch()
    suffix = ARCH_SUFFIX[arch]

    common_args = [
        "--m", str(args.m),
        "--n", str(args.n),
        "--k", str(args.k),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
        "--seed", str(args.seed),
        "--input-scale", str(args.input_scale),
    ]
    baseline = run_binary(f"baseline_ozaki_scheme{suffix}", common_args)
    dynamic = run_binary(
        f"optimized_ozaki_scheme_dynamic{suffix}",
        common_args + ["--dynamic-max-bits", str(args.dynamic_max_bits), "--dynamic-offset", str(args.dynamic_offset)],
    )
    fixed = run_binary(
        f"optimized_ozaki_scheme_fixed{suffix}",
        common_args + ["--fixed-bits", str(args.fixed_bits)],
    )

    baseline_ms = float(baseline["time_ms"])
    print("| Variant | Time (ms) | TFLOPS | Speedup vs native | Retained bits | Emulation used | Max abs error | Mean abs error |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(format_row("Native FP64", baseline, baseline_ms))
    print(format_row("Ozaki dynamic", dynamic, baseline_ms))
    print(format_row("Ozaki fixed", fixed, baseline_ms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
