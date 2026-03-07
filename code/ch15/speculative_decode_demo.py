"""speculative_decode_demo.py - Chapter 15 speculative decoding demo (tool).

Runs the Chapter 15 speculative decoding baseline/optimized benchmarks and
prints a small summary (timing + speedup).
"""

from __future__ import annotations

import argparse

import torch

from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness, BenchmarkMode
from ch15.baseline_speculative_decoding import get_benchmark as get_baseline_benchmark
from ch15.optimized_speculative_decoding import get_benchmark as get_optimized_benchmark


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for speculative_decode_demo.py")


def _fmt_throughput(result) -> str:
    thr = getattr(result, "throughput", None)
    if thr is None:
        return "throughput=n/a"
    tokens_per_s = getattr(thr, "tokens_per_s", None)
    if tokens_per_s is None:
        return "throughput=n/a"
    return f"tokens_per_s={float(tokens_per_s):,.0f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 15 speculative decoding demo")
    parser.add_argument("--iterations", type=int, default=5, help="Measurement iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--nvtx", action="store_true", help="Enable NVTX ranges for Nsight Systems tracing.")
    args = parser.parse_args()

    _require_cuda()

    config = BenchmarkConfig(
        iterations=int(args.iterations),
        warmup=int(args.warmup),
        device=torch.device("cuda"),
        enable_nvtx=bool(args.nvtx),
        enable_profiling=bool(args.nvtx),
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)

    baseline = get_baseline_benchmark()
    optimized = get_optimized_benchmark()

    baseline_result = harness.benchmark(baseline, name="speculative_decode_baseline")
    optimized_result = harness.benchmark(optimized, name="speculative_decode_optimized")

    if baseline_result.errors:
        raise RuntimeError(f"Baseline failed: {baseline_result.errors}")
    if optimized_result.errors:
        raise RuntimeError(f"Optimized failed: {optimized_result.errors}")

    base_ms = float(baseline_result.timing.mean_ms)
    opt_ms = float(optimized_result.timing.mean_ms)
    speedup = (base_ms / opt_ms) if opt_ms > 0 else float("inf")

    print(f"baseline mean:  {base_ms:.3f} ms/iter ({_fmt_throughput(baseline_result)})")
    print(f"optimized mean: {opt_ms:.3f} ms/iter ({_fmt_throughput(optimized_result)})")
    print(f"speedup:        {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
