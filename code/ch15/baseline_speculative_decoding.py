"""Baseline speculative decoding benchmark (target-only greedy decode)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.speculative_decoding_benchmarks import (
    SpeculativeDecodingBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    bench = SpeculativeDecodingBenchmark(
        use_speculative=False,
        label="baseline_speculative_decoding",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
