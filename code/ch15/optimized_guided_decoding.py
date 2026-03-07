"""Optimized guided decoding benchmark (GPU-resident mask reused across steps)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.guided_decoding_common import GuidedDecodingBenchmark, attach_benchmark_metadata


def get_benchmark() -> BaseBenchmark:
    bench = GuidedDecodingBenchmark(
        reuse_gpu_mask=True,
        label="optimized_guided_decoding",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
