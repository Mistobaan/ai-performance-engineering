#!/usr/bin/env python3
"""Optimized prefill/decode disaggregation benchmark (multi GPU)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.prefill_decode_disagg_common import (
    PeerPrefillDecodeDisaggBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    bench = PeerPrefillDecodeDisaggBenchmark(
        multi_gpu=True,
        label="optimized_prefill_decode_disagg_multigpu",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
