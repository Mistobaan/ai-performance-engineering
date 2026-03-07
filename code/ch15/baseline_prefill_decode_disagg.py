#!/usr/bin/env python3
"""Baseline prefill/decode disaggregation benchmark (single GPU)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.prefill_decode_disagg_common import (
    HostStagedPrefillDecodeDisaggBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    bench = HostStagedPrefillDecodeDisaggBenchmark(
        multi_gpu=False,
        label="baseline_prefill_decode_disagg",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
