"""Optimized TPOT/long-output disaggregated prefill/decode benchmark (single GPU)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch17.prefill_decode_disagg_multigpu_common import PrefillDecodeConfig
from ch17.prefill_decode_disagg_single_common import (
    OptimizedPrefillDecodeSingleGPUBenchmark,
    attach_benchmark_metadata,
)

TPOT_LONG_CONFIG = PrefillDecodeConfig(
    context_window=2048,
    decode_tokens=1024,
    requests_per_rank=16,
    num_layers=2,
)


def get_benchmark() -> BaseBenchmark:
    bench = OptimizedPrefillDecodeSingleGPUBenchmark(
        label="optimized_prefill_decode_disagg_tpot_long",
        cfg=TPOT_LONG_CONFIG,
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
