"""Baseline TPOT/long-output disaggregated prefill/decode benchmark (single GPU)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch17.prefill_decode_disagg_multigpu_common import PrefillDecodeConfig
from ch17.prefill_decode_disagg_single_common import (
    BaselinePrefillDecodeSingleGPUBenchmark,
    attach_benchmark_metadata,
)

TPOT_LONG_CONFIG = PrefillDecodeConfig(
    hidden_size=1024,
    num_layers=1,
    batch_size=4,
    requests_per_rank=8,
    context_window=4096,
    decode_tokens=1024,
)


def get_benchmark() -> BaseBenchmark:
    bench = BaselinePrefillDecodeSingleGPUBenchmark(
        label="baseline_prefill_decode_disagg_tpot_long",
        cfg=TPOT_LONG_CONFIG,
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
