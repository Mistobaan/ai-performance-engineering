"""Optimized EOS polling in decode loop: async stream-polled host checks.

Inspired by:
https://chaimrand.medium.com/optimizing-token-generation-in-pytorch-decoder-models-8e63b5a5fc80
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.hf_decoder_cache_benchmark import (  # noqa: E402
    HFDecoderCacheBenchmark,
    HFDecoderCacheConfig,
    attach_benchmark_metadata,
)


def get_benchmark() -> HFDecoderCacheBenchmark:
    cfg = HFDecoderCacheConfig(
        batch_size=4,
        prompt_tokens=128,
        decode_tokens=192,
        hidden_size=192,
        num_layers=3,
        num_heads=3,
        cache_mode="static",
        compile_decode_step=True,
        eos_sync_mode="async_streamed",
        # Poll less frequently to amortize host synchronization overhead.
        eos_poll_interval=96,
        stop_on_all_done=False,
        iterations=8,
        warmup=8,
        label="optimized_eos_sync_polling",
    )
    return attach_benchmark_metadata(HFDecoderCacheBenchmark(cfg), __file__)  # type: ignore[return-value]


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
