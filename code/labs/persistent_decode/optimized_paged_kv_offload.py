"""Optimized paged KV-cache benchmark with pinned staging + opportunistic FP8 KV.

- Uses pinned staging buffers with direct H2D copies.
- Uses FP8 KV only when a fused FlashAttention path is available on B200/GB200.
- Falls back to FP16 when the runtime/profiler stack disables that fused FP8 path.
"""

from __future__ import annotations

import torch

from labs.persistent_decode.paged_kv_offload_common import PagedKVConfig, PagedKVOffloadBenchmark


def get_benchmark() -> PagedKVOffloadBenchmark:
    cfg = PagedKVConfig(
        batch_size=4,
        num_heads=16,
        head_dim=128,
        max_seq_len=32768,
        page_tokens=2048,
        decode_tokens=64,
        repeat_pages=32,
        use_pinned_stage=True,
        use_async_stream=False,
        use_memmap=True,
        prefer_fp8=True,
        # Keep timing and profiler runs on the same semantic path: prefer FP8 when
        # fused attention is available, otherwise fall back to FP16 instead of
        # making profiling-only subprocesses fail.
        require_fused_fp8=False,
        fallback_dtype=torch.float16,
        prefetch_next_page=False,
        use_direct_h2d=True,
    )
    return PagedKVOffloadBenchmark(cfg, label="paged_kv_offload_optimized")


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
