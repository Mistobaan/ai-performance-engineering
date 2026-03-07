"""Optimized NVFP4 MLP for the Chapter 12 throughput comparison."""

from __future__ import annotations

from core.benchmark.nvfp4_mlp import NVFP4MLPBenchmark, NVFP4MLPConfig
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    config = NVFP4MLPConfig(
        batch_size=512,
        d_model=8192,
        d_ff=32768,
        num_layers=2,
        iterations=20,
        warmup=10,
        name="ch12_nvfp4_mlp",
    )
    return NVFP4MLPBenchmark(config, use_nvfp4=True)
