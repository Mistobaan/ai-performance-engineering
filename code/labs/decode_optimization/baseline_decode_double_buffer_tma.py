"""Baseline: CUDA decode kernel (global-load) for the TMA double-buffered comparison.

This benchmark reuses the `labs/moe_cuda` baseline kernel wrapper so that
`optimized_decode_double_buffer_tma.py` has a valid, equivalent baseline.
"""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.moe_cuda.baseline_decode_kernel import BaselineDecodeKernelBenchmark


class BaselineDecodeDoubleBufferTmaBenchmark(BaselineDecodeKernelBenchmark):
    """Explicit wrapper so inherited seed/setup metadata stays visible to audits."""


def get_benchmark() -> BaseBenchmark:
    return BaselineDecodeDoubleBufferTmaBenchmark()

