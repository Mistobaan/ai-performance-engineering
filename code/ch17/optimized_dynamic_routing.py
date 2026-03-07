"""Vectorized dynamic routing benchmark."""

from __future__ import annotations

from ch17.baseline_dynamic_routing import (  # noqa: E402
    _DynamicRoutingBenchmark,
)


class OptimizedDynamicRoutingBenchmark(_DynamicRoutingBenchmark):
    """Vectorized routing using pre-allocated tensors.
    
    Optimizations:
    - Pre-allocated tensors avoid per-iteration allocation
    - Vectorized boolean operations instead of Python loops
    - Benefits show at larger batch sizes (1024+)
    """
    def __init__(self) -> None:
        # Use larger batch size where vectorization provides benefit
        super().__init__(batch_size=1024, vectorized=True)


def get_benchmark():
    return OptimizedDynamicRoutingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
