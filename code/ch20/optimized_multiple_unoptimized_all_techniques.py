"""optimized_multiple_unoptimized_all_techniques.py - Alias to the combined techniques benchmark.

This file keeps the "all techniques" variant discoverable as an optimized_* suffix
for `baseline_multiple_unoptimized.py`. The canonical implementation lives in
`optimized_multiple_unoptimized.py`.
"""

from __future__ import annotations

import ch20.arch_config  # noqa: F401 - Apply chapter defaults

from ch20.optimized_multiple_unoptimized import OptimizedAllTechniquesBenchmark
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    return OptimizedAllTechniquesBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
