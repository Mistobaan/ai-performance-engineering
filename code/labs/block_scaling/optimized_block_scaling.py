"""Hardware blockscaled GEMM benchmark for the block scaling lab."""

from __future__ import annotations

import os
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark
from labs.block_scaling.block_scaling_benchmarks import (
    OptimizedBlockScalingBenchmarkBase,
)


def _skip_setup_verify() -> bool:
    value = os.getenv("AISP_BLOCK_SCALING_SKIP_VERIFY", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


class OptimizedBlockScalingBenchmark(OptimizedBlockScalingBenchmarkBase):
    """Compile once and measure only Blackwell's hardware blockscaled path."""

    def _post_setup(self) -> None:
        if not _skip_setup_verify():
            self.verification_summary = self._require_problem().verify_close()


def get_benchmark() -> BaseBenchmark:
    return OptimizedBlockScalingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
