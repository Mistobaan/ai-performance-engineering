"""Part 4: cluster DSMEM broadcast and Grace Hopper cluster launch."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.blackwell_matmul import (
    baseline_blackwell_matmul,
    optimized_blackwell_matmul_cluster,
)
from labs.blackwell_matmul.blackwell_benchmarks import (
    FeatureDescriptor,
    GraceBlackwellMatmulBenchmark,
)


class ClusterGraceBlackwellBenchmark(GraceBlackwellMatmulBenchmark):
    def __init__(self, size: int = 2048) -> None:
        descriptor = FeatureDescriptor(
            tag="cluster",
            notes="Part 4: DSMEM broadcast + CTA clusters (Grace-Blackwell)",
        )
        super().__init__(
            runner=optimized_blackwell_matmul_cluster,
            label="grace_blackwell_matmul_cluster",
            size=size,
            iterations=5,
            warmup=5,
            descriptor=descriptor,
            reference_runner=baseline_blackwell_matmul,
        )

def get_benchmark() -> GraceBlackwellMatmulBenchmark:
    return ClusterGraceBlackwellBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
