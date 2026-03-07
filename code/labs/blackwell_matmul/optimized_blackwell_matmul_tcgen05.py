"""Part 5: tcgen05 + TMEM + CTA-group variants (Blackwell SM100/SM12x)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.tcgen05_requirements import ensure_tcgen05_supported
from labs.blackwell_matmul import (
    baseline_blackwell_matmul,
    optimized_blackwell_matmul_tcgen05,
    optimized_blackwell_matmul_tcgen05_cta2,
)
from labs.blackwell_matmul.blackwell_benchmarks import (
    FeatureDescriptor,
    GraceBlackwellMatmulBenchmark,
)


class Tcgen05GraceBlackwellBenchmark(GraceBlackwellMatmulBenchmark):
    def __init__(self, size: int = 2048) -> None:
        ensure_tcgen05_supported()
        descriptor = FeatureDescriptor(
            tag="tcgen05_tmem",
            notes="tcgen05 tensor cores + TMEM epilogue (single-CTA per tile)",
        )
        super().__init__(
            runner=optimized_blackwell_matmul_tcgen05,
            label="grace_blackwell_matmul_tcgen05",
            size=size,
            iterations=5,
            warmup=5,
            descriptor=descriptor,
            reference_runner=baseline_blackwell_matmul,
        )
        self.required_capabilities = {"tcgen05": True}

    def setup(self) -> None:
        super().setup()
        assert self._lhs is not None and self._rhs is not None
        # Build/load the tcgen05 extension outside the timed region. If the
        # environment cannot compile the inline extension, skip instead of
        # failing the whole benchmark suite.
        try:
            with torch.no_grad():
                _ = self._runner(self._lhs, self._rhs)
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            raise RuntimeError(f"SKIPPED: tcgen05 inline extension unavailable ({exc})") from exc


class Tcgen05Cta2GraceBlackwellBenchmark(GraceBlackwellMatmulBenchmark):
    def __init__(self, size: int = 2048) -> None:
        ensure_tcgen05_supported()
        descriptor = FeatureDescriptor(
            tag="tcgen05_cta2",
            notes="tcgen05 + TMEM with CTA-group::2 multicast (cluster launch)",
        )
        super().__init__(
            runner=optimized_blackwell_matmul_tcgen05_cta2,
            label="grace_blackwell_matmul_tcgen05_cta2",
            size=size,
            iterations=5,
            warmup=5,
            descriptor=descriptor,
            reference_runner=baseline_blackwell_matmul,
        )
        self.required_capabilities = {"tcgen05": True, "cta_group": True}

    def setup(self) -> None:
        super().setup()
        assert self._lhs is not None and self._rhs is not None
        try:
            with torch.no_grad():
                _ = self._runner(self._lhs, self._rhs)
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            raise RuntimeError(f"SKIPPED: tcgen05 inline extension unavailable ({exc})") from exc

def get_benchmark() -> GraceBlackwellMatmulBenchmark:
    """Factory for discover_benchmarks()."""
    return Tcgen05GraceBlackwellBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
