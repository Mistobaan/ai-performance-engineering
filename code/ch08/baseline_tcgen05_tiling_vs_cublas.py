"""Baseline tcgen05 tiling matmul for cuBLAS comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.tcgen05_tiling_vs_cublas_benchmark_base import Tcgen05TilingVsCublasBase
from core.benchmark.tcgen05_requirements import ensure_tcgen05_supported
from core.common.tcgen05 import load_tiling_tcgen05_module
from core.harness.benchmark_harness import BaseBenchmark


class BaselineTcgen05TilingVsCublasBenchmark(Tcgen05TilingVsCublasBase):
    """Baseline: tcgen05 tiling kernel with pre-transposed B."""

    nvtx_label = "baseline_tcgen05_tiling_vs_cublas"

    def __init__(self) -> None:
        super().__init__()
        self.extension = None
        self.matrix_b_t = None

    def setup(self) -> None:
        ensure_tcgen05_supported(
            loader=load_tiling_tcgen05_module,
            module_name="ch08 tcgen05 tiling vs cuBLAS",
        )
        super().setup()
        if self.matrix_b is None:
            raise RuntimeError("Input matrices not initialized")
        if self.extension is None:
            self.extension = load_tiling_tcgen05_module()
        self.matrix_b_t = self.matrix_b.t().contiguous()

    def benchmark_fn(self) -> None:
        if self.extension is None or self.matrix_a is None or self.matrix_b_t is None:
            raise RuntimeError("Inputs or extension not initialized")
        with self._nvtx_range(self.nvtx_label):
            with torch.no_grad():
                self.output = self.extension.matmul_tiling_tcgen05_pretransposed(
                    self.matrix_a, self.matrix_b_t
                )

    def teardown(self) -> None:
        self.matrix_b_t = None
        super().teardown()


def get_benchmark() -> BaseBenchmark:
    return BaselineTcgen05TilingVsCublasBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
