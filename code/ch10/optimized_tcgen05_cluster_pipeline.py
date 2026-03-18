"""Optimized tcgen05 matmul using cluster launch with CUDA graph replay."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from core.benchmark.tcgen05_matmul_base import Tcgen05MatmulBenchmarkBase
from core.benchmark.tcgen05_requirements import ensure_tcgen05_supported
from core.common.tcgen05 import load_tcgen05_cluster_module
from core.harness.hardware_capabilities import ensure_dsmem_supported
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedTcgen05ClusterPipelineBenchmark(Tcgen05MatmulBenchmarkBase):
    """Chapter 10 optimized: cluster-launched tcgen05 GEMM."""

    matrix_rows = 6144
    matrix_cols = 6144
    shared_dim = 2048
    nvtx_label = "optimized_tcgen05_cluster_pipeline"

    def __init__(self) -> None:
        super().__init__()
        self.extension: Optional[object] = None
        self._matmul: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    def setup(self) -> None:
        ensure_dsmem_supported(description="tcgen05 cluster pipeline")
        ensure_tcgen05_supported(
            loader=load_tcgen05_cluster_module,
            module_name="ch10 tcgen05 cluster pipeline",
        )
        super().setup()
        if self.extension is None:
            self.extension = load_tcgen05_cluster_module()
        self._matmul = self.extension.matmul_tcgen05_cluster
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self._matmul is None or self.matrix_a is None or self.matrix_b is None:
            raise RuntimeError("Inputs or extension not initialized")
        with self._nvtx_range(self.nvtx_label):
            with torch.inference_mode():
                self.output = self._matmul(self.matrix_a, self.matrix_b)

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def teardown(self) -> None:
        super().teardown()

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_gemm_metrics

        metrics = compute_gemm_metrics(
            m=self.matrix_rows,
            n=self.matrix_cols,
            k=self.shared_dim,
            precision="fp16",
            bytes_per_element=2,
        )
        metrics["gemm.uses_tcgen05"] = 1.0
        metrics["gemm.cluster_launch"] = 1.0
        metrics["gemm.cuda_graph_replay"] = 0.0
        return metrics


def get_benchmark() -> BaseBenchmark:
    return OptimizedTcgen05ClusterPipelineBenchmark()
