"""Baseline: Fresh kernel launches without CUDA graph.

This baseline demonstrates the overhead of launching kernels individually
without CUDA graph replay. Each operation requires a separate kernel launch
with associated driver overhead.

The optimized version captures all operations in a CUDA graph and replays
them with a single submission, eliminating the per-kernel launch overhead.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin


class BaselineGraphBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Fresh kernel launches each iteration.
    
    This shows the overhead of launching kernels individually:
    - Each operation triggers a kernel launch
    - Driver overhead per launch (~5-10us)
    - No batching of operations
    
    For small/fast kernels, launch overhead can dominate.
    """
    
    def __init__(self):
        super().__init__()
        # Keep the tensor small enough that kernel launch overhead is visible.
        # The optimized variant uses CUDA graph replay to amortize those launches.
        self.batch_size = 16
        self.seq_len = 64
        self.hidden_dim = 256
        # Increase the number of tiny ops so the workload is launch-bound and
        # CUDA graph replay shows a clear steady-state speedup.
        self.num_loops = 256
        
        self.data: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def _compute_ops(self) -> None:
        """Same operations as optimized, but fresh launches each time."""
        # Each of these triggers a kernel launch
        self.data.mul_(0.99)
        for _ in range(self.num_loops):  # num_loops × 2 ops = 2*num_loops launches
            self.data.add_(0.001)
            self.data.mul_(1.0001)
        self.data.relu_()
        self.data.mul_(1.001)
    
    def setup(self) -> None:
        """Setup data tensor."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        
        self.data = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=dtype
        )
        self._verify_input = self.data.detach().clone()

        # Match the optimized variant's unavoidable "one execution" during graph capture:
        # run the ops once, then restore state so timed iterations start from the same tensor.
        initial_state = self.data.detach().clone()
        self._compute_ops()
        self.data.copy_(initial_state)

    
    def benchmark_fn(self) -> None:
        """Benchmark fresh kernel launches."""
        with self._nvtx_range("fresh_kernel_launches"):
            self._compute_ops()
    
    def teardown(self) -> None:
        """Clean up."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=100,
            warmup=20,
            ncu_replay_mode="application",
            adaptive_iterations=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return baseline graph metrics using standard helpers."""
        from core.benchmark.metrics import compute_graph_metrics
        
        # Baseline has full launch overhead per iteration
        num_nodes = (2 * self.num_loops) + 3
        baseline_launch_us = 8.0 * num_nodes  # ~8us per tiny launch
        
        metrics = compute_graph_metrics(
            baseline_launch_overhead_us=baseline_launch_us,
            graph_launch_overhead_us=baseline_launch_us,  # Same as baseline (no graph)
            num_nodes=num_nodes,
            num_iterations=100,
        )
        metrics["graph.uses_cuda_graph"] = 0.0
        return metrics

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.data is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        dtype = self._verify_input.dtype
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.data.detach().clone(),
            batch_size=self._verify_input.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=self.get_output_tolerance(),
        )


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineGraphBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
