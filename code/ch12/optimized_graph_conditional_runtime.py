"""Optimized graph execution with CUDA graph replay.

This module demonstrates CUDA graph replay benefits - eliminating kernel
launch overhead by pre-compiling the execution graph.

Key Features:
- CUDA graph capture and replay
- Eliminates per-kernel launch overhead
- Batches multiple operations into single submission

For conditional graph nodes (CUDA 12.4+), see the CUDA binary implementations.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.cuda import CUDAGraph

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


def supports_conditional_graphs() -> bool:
    """Check if conditional graph nodes are supported (CUDA 12.4+)."""
    if not torch.cuda.is_available():
        return False
    cuda_version = torch.version.cuda
    if cuda_version:
        try:
            version_num = int(cuda_version.replace(".", "")[:3])
            return version_num >= 124
        except (ValueError, TypeError):
            pass
    return False


class OptimizedGraphBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized benchmark using CUDA graph replay.
    
    CUDA graphs eliminate kernel launch overhead by:
    1. Capturing the execution pattern once
    2. Replaying it with a single GPU submission
    
    This is ~10-100x faster for launch-bound workloads.
    """
    
    def __init__(self):
        super().__init__()
        # Keep the tensor small enough that kernel launch overhead is visible.
        # CUDA graph replay amortizes those launches in the steady-state loop.
        self.batch_size = 16
        self.seq_len = 64
        self.hidden_dim = 256
        # Increase the number of tiny ops so the workload is launch-bound and
        # CUDA graph replay shows a clear steady-state speedup.
        self.num_loops = 256
        
        self.data: Optional[torch.Tensor] = None
        self._graph: Optional[CUDAGraph] = None
        self._graph_stream: Optional[torch.cuda.Stream] = None
        self._verify_input: Optional[torch.Tensor] = None
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def _compute_ops(self) -> None:
        """Operations to capture in graph - uses in-place ops only."""
        # All in-place operations for stable tensor addresses
        self.data.mul_(0.99)
        for _ in range(self.num_loops):  # num_loops × 2 ops = 2*num_loops launches normally
            self.data.add_(0.001)
            self.data.mul_(1.0001)
        self.data.relu_()
        self.data.mul_(1.001)
    
    def setup(self) -> None:
        """Setup CUDA graph with static buffers."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        
        # Static buffer - address must not change
        self.data = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=dtype
        )
        self._verify_input = self.data.detach().clone()

        # Non-default stream required for graph capture.
        self._graph_stream = torch.cuda.Stream()

        # Capture the execution graph while preserving the initial state so timed
        # iterations begin from the same tensor as the baseline variant.
        initial_state = self.data.detach().clone()
        self._graph = CUDAGraph()
        self._synchronize()
        with torch.cuda.stream(self._graph_stream):
            with torch.cuda.graph(self._graph, stream=self._graph_stream):
                self._compute_ops()
        self.data.copy_(initial_state)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark CUDA graph replay - much faster than fresh launches."""
        with self._nvtx_range("graph_replay"):
            self._graph.replay()
        if self._verify_input is None or self.data is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        if self._verify_input is None:
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
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        """Clean up."""
        self.data = None
        self._graph = None
        self._graph_stream = None
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
        """Return graph metrics using standard helpers."""
        from core.benchmark.metrics import compute_graph_metrics
        
        # Estimate launch overhead (typical kernel launch is 5-10us)
        num_nodes = (2 * self.num_loops) + 3
        baseline_launch_us = 8.0 * num_nodes  # ~8us per tiny launch
        graph_launch_us = 15.0  # Single graph launch overhead
        
        metrics = compute_graph_metrics(
            baseline_launch_overhead_us=baseline_launch_us,
            graph_launch_overhead_us=graph_launch_us,
            num_nodes=num_nodes,
            num_iterations=100,
        )
        metrics["graph.uses_cuda_graph"] = 1.0
        metrics["graph.conditional_support"] = 1.0 if supports_conditional_graphs() else 0.0
        return metrics


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedGraphBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
