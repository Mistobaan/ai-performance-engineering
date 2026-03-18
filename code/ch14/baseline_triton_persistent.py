"""Baseline Triton GEMM - Multiple kernel launches for batched operations.

This baseline demonstrates the cost of multiple kernel launches for
batched matrix multiplications, which persistent kernels optimize away.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


@triton.jit
def gemm_standard_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Standard GEMM: one block per output tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        mask_b = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        acc += tl.dot(a, b)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def matmul_standard_batched(
    a_batch: torch.Tensor,
    b_batch: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Multiple individual kernel launches for batched GEMM.
    
    This demonstrates the overhead of launching many small kernels.
    """
    batch_size = a_batch.shape[0]
    M, K = a_batch.shape[1], a_batch.shape[2]
    K, N = b_batch.shape[1], b_batch.shape[2]
    
    if out.shape != (batch_size, M, N):
        raise ValueError("matmul_standard_batched() requires a preallocated output buffer with matching shape")
    c_batch = out
    
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    
    # Launch separate kernel for each batch element (inefficient)
    for i in range(batch_size):
        a = a_batch[i]
        b = b_batch[i]
        c = c_batch[i]
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        gemm_standard_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    
    return c_batch


class BaselineTritonPersistentBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Multiple kernel launches for batched operations.
    
    Demonstrates the overhead of launching many small kernels,
    which persistent/batched kernels optimize away.
    """

    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self._output_buffer = None
        self.batch_size = 32  # Many small operations
        self.M = 256
        self.N = 256
        self.K = 256
        self.dtype = torch.float16
        self._last = 0.0
        
        flops = 2 * self.batch_size * self.M * self.N * self.K
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.M * self.N),
        )
        self.output = None
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.M * self.N),
        )

    def setup(self) -> None:
        """Setup: Initialize batched matrices."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        self.a = torch.randn(self.batch_size, self.M, self.K, device=self.device, dtype=self.dtype)
        self.b = torch.randn(self.batch_size, self.K, self.N, device=self.device, dtype=self.dtype)
        self._output_buffer = torch.empty(
            (self.batch_size, self.M, self.N), device=self.device, dtype=self.dtype
        )
        
        # Warmup
        for _ in range(3):
            _ = matmul_standard_batched(self.a, self.b, out=self._output_buffer)

    def benchmark_fn(self) -> None:
        """Benchmark: Multiple kernel launches."""
        self.output = matmul_standard_batched(self.a, self.b, out=self._output_buffer)
        self._last = float(self.output.sum())
        if self.output is None or self.a is None or self.b is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"a": self.a, "b": self.b},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.a = None
        self.b = None
        self._output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.a is None or self.b is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineTritonPersistentBenchmark()
