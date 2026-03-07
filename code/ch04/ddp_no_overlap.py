"""ddp_no_overlap.py - DDP-style step without communication/compute overlap.

This baseline uses standard DDP which synchronizes gradient all-reduce
after the backward pass completes. No overlap between communication and
computation.

This module is an implementation detail; the harness entrypoint is
`baseline_no_overlap.py`.

The optimized variant is implemented in `ddp_overlap.py` and exposed via
`optimized_no_overlap.py`.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from core.benchmark.gpu_requirements import skip_if_insufficient_gpus

try:
    from distributed_helper import setup_single_gpu_env
except ImportError:
    def setup_single_gpu_env():
        if "RANK" not in os.environ:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("LOCAL_RANK", "0")

from typing import Optional

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from ch04.verification_payload_mixin import VerificationPayloadMixin


class MultiLayerNet(nn.Module):
    """Multi-layer network for benchmarking."""
    
    def __init__(self, size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class BaselineNoOverlapBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """DDP without communication overlap - baseline."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        self.output = None
        self._host_buffer = None
        self.rank = 0
        self.world_size = 1
        self.initialized = False
        self.batch_size = 128
        self.hidden_size = 1024
        tokens = self.batch_size * self.hidden_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: smoke-fast, no distributed init to avoid hangs."""
        print("[baseline_no_overlap] setup starting", flush=True)
        skip_if_insufficient_gpus()
        # Treat as single process even if torchrun sets RANK/LOCAL_RANK; skip dist.init.
        self.rank = 0
        self.world_size = 1
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        print("[baseline_no_overlap] set_device done", flush=True)

        torch.manual_seed(42)
        print("[baseline_no_overlap] seed set", flush=True)
        model = MultiLayerNet(self.hidden_size).to(self.device)
        print("[baseline_no_overlap] model to device", flush=True)
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        print("[baseline_no_overlap] optimizer ready", flush=True)

        self.data = torch.randn(self.batch_size, self.hidden_size, device=self.device)
        self.target = torch.randn(self.batch_size, 1, device=self.device)
        self._host_buffer = torch.empty_like(self.data, device="cpu")
        print("[baseline_no_overlap] data allocated", flush=True)
        torch.cuda.synchronize(self.device)
        print("[baseline_no_overlap] setup done", flush=True)
    
    def benchmark_fn(self) -> None:
        """Benchmark: DDP training step without overlap."""
        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("no_overlap", enable=enable_nvtx):
            output = self.model(self.data)
            loss = nn.functional.mse_loss(output, self.target)
            loss.backward()
            self._host_buffer.copy_(self.data, non_blocking=False)
            self.data.copy_(self._host_buffer, non_blocking=False)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.output = output.detach()

    def capture_verification_payload(self) -> None:
        if self.data is None or self.target is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        param_count = sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0
        self._set_verification_payload(
            inputs={"data": self.data, "target": self.target},
            output=self.output,
            batch_size=int(self.batch_size),
            parameter_count=param_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if dist.is_initialized() and self.initialized:
            dist.destroy_process_group()
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        torch.cuda.empty_cache()
        self._config = None
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration (smoke-fast)."""
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.data is None:
            return "Data tensor not initialized"
        if self.target is None:
            return "Target tensor not initialized"
        # Check that model can produce valid output
        try:
            with torch.no_grad():
                test_output = self.model(self.data)
                if test_output.shape[0] != 128:
                    return f"Output batch size mismatch: expected 128, got {test_output.shape[0]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineNoOverlapBenchmark()


def _parse_args():
    parser = argparse.ArgumentParser(description="DDP baseline without communication overlap.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of measurement iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations.")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable profiling for the run.")
    parser.add_argument("--enable-memory-tracking", action="store_true", help="Track GPU memory usage.")
    return parser.parse_args()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
