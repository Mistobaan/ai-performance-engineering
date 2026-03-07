"""optimized_performance.py - Optimized performance benchmark with larger batch size."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch01.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass


from typing import Optional

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.verification_mixin import VerificationPayloadMixin
from ch01.workload_config import WORKLOAD


class OptimizedPerformanceBatchBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark implementation with larger batch size optimization."""

    signature_equivalence_group = "ch01_performance_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.workload = WORKLOAD
        self.batch_size = batch_size if batch_size != 32 else self.workload.microbatch_size
        self.model = None
        self.microbatches = None
        self.targets = None
        self.optimizer = None
        self.fusion = 8
        self.hidden_dim = self.workload.performance_hidden_dim
        self._verify_input = None
        self._verify_output = None
        self.parameter_count = 0
        samples = float(self.batch_size * self.workload.performance_microbatches)
        self.register_workload_metadata(samples_per_iteration=samples)
    
    def setup(self) -> None:
        """Setup: initialize model, fixed inputs, and verification output."""
        # Seed FIRST for deterministic verification
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 10),
        )
        
        if self.device.type == "cuda":
            # Optimization: Use FP16 for faster computation (tensor cores)
            self.model = self.model.half()
            dtype = torch.float16
            self.model = self.model.to(self.device)
            # Skip torch.compile for this small model - overhead exceeds benefit
            # The speedup comes from FP16 + batch fusion instead
        else:
            self.model = self.model.to(self.device)
            dtype = torch.float32
        
        # Match baseline: use eval() mode (baseline has this even though it does backward pass)
        self.model.eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        microbatches = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=dtype).contiguous()
            for _ in range(self.workload.performance_microbatches)
        ]
        targets = [
            torch.randint(0, 10, (self.batch_size,), device=self.device)
            for _ in range(self.workload.performance_microbatches)
        ]
        self.microbatches = microbatches
        self.targets = targets
        
        # Create FIXED verification input - output will be captured at END of benchmark_fn()
        # Use FP32 verification inputs so baseline/optimized signatures match.
        # The FP16 optimization remains in the timed training loop.
        self._verify_input = self.microbatches[0].float().clone()
        self._verify_output = None  # Will be set at end of benchmark_fn()
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        # Warm up compiled model so the measurement loop only sees steady-state cost.
        for _ in range(3):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(self.microbatches[0])
            loss = torch.nn.functional.cross_entropy(logits, self.targets[0])
            loss.backward()
            self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.optimizer.zero_grad(set_to_none=True)
        # Pre-build fused batches so the benchmark loop can issue fewer, larger kernels.
        self._fused_batches = []
        self._fused_targets = []
        for start in range(0, len(self.microbatches), self.fusion):
            batch = torch.cat(self.microbatches[start : start + self.fusion], dim=0)
            target = torch.cat(self.targets[start : start + self.fusion], dim=0)
            self._fused_batches.append(batch)
            self._fused_targets.append(target)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        with self._nvtx_range("optimized_performance_batch"):
            # Optimization: Larger batch size improves GPU utilization
            # Process more samples per forward pass, reducing overhead per sample
            for data, target in zip(self._fused_batches, self._fused_targets):
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(data)
                loss = torch.nn.functional.cross_entropy(logits, target)
                loss.backward()
                self.optimizer.step()

    def capture_verification_payload(self) -> None:
        if self.model is None or self._verify_input is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        # Convert to FP32 for consistent comparison with baseline
        with torch.no_grad():
            verify_input = self._verify_input
            # Cast verification input to model dtype outside the timed region.
            model_params = list(self.model.parameters())
            if model_params:
                verify_input = verify_input.to(dtype=model_params[0].dtype, device=self.device)
            self._verify_output = self.model(verify_input).float().clone()
        self._set_verification_payload(
            inputs={"verify_input": self._verify_input},
            output=self._verify_output,
            batch_size=self._verify_input.shape[0],
            parameter_count=int(self.parameter_count),
            precision_flags={
                "fp16": bool(model_params) and model_params[0].dtype == torch.float16,
                "bf16": bool(model_params) and model_params[0].dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32),
            },
            output_tolerance=(0.5, 0.5),
        )

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.microbatches, self.targets, self.optimizer
        self._fused_batches = None
        self._fused_targets = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=5,
            warmup=10,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_environment_metrics
        return compute_environment_metrics(
            gpu_count=getattr(self, 'gpu_count', 1),
            gpu_memory_gb=getattr(self, 'gpu_memory_gb', 80.0),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if not self.microbatches:
            return "Data not initialized"
        # Use first microbatch as validation probe
        probe = self.microbatches[0]
        try:
            with torch.no_grad():
                test_output = self.model(probe)
                if test_output.shape[0] != probe.shape[0]:
                    return f"Output batch size mismatch: expected {probe.shape[0]}, got {test_output.shape[0]}"
                if test_output.shape[1] != 10:
                    return f"Output shape mismatch: expected num_classes=10, got {test_output.shape[1]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedPerformanceBatchBenchmark(batch_size=32)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
