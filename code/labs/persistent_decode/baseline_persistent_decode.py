"""Baseline per-token decode: one kernel launch per timestep."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


class BaselinePersistentDecodeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Naive decode: host loop over tokens, launch work per timestep."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.output: Optional[torch.Tensor] = None
        batch, seq_len, _ = resolve_shapes()
        self.seq_len = seq_len
        self.batch = batch
        self.head_dim = None
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.inputs = build_inputs(self.device)
        self.head_dim = self.inputs.q.shape[-1]
        self._synchronize()

    def _decode_step(self, t: int) -> None:
        assert self.inputs is not None
        # Compute a simple dot per sequence for timestep t, then scale V.
        q_t = self.inputs.q[:, t, :]  # [batch, head_dim]
        k_t = self.inputs.k[:, t, :]
        v_t = self.inputs.v[:, t, :]

        dot = (q_t * k_t).sum(dim=-1, keepdim=True)  # [batch, 1]
        self.inputs.out[:, t, :] = v_t * dot

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Inputs not initialized")

        with self._nvtx_range("baseline_per_token"):
            for t in range(self.seq_len):
                self._decode_step(t)
            if self.inputs is not None:
                # Capture a slice of the output tensor
                self.output = self.inputs.out[:1, : min(8, self.inputs.out.shape[1])].detach().float().clone()
        if self.inputs is None or self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={
                "q": self.inputs.q.detach(),
                "k": self.inputs.k.detach(),
                "v": self.inputs.v.detach(),
            },
            output=self.output,
            batch_size=self.batch,
            parameter_count=0,
            precision_flags={
                "fp16": self.inputs.q.dtype == torch.float16,
                "bf16": self.inputs.q.dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.output = None

    def get_config(self) -> BenchmarkConfig:
        # Keep iterations small; focus on relative speedups and profiling
        return BenchmarkConfig(
            iterations=12,
            warmup=5,
            measurement_timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "persistent_decode.batch_size": float(getattr(self, 'batch_size', 0)),
            "persistent_decode.seq_len": float(getattr(self, 'seq_len', 0)),
            "persistent_decode.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None

def get_benchmark() -> BaseBenchmark:
    return BaselinePersistentDecodeBenchmark()

if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
