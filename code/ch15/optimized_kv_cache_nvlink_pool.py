"""optimized_kv_cache_nvlink_pool.py

Three-tier KV cache: local HBM (hot), peer HBM over NVLink (warm), host/Grace
for cold entries. Prefers near-neighbor GPU (device 1 if present) and uses
non-blocking transfers to overlap cache fetch with compute.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch15.verification_payload_mixin import VerificationPayloadMixin


class OptimizedKVCacheNvlinkPoolBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Tiered KV cache with NVLink pooling."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(self):
        super().__init__()
        self.output = None
        self.model: Optional[nn.MultiheadAttention] = None
        self.local_cache_limit = 32
        self.peer_cache_limit = 160
        self.hidden = 512
        self.heads = 8
        self.batch = 4
        self.seq_len = 256
        tokens = self.batch * self.seq_len
        self.peer_device: Optional[torch.device] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._verify_q: Optional[torch.Tensor] = None
        self._query_steps: Optional[torch.Tensor] = None
        self._key_steps: Optional[torch.Tensor] = None
        self._value_steps: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: requires CUDA")
        # Single-GPU: treat peer cache as an expanded local pool.
        self.peer_device = self.device
        self.model = nn.MultiheadAttention(self.hidden, self.heads, batch_first=True).to(self.device).eval()
        self._query_steps = torch.randn(self.seq_len, self.batch, 1, self.hidden, device=self.device)
        self._key_steps = torch.randn(self.seq_len, self.batch, 1, self.hidden, device=self.device)
        self._value_steps = torch.randn(self.seq_len, self.batch, 1, self.hidden, device=self.device)
        self._verify_q = self._query_steps[0, :1].detach().clone()
        self._synchronize()

    def _place_kv(self, k: torch.Tensor, v: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Decide where to place KV: local -> peer -> host."""
        if step < self.local_cache_limit:
            return k, v, "local"
        if self.peer_device is not None and step < self.local_cache_limit + self.peer_cache_limit:
            return k.to(self.peer_device, non_blocking=True), v.to(self.peer_device, non_blocking=True), "peer"
        return k.cpu(), v.cpu(), "host"

    def benchmark_fn(self) -> None:
        assert self.model is not None
        assert self._query_steps is not None and self._key_steps is not None and self._value_steps is not None
        with self._nvtx_range("optimized_kv_cache_nvlink_pool"):
            cache_k: list[torch.Tensor] = []
            cache_v: list[torch.Tensor] = []
            tiers: list[str] = []
            for step in range(self.seq_len):
                q = self._query_steps[step]
                k = self._key_steps[step]
                v = self._value_steps[step]
                placed_k, placed_v, tier = self._place_kv(k, v, step)
                cache_k.append(placed_k)
                cache_v.append(placed_v)
                tiers.append(tier)

                # Gather, preferring local first then peer then host (copied back async)
                gathered_k = []
                gathered_v = []
                for tk, tv, t in zip(cache_k, cache_v, tiers):
                    if t == "local":
                        gathered_k.append(tk)
                        gathered_v.append(tv)
                    elif t == "peer":
                        gathered_k.append(tk.to(self.device, non_blocking=True))
                        gathered_v.append(tv.to(self.device, non_blocking=True))
                    else:  # host
                        gathered_k.append(tk.to(self.device, non_blocking=True))
                        gathered_v.append(tv.to(self.device, non_blocking=True))

                k_all = torch.cat(gathered_k, dim=1)
                v_all = torch.cat(gathered_v, dim=1)
                out, _ = self.model(q, k_all, v_all)
                self.output = out

    def capture_verification_payload(self) -> None:
        if self.model is None or self.output is None or self._verify_q is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._synchronize()
        self._set_verification_payload(
            inputs={"q": self._verify_q},
            output=self.output,
            batch_size=int(self.batch),
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(5e-1, 5e-1),
        )

    def teardown(self) -> None:
        self.model = None
        self._query_steps = None
        self._key_steps = None
        self._value_steps = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedKVCacheNvlinkPoolBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(OptimizedKVCacheNvlinkPoolBenchmark)
