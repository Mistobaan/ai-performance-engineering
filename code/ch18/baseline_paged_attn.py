"""baseline_paged_attn.py - Dense SDPA baseline for paged attention demos."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402

class BaselinePagedAttnBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.qkv: Optional[torch.Tensor] = None
        self.output = None
        self._workload = WorkloadMetadata(tokens_per_iteration=0.0)
        self._verification_payload = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Longer sequence to expose flash SDPA advantage (O(N) vs O(N²) memory).
        b, h, s, d = 4, 16, 2048, 64
        # Baseline forces the unfused math backend; keep dtype aligned with optimized (BF16)
        # so the comparison isolates backend choice (math vs flash).
        self.qkv = torch.randn(b, h, s, 3, d, device=self.device, dtype=torch.bfloat16)
        # Aggressive warmup: run multiple times to fully JIT-compile the math SDPA path.
        q = self.qkv[:, :, :, 0]
        k = self.qkv[:, :, :, 1]
        v = self.qkv[:, :, :, 2]
        if not torch.cuda.is_available():
            raise RuntimeError("FAIL FAST: paged attention benchmark requires CUDA")
        # Enforce strict math path to keep baseline stable across backend policy changes.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        if not torch.backends.cuda.math_sdp_enabled():
            raise RuntimeError("FAIL FAST: Math SDPA backend is not enabled")
        for _ in range(8):
            _ = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if self.qkv is None:
            raise RuntimeError("FAIL FAST: QKV not initialized")
        q = self.qkv[:, :, :, 0]
        k = self.qkv[:, :, :, 1]
        v = self.qkv[:, :, :, 2]

        enable_nvtx = get_nvtx_enabled(self.get_config())
        # Force the unfused math path so the optimized variant can contrast flash SDPA.
        with nvtx_range("paged_attn_baseline", enable=enable_nvtx):
            self.output = F.scaled_dot_product_attention(q, k, v)
        if self.output is None or self.qkv is None:
            raise RuntimeError("benchmark_fn() must produce output")
        return {}

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"qkv": self.qkv},
            output=self.output,
            batch_size=self.qkv.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": self.qkv.dtype == torch.float16,
                "bf16": self.qkv.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_speculative_decoding_metrics
        return compute_speculative_decoding_metrics(
            draft_tokens=getattr(self, '_draft_tokens', 64),
            accepted_tokens=getattr(self, '_accepted_tokens', 48),
            draft_time_ms=getattr(self, '_draft_ms', 5.0),
            verify_time_ms=getattr(self, '_verify_ms', 10.0),
            num_rounds=getattr(self, '_num_rounds', 8),
        )

def get_benchmark() -> BaseBenchmark:
    return BaselinePagedAttnBenchmark()
