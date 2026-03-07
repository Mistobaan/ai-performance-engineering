"""Baseline monolithic prefill+decode benchmark (native to Chapter 17)."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.cuda_event_timing import elapsed_ms, elapsed_ms_list
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class SimpleLLM(nn.Module):
    """Simplified LLM used for monolithic prefill+decode."""

    def __init__(self, hidden_dim: int = 1024, num_layers: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def prefill(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """Prefill over the full prompt (compute-bound path)."""
        x = torch.randn(
            prompt_tokens.size(0),
            prompt_tokens.size(1),
            self.hidden_dim,
            device=prompt_tokens.device,
            dtype=torch.bfloat16,
        )
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x[:, -1:, :]

    def decode(self, kv_cache: torch.Tensor, num_tokens: int = 16) -> torch.Tensor:
        """Decode a small number of tokens (memory-bound path)."""
        outputs = []
        x = kv_cache
        for _ in range(num_tokens):
            for layer in self.layers:
                x = torch.relu(layer(x))
            outputs.append(x)
        return torch.cat(outputs, dim=1)


class BaselinePrefillDecodeMonolithicBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Monolithic prefill+decode baseline (no disaggregation)."""

    def __init__(self):
        super().__init__()
        self.model: Optional[SimpleLLM] = None
        self.prompt: Optional[torch.Tensor] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self._history: Dict[str, List[float]] = {"ttft": [], "tpot": []}
        # Workload dimensions for signature matching
        self.batch_size = 1
        self.prefill_seq = 256
        self.decode_seq = 16
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=self.prefill_seq + self.decode_seq,
        )
        self.output: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self._verification_payload = None
        self._pending_ttft_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._pending_tpot_pairs: List[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = SimpleLLM(hidden_dim=1024, num_layers=12).to(self.device).to(torch.bfloat16).eval()
        self.prompt = torch.randint(0, 10000, (1, 256), device=self.device)
        with torch.no_grad():
            self.kv_cache = self.model.prefill(self.prompt)
        torch.cuda.synchronize(self.device)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if self.model is None or self.prompt is None:
            raise RuntimeError("Model or prompt not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())

        with nvtx_range("inference_monolithic", enable=enable_nvtx):
            with torch.no_grad():
                request_start = torch.cuda.Event(enable_timing=True)
                prefill_end = torch.cuda.Event(enable_timing=True)
                request_start.record()
                kv_cache = self.model.prefill(self.prompt)
                prefill_end.record()

                num_tokens = 16
                token_event_pairs: List[tuple[torch.cuda.Event, torch.cuda.Event]] = []
                token_output = kv_cache
                for _ in range(num_tokens):
                    token_start = torch.cuda.Event(enable_timing=True)
                    token_end = torch.cuda.Event(enable_timing=True)
                    token_start.record()
                    token_output = self.model.decode(token_output[:, -1:, :], num_tokens=1)
                    token_end.record()
                    token_event_pairs.append((token_start, token_end))
                self.output = token_output

                self._pending_ttft_pair = (request_start, prefill_end)
                self._pending_tpot_pairs = token_event_pairs
                return {}

    def finalize_iteration_metrics(self) -> Optional[Dict[str, List[float]]]:
        if self._pending_ttft_pair is None:
            return None
        ttft_ms = elapsed_ms(self._pending_ttft_pair)
        tpot_times_ms = elapsed_ms_list(self._pending_tpot_pairs)
        self._pending_ttft_pair = None
        self._pending_tpot_pairs = []
        self._history["ttft"].append(ttft_ms)
        self._history["tpot"].extend(tpot_times_ms)
        return {
            "ttft_times_ms": [ttft_ms],
            "tpot_times_ms": tpot_times_ms,
        }

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        if self.prompt is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        dtype = self.output.dtype
        self._set_verification_payload(
            inputs={"prompt": self.prompt},
            output=self.output,
            batch_size=int(self.batch_size),
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.prompt = None
        self.kv_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        self.finalize_iteration_metrics()
        if not self._history["ttft"]:
            return None
        return {
            "monolithic.ttft_ms": float(sum(self._history["ttft"]) / len(self._history["ttft"])),
            "monolithic.tpot_mean_ms": float(sum(self._history["tpot"]) / len(self._history["tpot"])),
        }

    def validate_result(self) -> Optional[str]:
        self.finalize_iteration_metrics()
        if not self._history["ttft"]:
            return "No TTFT samples recorded"
        if not self._history["tpot"]:
            return "No TPOT samples recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery (Chapter 17 baseline)."""
    return BaselinePrefillDecodeMonolithicBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
