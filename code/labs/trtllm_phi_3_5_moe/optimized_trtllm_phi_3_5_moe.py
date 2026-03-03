"""Optimized TensorRT-LLM generation for Phi-3.5-MoE."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.trtllm_phi_3_5_moe.trtllm_common import (
    build_prompt_tokens,
    ensure_trtllm_assets,
    load_trtllm_runtime,
    parse_trtllm_args,
    resolve_model_path,
    slice_logits,
)


class OptimizedTrtLlmPhi35MoeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """TensorRT-LLM optimized generation for Phi-3.5-MoE."""

    def __init__(self) -> None:
        super().__init__()
        args = parse_trtllm_args()
        self.model_path = Path(args.model_path)
        self.engine_path = Path(args.engine_path) if args.engine_path is not None else None
        self.prompt_len = args.prompt_len
        self.max_new_tokens = args.max_new_tokens
        self.batch_size = args.batch_size
        self.vocab_slice = args.vocab_slice
        self.runner = None
        self.tokenizer = None
        self.sampling_config = None
        self.input_ids: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        tokens = float(self.prompt_len + self.max_new_tokens)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=tokens,
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=tokens,
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA is required for the TRT-LLM Phi-3.5-MoE benchmark")
        self.model_path = resolve_model_path(self.model_path)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("SKIPPED: transformers is required for tokenizer support") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        input_ids, attention_mask = build_prompt_tokens(
            self.tokenizer,
            prompt_len=self.prompt_len,
            batch_size=self.batch_size,
        )
        self.input_ids = input_ids.to(self.device)
        self.attention_mask = attention_mask.to(self.device)
        ensure_trtllm_assets(
            self.model_path,
            engine_path=self.engine_path,
            require_engine=True,
        )
        runtime = load_trtllm_runtime()
        ModelRunner = runtime.ModelRunner
        SamplingConfig = runtime.SamplingConfig
        if self.engine_path is None:
            raise RuntimeError("SKIPPED: TensorRT-LLM engine_path is required")
        if self.engine_path.is_dir():
            if not hasattr(ModelRunner, "from_dir"):
                raise RuntimeError("SKIPPED: ModelRunner.from_dir is unavailable; provide an engine file path")
            self.runner = ModelRunner.from_dir(str(self.engine_path))
        else:
            self.runner = ModelRunner.from_engine(str(self.engine_path))
        self.sampling_config = SamplingConfig(
            end_id=int(self.tokenizer.eos_token_id),
            pad_id=int(self.tokenizer.pad_token_id),
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            return_dict=True,
            top_k=1,
            top_p=0.0,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.input_ids is None:
            raise RuntimeError("Benchmark not initialized")
        if self.runner is None or self.sampling_config is None:
            raise RuntimeError("TensorRT-LLM backend not initialized")
        with self._nvtx_range("optimized_trtllm_phi_3_5_moe"):
            if self.attention_mask is None:
                raise RuntimeError("Attention mask not initialized")
            batch_inputs = []
            for i in range(self.batch_size):
                valid_len = int(self.attention_mask[i].sum().item())
                batch_inputs.append(self.input_ids[i, :valid_len].contiguous())
            outputs = self.runner.generate(
                batch_inputs,
                sampling_config=self.sampling_config,
                output_generation_logits=True,
            )
            if not isinstance(outputs, dict) or "generation_logits" not in outputs:
                raise RuntimeError("TensorRT-LLM generate must return generation_logits when requested")
            logits = self._normalize_generation_logits(outputs["generation_logits"])
            self.output = slice_logits(logits, self.vocab_slice).float()
        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    @staticmethod
    def _normalize_generation_logits(generation_logits) -> torch.Tensor:
        """Normalize TRT-LLM generation logits into [batch, vocab] for verification parity."""
        if isinstance(generation_logits, (list, tuple)):
            if len(generation_logits) == 0:
                raise RuntimeError("TensorRT-LLM generate returned empty generation_logits sequence")
            generation_logits = generation_logits[0]

        if not isinstance(generation_logits, torch.Tensor):
            raise RuntimeError(
                "Unsupported generation_logits type "
                f"{type(generation_logits).__name__}; expected Tensor or sequence of Tensor."
            )

        logits = generation_logits
        # TRT-LLM may return [batch, beam, step, vocab] or [batch, step, vocab].
        while logits.dim() > 2:
            logits = logits.select(1, 0)

        if logits.dim() != 2:
            raise RuntimeError(
                "TensorRT-LLM generation_logits could not be normalized to [batch, vocab]; "
                f"got shape={tuple(generation_logits.shape)}"
            )
        return logits

    def capture_verification_payload(self) -> None:
        if self.output is None or self.input_ids is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        verify_output = self.output[:1, :128]
        # Keep signature fields backend-agnostic so baseline Transformers and optimized
        # TRT-LLM engine runs compare on equivalent workload semantics.
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=verify_output.detach().clone(),
            batch_size=int(self.batch_size),
            parameter_count=0,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(5e-2, 5e-2),
        )

    def teardown(self) -> None:
        self.runner = None
        self.tokenizer = None
        self.sampling_config = None
        self.input_ids = None
        self.attention_mask = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=2)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedTrtLlmPhi35MoeBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
