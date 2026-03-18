"""Baseline for FP8 decode optimization: BF16 prefill-only workload.

This baseline matches `optimized_decode_fp8.py` exactly, but runs in BF16.
Keeping a dedicated baseline ensures the FP8 comparison is workload-equivalent.
"""

from __future__ import annotations

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata


def get_benchmark() -> DecodeBenchmark:
    # Prefill-dominant regime where FP8 benefits are visible.
    cfg = DecodeConfig(
        batch_size=128,
        prompt_tokens=1024,
        decode_tokens=0,
        hidden_size=8192,
        use_pinned_host=False,
        use_copy_stream=False,
        use_compute_stream=False,
        use_cuda_graphs=False,
        use_torch_compile=False,
        label="baseline_decode_fp8",
        iterations=12,
        warmup=15,
    )
    bench = attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)
    bench.signature_equivalence_group = "labs_decode_fp8_precision"
    bench.signature_equivalence_ignore_fields = ("precision_flags",)
    bench.story_metadata = {
        "pair_role": "canonical",
        "variant_role": "baseline",
        "chapter_alignment": "native",
        "chapter_native_exemplar": True,
        "comparison_axis": "bf16_vs_fp8_transformer_engine",
        "execution_pattern": "prefill_only_decode",
        "control_reason": (
            "This baseline keeps the FP8 workload fixed while using BF16 nn.Linear "
            "modules as the non-Transformer-Engine reference path."
        ),
    }
    return bench

