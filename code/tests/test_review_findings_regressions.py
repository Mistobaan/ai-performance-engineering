from __future__ import annotations

from pathlib import Path

import torch

from ch08.tcgen05_custom_vs_cublas_benchmark_base import Tcgen05CustomVsCublasBase
from ch08.threshold_tma_benchmark_base import ThresholdBenchmarkBaseTMA
from ch08.tiling_benchmark_base import TilingBenchmarkBase
from core.harness.run_benchmarks import INFORMATIONAL_BENCHMARKS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _setup_section(rel_path: str) -> str:
    text = _read(rel_path)
    return text.split("def benchmark_fn", 1)[0]


def _benchmark_section(rel_path: str) -> str:
    text = _read(rel_path)
    return text.split("def benchmark_fn", 1)[1].split("def capture_verification_payload", 1)[0]


def test_ch02_cublas_setup_keeps_warmup_symmetric() -> None:
    baseline_setup = _setup_section("ch02/baseline_cublas.py")
    optimized_setup = _setup_section("ch02/optimized_cublas.py")

    for setup_text in (baseline_setup, optimized_setup):
        assert "for _ in range(10):" in setup_text
        assert "_ = torch.matmul(self.A, self.B)" in setup_text

    assert "warm cuBLAS identically" in baseline_setup
    assert "warmed-up heuristics" in optimized_setup


def test_ch06_attention_ilp_pair_keeps_math_fixed_and_only_changes_ilp_schedule() -> None:
    baseline_text = _read("ch06/baseline_attention_ilp.py")
    optimized_text = _read("ch06/optimized_attention_ilp.py")
    readme_text = _read("ch06/README.md")

    for source in (baseline_text, optimized_text):
        assert "load_ilp_extension" in source
        assert "self.attention_terms = (query * key * 0.1).contiguous().reshape(-1)" in source
        assert "WORKLOAD" in source
        assert "MultiheadAttention" not in source
        assert "scaled_dot_product_attention" not in source
        assert "torch.cuda.Stream" not in source

    assert "self._extension.sequential_ops(dst, src)" in baseline_text
    assert "self._extension.unrolled_ilp(dst, src)" in optimized_text
    assert '"attention_ilp.independent_chains_per_thread": 1.0' in baseline_text
    assert '"attention_ilp.independent_chains_per_thread": 4.0' in optimized_text
    assert "keep the math fixed while changing independent chains per thread" in readme_text


def test_ch06_optimized_adaptive_uses_chunk_plan_without_extra_staging_buffers() -> None:
    optimized_text = _read("ch06/optimized_adaptive.py")

    assert "self.chunk_plan: list[tuple[int, int]] = []" in optimized_text
    assert "self._output_buffer = torch.empty_like(self.input)" in optimized_text
    assert "for start, end in self.chunk_plan:" in optimized_text
    assert "window = self.input[start:end]" in optimized_text
    assert "self._output_buffer[start:end].copy_(transformed)" in optimized_text

    for forbidden in ("host_buffer", "device_buffer", "pin_memory", "torch.cuda.Stream"):
        assert forbidden not in optimized_text


def test_ch08_bridge_control_pairs_are_explicitly_marked_in_structured_metrics() -> None:
    threshold = object.__new__(ThresholdBenchmarkBaseTMA)
    threshold.rows = ThresholdBenchmarkBaseTMA.rows
    threshold.threshold = ThresholdBenchmarkBaseTMA.threshold
    threshold.inner_iterations = ThresholdBenchmarkBaseTMA.inner_iterations
    threshold_metrics = threshold.get_custom_metrics()
    assert threshold_metrics["story.control_pair"] == 1.0
    assert threshold_metrics["story.chapter_native_exemplar"] == 0.0
    assert threshold_metrics["story.bridge_to_ch10"] == 1.0

    tiling = object.__new__(TilingBenchmarkBase)
    tiling.matrix_rows = TilingBenchmarkBase.matrix_rows
    tiling.shared_dim = TilingBenchmarkBase.shared_dim
    tiling.matrix_cols = TilingBenchmarkBase.matrix_cols
    tiling.inner_iterations = TilingBenchmarkBase.inner_iterations
    tiling.nvtx_label = "tiling"
    tiling_metrics = tiling.get_custom_metrics()
    assert tiling_metrics["story.control_pair"] == 1.0
    assert tiling_metrics["story.chapter_native_exemplar"] == 0.0
    assert tiling_metrics["story.bridge_to_ch09"] == 1.0

    tcgen05 = object.__new__(Tcgen05CustomVsCublasBase)
    tcgen05.matrix_rows = Tcgen05CustomVsCublasBase.matrix_rows
    tcgen05.shared_dim = Tcgen05CustomVsCublasBase.shared_dim
    tcgen05.matrix_cols = Tcgen05CustomVsCublasBase.matrix_cols
    tcgen05_metrics = tcgen05.get_custom_metrics()
    assert tcgen05_metrics["story.control_pair"] == 1.0
    assert tcgen05_metrics["story.chapter_native_exemplar"] == 0.0
    assert tcgen05_metrics["story.bridge_to_ch09"] == 1.0

    baseline_nvfp4_text = _read("ch08/baseline_nvfp4_mlp.py")
    optimized_nvfp4_text = _read("ch08/optimized_nvfp4_mlp.py")
    for source in (baseline_nvfp4_text, optimized_nvfp4_text):
        assert '"story.control_pair": 1.0' in source
        assert '"story.chapter_native_exemplar": 0.0' in source
        assert '"story.bridge_to_ch09": 1.0' in source


def test_ch08_readme_calls_out_bridge_controls_and_historical_tcgen05_naming() -> None:
    readme_text = _read("ch08/README.md")
    baseline_tcgen05_text = _read("ch08/baseline_tcgen05_custom_vs_cublas.py")
    optimized_tcgen05_text = _read("ch08/optimized_tcgen05_custom_vs_cublas.py")

    assert "chapter-native exemplars" in readme_text
    assert "`thresholdtma`, `tiling`, `tiling_tcgen05`, `tcgen05_custom_vs_cublas`, and `nvfp4_mlp`" in readme_text
    assert "custom-versus-library comparison target" in readme_text
    assert "historical baseline/optimized filenames" not in readme_text
    assert "custom tcgen05 versus cuBLAS comparison" in baseline_tcgen05_text
    assert "Custom tcgen05 kernel side of the comparison pair." in baseline_tcgen05_text
    assert "Vendor cuBLAS reference side of the comparison pair." in optimized_tcgen05_text


def test_ch15_split_moe_targets_isolate_dispatch_from_routing() -> None:
    dispatch_baseline = _read("ch15/baseline_moe_dispatch.py")
    dispatch_optimized = _read("ch15/optimized_moe_dispatch.py")
    routing_baseline = _read("ch15/baseline_moe_routing_topology_aware.py")
    routing_optimized = _read("ch15/optimized_moe_routing_topology_aware.py")
    readme_text = _read("ch15/README.md")

    assert 'route_mode = "uniform"' in dispatch_baseline
    assert 'route_mode = "uniform"' in dispatch_optimized
    assert 'dispatch_mode = "mask_scan"' in dispatch_baseline
    assert 'dispatch_mode = "active_experts"' in dispatch_optimized
    assert 'dispatch_mode = "mask_scan"' in routing_baseline
    assert 'dispatch_mode = "mask_scan"' in routing_optimized
    assert 'route_mode = "uniform"' in routing_baseline
    assert 'route_mode = "topology_aware"' in routing_optimized
    assert "baseline_moe_routing_simple.py" not in readme_text
    assert "baseline_moe_dispatch.py" in readme_text
    assert "baseline_moe_routing_topology_aware.py" in readme_text


def test_ch18_split_paged_attention_targets_isolate_backend_from_layout() -> None:
    common_text = _read("ch18/paged_attn_split_common.py")
    readme_text = _read("ch18/README.md")

    assert "class DensePagedAttnBase" in common_text
    assert "class LayoutPagedAttnBase" in common_text
    assert 'metrics["paged_attn.backend_math"] = 1.0 if self.backend == "math" else 0.0' in common_text
    assert "def _build_block_table" in common_text
    assert "return torch.stack(" in common_text
    assert "return create_block_mask(" in common_text
    assert "dense masked decode versus block-table-driven FlexAttention sparse kernels" in readme_text
    assert "fused FlexAttention block-mask kernel" in readme_text
    assert "baseline_paged_attn_backend.py" in readme_text
    assert "baseline_paged_attn_layout.py" in readme_text
    assert "optimized_paged_attn_vllm.py" not in readme_text


def test_ch16_blackwell_paged_attention_variant_is_intentional() -> None:
    readme_text = _read("ch16/README.md")
    source = _read("ch16/optimized_paged_attention_blackwell.py")

    assert "paged_attention_blackwell" in readme_text
    assert "intentional optimized variant" in readme_text
    assert "story_metadata" in source
    assert '"variant"' in source
    assert "paged_attention_blackwell" not in INFORMATIONAL_BENCHMARKS["ch16"]


def test_reviewed_pair_fixes_remain_applied() -> None:
    baseline_regional = _read("ch14/baseline_regional_triton.py")
    baseline_regional_setup = _setup_section("ch14/baseline_regional_triton.py")
    baseline_regional_bench = _benchmark_section("ch14/baseline_regional_triton.py")
    sliding_window = _read("ch14/optimized_sliding_window.py")
    blackwell = _read("ch16/optimized_paged_attention_blackwell.py")
    baseline_memory = _read("ch17/baseline_memory.py")
    optimized_memory = _read("ch17/optimized_memory.py")
    fp4_baseline = _read("ch19/baseline_fp4_weight_quantization.py")
    baseline_kv = _read("ch20/baseline_integrated_kv_cache.py")
    optimized_kv = _read("ch20/optimized_integrated_kv_cache.py")
    optimized_memory_standard = _read("ch20/optimized_memory_standard.py")
    baseline_memory_standard = _read("ch20/baseline_memory_standard.py")
    baseline_pipeline_bench = _benchmark_section("ch20/baseline_pipeline_sequential.py")

    assert "self._compiled_model = torch.compile(" in baseline_regional_setup
    assert "torch.compile(" not in baseline_regional_bench
    assert "allowed_benchmark_fn_antipatterns" not in baseline_regional

    assert "iterations=20" in sliding_window
    assert "warmup=5" in sliding_window
    assert "full causal SDPA" in sliding_window
    assert "historical" in sliding_window

    assert "fp8_kv" not in blackwell
    assert "FP8 KV cache benefits" not in blackwell
    assert '"fp8": False' in blackwell

    for source in (baseline_memory, optimized_memory):
        assert "iterations=10" in source
        assert "warmup=5" in source

    assert "_ = weight.sum()" not in fp4_baseline

    for source in (baseline_kv, optimized_kv):
        assert "for pos in range(seq_len):" in source
        assert "token = x[:, pos:pos+1, :]" in source
    assert "range(0, seq_len, 8)" not in optimized_kv

    assert "class OptimizedMemoryStandardBenchmark" in optimized_memory_standard
    assert "OptimizedMemoryHBM3eBenchmark" not in optimized_memory_standard
    assert "HBM3e" not in baseline_memory_standard

    assert "with torch.no_grad():" in baseline_pipeline_bench


def test_ch13_pair_remediations_keep_canonical_and_informational_targets_split() -> None:
    canonical_quant = _read("ch13/optimized_torchao_quantization.py")
    compiled_quant = _read("ch13/optimized_torchao_quantization_compiled.py")
    canonical_kv = _read("ch13/optimized_kv_cache_naive.py")
    flash_kv = _read("ch13/optimized_kv_cache_naive_flash_blockwise.py")

    assert "torch.compile(self.model" not in canonical_quant
    assert "torch.compile(self.model" in compiled_quant
    assert "torchao_quantization_compiled" in INFORMATIONAL_BENCHMARKS["ch13"]

    assert "for pos in range(seq_len):" in canonical_kv
    assert "range(0, seq_len, self.block_size)" not in canonical_kv
    assert "range(0, seq_len, self.block_size)" in flash_kv
    assert "kv_cache_naive_flash_blockwise" in INFORMATIONAL_BENCHMARKS["ch13"]


def test_ch18_and_fullstack_pairs_keep_semantics_fixed() -> None:
    baseline_flexdecode = _read("ch18/baseline_flexdecoding.py")
    optimized_flexdecode = _read("ch18/optimized_flexdecoding.py")
    moe_common = _read("labs/fullstack_cluster/moe_hybrid_ep_common.py")

    assert '"comparison_axis": "full_kv_mask_vs_windowed_kv_slice"' in baseline_flexdecode
    assert '"execution_pattern": "masked_full_cache_decode"' in baseline_flexdecode
    assert "self.model.decode(token, position)" not in optimized_flexdecode
    assert "full KV cache with a sliding-window mask" in baseline_flexdecode
    assert "window_slice_decode" in optimized_flexdecode
    assert "k_slice = self.model.k_cache[:, start:end]" not in baseline_flexdecode
    assert "k_slice = self.model.k_cache[:, start:end]" in optimized_flexdecode
    assert 'route_mode="uniform"' in moe_common
    assert 'route_mode="topology_aware" if optimized else "uniform"' not in moe_common


def test_ch17_memory_pair_keeps_discrete_input_distribution() -> None:
    baseline_memory = _read("ch17/baseline_memory.py")
    optimized_memory = _read("ch17/optimized_memory.py")

    assert "torch.randint(" in baseline_memory
    assert "256," in baseline_memory
    assert "dtype=torch.uint8" in baseline_memory
    assert "random_(0, 256).floor_()" in optimized_memory
    assert "discrete 0..255 population" in optimized_memory

    sample = torch.empty(4096, dtype=torch.float32)
    sample.random_(0, 256).floor_()
    assert torch.equal(sample, sample.floor())
    assert float(sample.min().item()) >= 0.0
    assert float(sample.max().item()) <= 255.0


def test_ch20_multiple_unoptimized_no_longer_claims_fused_ops() -> None:
    source = _read("ch20/optimized_multiple_unoptimized.py")

    assert "does not implement a fused MLP kernel today" in source
    assert '"ch20.uses_fused_ops": 0.0' in source
