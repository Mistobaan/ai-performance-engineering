from pathlib import Path

from core.harness.run_benchmarks import _canonicalize_optimized_variants_for_full_sweep


def test_nvfp4_group_full_sweep_uses_canonical_default(monkeypatch):
    monkeypatch.delenv("AISP_INCLUDE_NVFP4_GROUP_GEMM_VARIANTS", raising=False)

    baseline = Path("/tmp/labs/nvfp4_group_gemm/baseline_nvfp4_group_gemm_case0.py")
    canonical = Path("/tmp/labs/nvfp4_group_gemm/optimized_nvfp4_group_gemm_case0.py")
    variant_a = Path("/tmp/labs/nvfp4_group_gemm/optimized_nvfp4_group_gemm_case0_cutlass2sm.py")
    variant_b = Path("/tmp/labs/nvfp4_group_gemm/optimized_nvfp4_group_gemm_case0_cutlass1sm_n128.py")
    pairs = [(baseline, [canonical, variant_a, variant_b], "nvfp4_group_gemm_case0")]

    filtered, suppressed = _canonicalize_optimized_variants_for_full_sweep(
        "labs/nvfp4_group_gemm",
        pairs,
        include_alias_pairs=False,
        example_filters=None,
    )

    assert suppressed == 2
    assert filtered == [(baseline, [canonical], "nvfp4_group_gemm_case0")]


def test_nvfp4_group_full_sweep_respects_env_override(monkeypatch):
    monkeypatch.setenv("AISP_INCLUDE_NVFP4_GROUP_GEMM_VARIANTS", "1")

    baseline = Path("/tmp/labs/nvfp4_group_gemm/baseline_nvfp4_group_gemm_case0.py")
    canonical = Path("/tmp/labs/nvfp4_group_gemm/optimized_nvfp4_group_gemm_case0.py")
    variant = Path("/tmp/labs/nvfp4_group_gemm/optimized_nvfp4_group_gemm_case0_cutlass2sm.py")
    pairs = [(baseline, [canonical, variant], "nvfp4_group_gemm_case0")]

    filtered, suppressed = _canonicalize_optimized_variants_for_full_sweep(
        "labs/nvfp4_group_gemm",
        pairs,
        include_alias_pairs=False,
        example_filters=None,
    )

    assert suppressed == 0
    assert filtered == pairs


def test_variant_filter_skips_for_alias_targets(monkeypatch):
    monkeypatch.delenv("AISP_INCLUDE_NVFP4_GROUP_GEMM_VARIANTS", raising=False)

    baseline = Path("/tmp/labs/nvfp4_group_gemm/baseline_nvfp4_group_gemm_case0.py")
    canonical = Path("/tmp/labs/nvfp4_group_gemm/optimized_nvfp4_group_gemm_case0.py")
    variant = Path("/tmp/labs/nvfp4_group_gemm/optimized_nvfp4_group_gemm_case0_cutlass2sm.py")
    pairs = [(baseline, [canonical, variant], "nvfp4_group_gemm_case0")]

    filtered_alias, suppressed_alias = _canonicalize_optimized_variants_for_full_sweep(
        "labs/nvfp4_group_gemm",
        pairs,
        include_alias_pairs=True,
        example_filters=None,
    )
    assert suppressed_alias == 0
    assert filtered_alias == pairs


def test_variant_filter_applies_for_canonical_example_filter(monkeypatch):
    monkeypatch.delenv("AISP_INCLUDE_NVFP4_GROUP_GEMM_VARIANTS", raising=False)

    baseline = Path("/tmp/labs/nvfp4_group_gemm/baseline_nvfp4_group_gemm_case0.py")
    canonical = Path("/tmp/labs/nvfp4_group_gemm/optimized_nvfp4_group_gemm_case0.py")
    variant = Path("/tmp/labs/nvfp4_group_gemm/optimized_nvfp4_group_gemm_case0_cutlass2sm.py")
    pairs = [(baseline, [canonical, variant], "nvfp4_group_gemm_case0")]

    filtered_examples, suppressed_examples = _canonicalize_optimized_variants_for_full_sweep(
        "labs/nvfp4_group_gemm",
        pairs,
        include_alias_pairs=False,
        example_filters={"nvfp4_group_gemm_case0"},
    )
    assert suppressed_examples == 1
    assert filtered_examples == [(baseline, [canonical], "nvfp4_group_gemm_case0")]


def test_non_nvfp4_group_chapters_are_unchanged(monkeypatch):
    monkeypatch.delenv("AISP_INCLUDE_NVFP4_GROUP_GEMM_VARIANTS", raising=False)

    baseline = Path("/tmp/ch10/baseline_matmul.py")
    canonical = Path("/tmp/ch10/optimized_matmul.py")
    variant = Path("/tmp/ch10/optimized_matmul_tcgen05.py")
    pairs = [(baseline, [canonical, variant], "matmul")]

    filtered, suppressed = _canonicalize_optimized_variants_for_full_sweep(
        "ch10",
        pairs,
        include_alias_pairs=False,
        example_filters=None,
    )

    assert suppressed == 0
    assert filtered == pairs
