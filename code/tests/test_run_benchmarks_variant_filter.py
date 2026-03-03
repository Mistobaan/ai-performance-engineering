from pathlib import Path

from core.harness.run_benchmarks import _canonicalize_optimized_variants_for_full_sweep


def test_full_sweep_prefers_canonical_optimized_when_present():
    baseline = Path("/tmp/ch10/baseline_matmul.py")
    canonical = Path("/tmp/ch10/optimized_matmul.py")
    variant_a = Path("/tmp/ch10/optimized_matmul_tc.py")
    variant_b = Path("/tmp/ch10/optimized_matmul_splitk.py")
    pairs = [(baseline, [canonical, variant_a, variant_b], "matmul")]

    filtered, suppressed = _canonicalize_optimized_variants_for_full_sweep(
        pairs,
    )

    assert suppressed == 2
    assert filtered == [(baseline, [canonical], "matmul")]


def test_full_sweep_keeps_all_variants_when_no_canonical_optimized_exists():
    baseline = Path("/tmp/ch10/baseline_matmul.py")
    variant_a = Path("/tmp/ch10/optimized_matmul_tc.py")
    variant_b = Path("/tmp/ch10/optimized_matmul_splitk.py")
    pairs = [(baseline, [variant_a, variant_b], "matmul")]

    filtered, suppressed = _canonicalize_optimized_variants_for_full_sweep(
        pairs,
    )

    assert suppressed == 0
    assert filtered == pairs


def test_full_sweep_keeps_alias_entries_unchanged():
    baseline = Path("/tmp/ch10/baseline_matmul.py")
    canonical = Path("/tmp/ch10/optimized_matmul.py")
    variant = Path("/tmp/ch10/optimized_matmul_tc.py")
    pairs = [(baseline, [canonical, variant], "matmul_tc")]

    filtered_alias, suppressed_alias = _canonicalize_optimized_variants_for_full_sweep(
        pairs,
    )
    assert suppressed_alias == 0
    assert filtered_alias == pairs


def test_non_lab_chapters_use_same_generic_canonicalization():
    baseline = Path("/tmp/ch10/baseline_matmul.py")
    canonical = Path("/tmp/ch10/optimized_matmul.py")
    variant = Path("/tmp/ch10/optimized_matmul_tc.py")
    pairs = [(baseline, [canonical, variant], "matmul")]

    filtered_examples, suppressed_examples = _canonicalize_optimized_variants_for_full_sweep(
        pairs,
    )
    assert suppressed_examples == 1
    assert filtered_examples == [(baseline, [canonical], "matmul")]


def test_chapter_without_canonical_is_unchanged():
    baseline = Path("/tmp/ch10/baseline_matmul.py")
    variant = Path("/tmp/ch10/optimized_matmul_tcgen05.py")
    pairs = [(baseline, [variant], "matmul")]

    filtered, suppressed = _canonicalize_optimized_variants_for_full_sweep(
        pairs,
    )

    assert suppressed == 0
    assert filtered == pairs
