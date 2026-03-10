from __future__ import annotations

from core.scripts.audit_book_code_alignment import (
    REGISTRY_PATH,
    chapter_dir,
    load_registry,
    resolve_label,
)


def test_registry_entries_resolve_to_live_code_paths() -> None:
    registry = load_registry(REGISTRY_PATH)
    assert registry

    for (chapter, label), entry in registry.items():
        result = resolve_label(chapter, label, chapter_dir(chapter), registry)
        assert result.status == "ok", f"{label}: {result.notes}"
        assert result.classification == entry.classification
        assert result.canonical_path == entry.canonical_path


def test_generic_resolution_keeps_legacy_book_labels_mapped() -> None:
    registry = load_registry(REGISTRY_PATH)

    expectations = {
        (4, "before_no_overlap.py"): "ch04/baseline_no_overlap.py",
        (8, "threshold_naive.cu"): "ch08/baseline_threshold.cu",
    }

    for (chapter, label), expected_path in expectations.items():
        result = resolve_label(chapter, label, chapter_dir(chapter), registry)
        assert result.status == "ok", f"{label}: {result.notes}"
        assert result.canonical_path == expected_path
