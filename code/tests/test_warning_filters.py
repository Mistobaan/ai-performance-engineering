from __future__ import annotations

from pathlib import Path
import warnings

from core.utils.warning_filters import suppress_known_cuda_capability_warnings


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_warning_filter_context_does_not_leak_global_filters() -> None:
    before = list(warnings.filters)
    with suppress_known_cuda_capability_warnings():
        inside = list(warnings.filters)
        assert len(inside) >= len(before)
    after = list(warnings.filters)
    assert after == before


def test_no_import_time_global_warning_filters_in_runtime_modules() -> None:
    paths = [
        REPO_ROOT / "core" / "harness" / "arch_config.py",
        REPO_ROOT / "core" / "utils" / "chapter_compare_template.py",
        REPO_ROOT / "core" / "benchmark" / "bench_commands.py",
        REPO_ROOT / "core" / "benchmark" / "benchmark_peak.py",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "warnings.filterwarnings(" not in text, f"global warning filter leaked back into {path}"
