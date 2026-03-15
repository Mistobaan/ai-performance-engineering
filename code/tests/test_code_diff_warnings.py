import json
from pathlib import Path

from core.code_diff import find_code_pair
from core.analysis import llm_profile_analyzer


def test_find_code_pair_surfaces_read_failures(tmp_path: Path, monkeypatch) -> None:
    chapter_dir = tmp_path / "ch_demo"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = chapter_dir / "baseline_demo.py"
    optimized_path = chapter_dir / "optimized_demo.py"
    baseline_path.write_text("baseline", encoding="utf-8")
    optimized_path.write_text("optimized", encoding="utf-8")

    original_read_text = Path.read_text

    def _patched_read_text(self: Path, *args, **kwargs):  # type: ignore[override]
        if self == baseline_path:
            raise OSError("boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _patched_read_text)

    result = find_code_pair(chapter_dir, "demo")

    assert result["baseline_code"] is None
    assert result["optimized_code"] == "optimized"
    assert result["optimized_path"] == str(optimized_path)
    assert result["warnings"]
    assert str(baseline_path) in result["warnings"][0]


def test_analyze_with_llm_preserves_source_read_warnings(tmp_path: Path, monkeypatch) -> None:
    differential_path = tmp_path / "differential_demo.json"
    baseline_path = tmp_path / "baseline_demo.py"
    optimized_path = tmp_path / "optimized_demo.py"
    output_path = tmp_path / "analysis.md"

    differential_path.write_text(json.dumps({"overall_speedup": 1.2}), encoding="utf-8")
    baseline_path.write_text("baseline", encoding="utf-8")
    optimized_path.write_text("optimized", encoding="utf-8")

    original_read_text = Path.read_text

    def _patched_read_text(self: Path, *args, **kwargs):  # type: ignore[override]
        if self == optimized_path:
            raise OSError("optimized boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _patched_read_text)

    class _FakeAnalyzer:
        def __init__(self, provider=None):
            self.provider = provider

        def analyze_differential(self, diff_report, baseline_code=None, optimized_code=None):
            assert diff_report["overall_speedup"] == 1.2
            assert baseline_code == "baseline"
            assert optimized_code is None
            return llm_profile_analyzer.LLMAnalysisResult(
                provider="fake",
                model="fake-model",
                why_faster="Because.",
            )

    monkeypatch.setattr(llm_profile_analyzer, "LLMProfileAnalyzer", _FakeAnalyzer)

    result = llm_profile_analyzer.analyze_with_llm(
        differential_path,
        baseline_code_path=baseline_path,
        optimized_code_path=optimized_path,
        output_path=output_path,
    )

    assert result is not None
    assert result.warnings
    assert str(optimized_path) in result.warnings[0]
    markdown = output_path.read_text(encoding="utf-8")
    assert "## Analysis Warnings" in markdown
    assert "optimized boom" in markdown
