from pathlib import Path

from core.analysis import llm_profile_analyzer
from core.harness import run_benchmarks


def test_run_llm_analysis_for_benchmark_surfaces_source_read_warning(tmp_path: Path, monkeypatch) -> None:
    chapter_dir = tmp_path / "ch_demo"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = chapter_dir / "baseline_demo.py"
    optimized_path = chapter_dir / "optimized_demo.py"
    baseline_path.write_text("baseline", encoding="utf-8")
    optimized_path.write_text("optimized", encoding="utf-8")

    benchmark_result = {
        "example": "demo",
        "baseline_time_ms": 10.0,
        "optimizations": [
            {"status": "succeeded", "speedup": 1.5, "time_ms": 6.5, "file": "optimized_demo.py"}
        ],
    }

    original_read_text = Path.read_text

    def _patched_read_text(self: Path, *args, **kwargs):  # type: ignore[override]
        if self == optimized_path:
            raise OSError("optimized source boom")
        return original_read_text(self, *args, **kwargs)

    class _FakeAnalyzer:
        def __init__(self, provider=None):
            self.provider = provider

        def analyze_differential(self, diff_report, baseline_code=None, optimized_code=None, environment=None):
            assert diff_report["overall_speedup"] == 1.5
            assert baseline_code == "baseline"
            assert optimized_code is None
            assert environment is not None
            return llm_profile_analyzer.LLMAnalysisResult(
                provider="fake",
                model="fake-model",
                why_faster="ok",
            )

    monkeypatch.setattr(Path, "read_text", _patched_read_text)
    monkeypatch.setattr(llm_profile_analyzer, "LLMProfileAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(llm_profile_analyzer, "collect_environment_context", lambda: llm_profile_analyzer.EnvironmentContext(python_version="3.12"))

    result = run_benchmarks._run_llm_analysis_for_benchmark(
        benchmark_result,
        profiling_output_dir=None,
        chapter_dir=chapter_dir,
        llm_provider="fake",
        use_cache=False,
    )

    assert result is not None
    assert result["warnings"]
    assert str(optimized_path) in result["warnings"][0]
    markdown = (chapter_dir / "llm_analysis" / "llm_analysis_demo.md").read_text(encoding="utf-8")
    assert "## Analysis Warnings" in markdown
    assert "optimized source boom" in markdown


def test_run_llm_analysis_for_benchmark_cache_key_warning_forces_fresh_analysis(tmp_path: Path, monkeypatch) -> None:
    chapter_dir = tmp_path / "ch_demo"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = chapter_dir / "baseline_demo.py"
    optimized_path = chapter_dir / "optimized_demo.py"
    baseline_path.write_text("baseline", encoding="utf-8")
    optimized_path.write_text("optimized", encoding="utf-8")

    llm_dir = chapter_dir / "llm_analysis"
    llm_dir.mkdir(parents=True, exist_ok=True)
    (llm_dir / "llm_analysis_demo.md").write_text("cached", encoding="utf-8")
    cache_key_path = llm_dir / ".cache_key_demo"
    cache_key_path.write_text("cached-key", encoding="utf-8")

    benchmark_result = {
        "example": "demo",
        "baseline_time_ms": 10.0,
        "optimizations": [
            {"status": "succeeded", "speedup": 1.25, "time_ms": 8.0, "file": "optimized_demo.py"}
        ],
    }

    original_read_text = Path.read_text

    def _patched_read_text(self: Path, *args, **kwargs):  # type: ignore[override]
        if self == cache_key_path:
            raise OSError("cache key boom")
        return original_read_text(self, *args, **kwargs)

    calls = {"count": 0}

    class _FakeAnalyzer:
        def __init__(self, provider=None):
            self.provider = provider

        def analyze_differential(self, diff_report, baseline_code=None, optimized_code=None, environment=None):
            calls["count"] += 1
            return llm_profile_analyzer.LLMAnalysisResult(
                provider="fake",
                model="fake-model",
                why_faster="fresh",
            )

    monkeypatch.setattr(Path, "read_text", _patched_read_text)
    monkeypatch.setattr(llm_profile_analyzer, "LLMProfileAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(llm_profile_analyzer, "collect_environment_context", lambda: llm_profile_analyzer.EnvironmentContext(python_version="3.12"))

    result = run_benchmarks._run_llm_analysis_for_benchmark(
        benchmark_result,
        profiling_output_dir=None,
        chapter_dir=chapter_dir,
        llm_provider="fake",
        use_cache=True,
    )

    assert result is not None
    assert calls["count"] == 1
    assert not result.get("cached", False)
    assert result["warnings"]
    assert str(cache_key_path) in result["warnings"][0]
