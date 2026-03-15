from core.analysis.report_generator import PerformanceReport
from core.benchmark import comparison
from core.benchmark import performance_targets


def test_load_peak_benchmark_results_surfaces_invalid_json_warning(tmp_path):
    report_path = tmp_path / "benchmark_peak_results_demo.json"
    report_path.write_text("{bad", encoding="utf-8")

    payload, warning, artifact_path = performance_targets._load_peak_benchmark_results(tmp_path)

    assert payload is None
    assert warning is not None
    assert str(report_path) in warning
    assert artifact_path == str(report_path)


def test_build_targets_preserves_warning_and_default_source(tmp_path):
    report_path = tmp_path / "benchmark_peak_results_demo.json"
    report_path.write_text("[]", encoding="utf-8")

    targets, warnings, artifact_path, source = performance_targets._build_targets(tmp_path)

    assert targets["overall"]["fp16_compute_tflops"]["target"] == 2000
    assert warnings
    assert "expected JSON object, got list" in warnings[0]
    assert artifact_path == str(report_path)
    assert source == "defaults_due_to_peak_results_warning"


def test_report_generator_includes_target_provenance_warnings(monkeypatch):
    monkeypatch.setattr(
        "core.analysis.report_generator.get_targets_metadata",
        lambda: {
            "source": "defaults_due_to_peak_results_warning",
            "artifact_path": "/tmp/bad_peak_results.json",
            "warnings": ["Failed to read peak benchmark results from /tmp/bad_peak_results.json: boom"],
        },
    )

    report = PerformanceReport().generate_markdown()

    assert "Target Provenance Warnings" in report
    assert "/tmp/bad_peak_results.json" in report
    assert "defaults_due_to_peak_results_warning" in report


def test_chapter_metric_config_logs_target_metadata_warning_once(monkeypatch, caplog):
    monkeypatch.setattr(
        comparison,
        "_target_metadata_loader",
        lambda: {"warnings": ["bad target metadata"], "artifact_path": "/tmp/x", "source": "defaults"},
    )
    monkeypatch.setattr(comparison, "_target_metadata_warning_logged", False)

    with caplog.at_level("WARNING"):
        comparison.get_chapter_metric_config("ch01")
        comparison.get_chapter_metric_config("ch01")

    messages = [record.getMessage() for record in caplog.records if "Performance target metadata warnings" in record.getMessage()]
    assert len(messages) == 1
    assert "bad target metadata" in messages[0]
