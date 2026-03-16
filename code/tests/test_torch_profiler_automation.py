from __future__ import annotations

import json
import subprocess
from pathlib import Path

from core.profiling.torch_profiler import TorchProfilerAutomation


def test_torch_profiler_automation_surfaces_malformed_artifact_warnings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "demo.py"
    script_path.write_text("print('demo')\n", encoding="utf-8")

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        capture_dir = Path(cmd[cmd.index("--output-dir") + 1])
        capture_dir.mkdir(parents=True, exist_ok=True)
        (capture_dir / "trace.json").write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
        (capture_dir / "torch_profile_summary.json").write_text("[]", encoding="utf-8")
        (capture_dir / "metadata.json").write_text("{bad-json", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    automation = TorchProfilerAutomation(tmp_path / "runs")
    result = automation.profile(script=script_path, output_name="demo")

    assert result["success"] is True
    assert result["summary"] is None
    assert result["metadata"] is None
    assert result["summary_error"]
    assert "expected dict, got list" in result["summary_error"]
    assert result["metadata_error"]
    assert "Failed to read torch profiler metadata JSON" in result["metadata_error"]
    assert result["warnings"] == automation.last_run["warnings"]
    assert result["summary_path"].endswith("torch_profile_summary.json")
    assert result["metadata_path"].endswith("metadata.json")


def test_torch_profiler_automation_surfaces_missing_trace_warning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "demo.py"
    script_path.write_text("print('demo')\n", encoding="utf-8")

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        capture_dir = Path(cmd[cmd.index("--output-dir") + 1])
        capture_dir.mkdir(parents=True, exist_ok=True)
        (capture_dir / "torch_profile_summary.json").write_text(json.dumps({"top_ops": []}), encoding="utf-8")
        (capture_dir / "metadata.json").write_text(json.dumps({"mode": "full"}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    automation = TorchProfilerAutomation(tmp_path / "runs")
    result = automation.profile(script=script_path, output_name="demo")

    assert result["success"] is True
    assert result["trace_path"] is None
    assert result["trace_warning"]
    assert "Missing torch profiler trace artifact" in result["trace_warning"]
    assert result["warnings"] == [result["trace_warning"]]
