from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_detect_regressions(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "core.analysis.detect_regressions", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_detect_regressions_cli_fails_cleanly_on_malformed_current_json(tmp_path: Path) -> None:
    current_path = tmp_path / "current.json"
    baseline_path = tmp_path / "baseline.json"
    current_path.write_text("{bad-json", encoding="utf-8")
    baseline_path.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")

    result = _run_detect_regressions(
        "--current",
        str(current_path),
        "--baseline",
        str(baseline_path),
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 1
    assert "Failed to read benchmark run JSON from" in result.stderr
    assert str(current_path) in result.stderr
    assert "Traceback" not in result.stderr


def test_detect_regressions_cli_fails_cleanly_on_wrong_top_level_shape(tmp_path: Path) -> None:
    current_path = tmp_path / "current.json"
    baseline_path = tmp_path / "baseline.json"
    current_path.write_text("[]", encoding="utf-8")
    baseline_path.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")

    result = _run_detect_regressions(
        "--current",
        str(current_path),
        "--baseline",
        str(baseline_path),
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 1
    assert "expected JSON object, got list" in result.stderr
    assert "Traceback" not in result.stderr
