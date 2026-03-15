from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.benchmark.continuous_benchmark import load_config, run_command


def test_load_config_reports_path_for_malformed_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        load_config(config_path)

    message = str(excinfo.value)
    assert "continuous benchmark config" in message
    assert str(config_path) in message


def test_run_command_surfaces_unreadable_output_json_warning(tmp_path: Path) -> None:
    bad_output = tmp_path / "bad_output.json"
    bad_output.write_text("{not-json", encoding="utf-8")

    payload = run_command(
        {
            "name": "demo",
            "command": [sys.executable, "-c", "print('ok')"],
            "workdir": str(tmp_path),
            "output_json": str(bad_output),
        }
    )

    assert payload["returncode"] == 0
    assert payload["benchmark_output_artifact"] == str(bad_output)
    assert payload["benchmark_output_warning"]
    assert str(bad_output) in payload["benchmark_output_warning"]
    assert payload["warnings"]


def test_cli_persists_benchmark_output_warning_in_summary(tmp_path: Path) -> None:
    bad_output = tmp_path / "bad_output.json"
    bad_output.write_text("{not-json", encoding="utf-8")

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "name": "demo",
                    "command": [sys.executable, "-c", "print('ok')"],
                    "workdir": str(tmp_path),
                    "output_json": str(bad_output),
                }
            ]
        ),
        encoding="utf-8",
    )

    artifact_dir = tmp_path / "artifacts"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.benchmark.continuous_benchmark",
            str(config_path),
            "--artifact-dir",
            str(artifact_dir),
            "--tag",
            "warning-demo",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    summary_files = sorted(artifact_dir.glob("benchmark_run_*.json"))
    assert summary_files

    payload = json.loads(summary_files[-1].read_text(encoding="utf-8"))
    bench = payload["benchmarks"][0]
    assert bench["benchmark_output_warning"]
    assert str(bad_output) in bench["benchmark_output_warning"]
