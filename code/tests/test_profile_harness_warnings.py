from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from core.scripts.harness import profile_harness
from core.scripts.harness.example_registry import EXAMPLE_BY_NAME


def test_maybe_skip_output_surfaces_bad_command_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "nsys" / "ch01_performance_basics"
    out_dir.mkdir(parents=True)
    (out_dir / "command.json").write_text("{not-json", encoding="utf-8")
    (out_dir / "status.json").write_text(json.dumps({"exit_code": 0}), encoding="utf-8")

    should_skip, warning = profile_harness.maybe_skip_output(out_dir, True)

    assert should_skip is False
    assert warning
    assert str(out_dir / "command.json") in warning
    reuse_warning = json.loads((out_dir / "reuse_warning.json").read_text(encoding="utf-8"))
    assert reuse_warning["decided_to_rerun"] is True


def test_maybe_skip_output_surfaces_bad_exit_code_type(tmp_path: Path) -> None:
    out_dir = tmp_path / "nsys" / "ch01_performance_basics"
    out_dir.mkdir(parents=True)
    (out_dir / "command.json").write_text(json.dumps({"command": ["python", "-m", "x"]}), encoding="utf-8")
    (out_dir / "status.json").write_text(json.dumps({"exit_code": {"bad": "type"}}), encoding="utf-8")

    should_skip, warning = profile_harness.maybe_skip_output(out_dir, True)

    assert should_skip is False
    assert warning
    assert "Expected integer exit_code" in warning


def test_profile_harness_cli_logs_reuse_warning_for_malformed_existing_output(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    example = EXAMPLE_BY_NAME["ch01_performance_basics"]
    timestamps = [
        time.strftime("%Y%m%d_%H%M%S", time.gmtime()),
        time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time() + 1)),
    ]
    for stamp in timestamps:
        out_dir = output_root / stamp / "nsys" / example.name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "command.json").write_text("{not-json", encoding="utf-8")
        (out_dir / "status.json").write_text(json.dumps({"exit_code": 0}), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.harness.profile_harness",
            "--examples",
            example.name,
            "--profile",
            "nsys",
            "--skip-existing",
            "--dry-run",
            "--output-root",
            str(output_root),
            "--max-examples",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "reuse-warning" in proc.stdout
    warning_files = sorted(output_root.glob("*/nsys/ch01_performance_basics/reuse_warning.json"))
    assert warning_files
    payload = json.loads(warning_files[-1].read_text(encoding="utf-8"))
    assert payload["decided_to_rerun"] is True
