from __future__ import annotations

import subprocess
import sys
from pathlib import Path


TEST_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_update_history_index_cli_rejects_malformed_summary(tmp_path: Path) -> None:
    summary_json = tmp_path / "summary.json"
    summary_json.write_text("{not-json", encoding="utf-8")
    regression_summary = tmp_path / "regression_summary.md"
    regression_summary.write_text("# placeholder\n", encoding="utf-8")
    history_root = tmp_path / "history"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.benchmarks.update_history_index",
            "--summary-json",
            str(summary_json),
            "--regression-summary",
            str(regression_summary),
            "--history-root",
            str(history_root),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Failed to read tier-1 summary JSON" in completed.stderr
    assert str(summary_json) in completed.stderr
    assert "Traceback" not in completed.stderr

