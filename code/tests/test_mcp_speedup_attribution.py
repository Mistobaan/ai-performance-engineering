from __future__ import annotations

import json
from pathlib import Path

import mcp.mcp_server as mcp_server


def test_extract_speedup_attribution_surfaces_malformed_log_lines(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "benchmark.log"
    valid_payload = {
        "chapter": "ch01",
        "example": "demo",
        "technique": "optimized_demo.py",
        "status": "succeeded",
        "speedup": 1.2,
        "time_ms": 10.0,
    }
    log_path.write_text(
        "\n".join(
            [
                "{not-json",
                json.dumps({"message": f"EVENT optimization_result {json.dumps(valid_payload)}"}),
            ]
        ),
        encoding="utf-8",
    )

    attribution = mcp_server._extract_speedup_attribution({"run_dir": str(run_dir)})

    assert attribution is not None
    assert len(attribution["items"]) == 1
    assert attribution["warning_counts"]["malformed_json_lines"] == 1
    assert attribution["warning_counts"]["malformed_event_payloads"] == 0
    assert attribution["warnings"]


def test_extract_speedup_attribution_returns_warning_only_on_malformed_event_payloads(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "benchmark.log"
    log_path.write_text(
        json.dumps({"message": "EVENT optimization_result {not-json"}) + "\n",
        encoding="utf-8",
    )

    attribution = mcp_server._extract_speedup_attribution({"run_dir": str(run_dir)})

    assert attribution is not None
    assert attribution["items"] == []
    assert attribution["warning_counts"]["malformed_json_lines"] == 0
    assert attribution["warning_counts"]["malformed_event_payloads"] == 1
    assert any("failed to parse EVENT payload JSON" in warning for warning in attribution["warnings"])
