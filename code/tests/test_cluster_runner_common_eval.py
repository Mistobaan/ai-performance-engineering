from __future__ import annotations

from typing import Any, Dict

import core.cluster.runner as cluster_runner


def test_run_cluster_common_eval_common_answer_fast_composes_expected_flags(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_run_cluster_eval_suite(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True, "run_id": kwargs.get("run_id"), "command": ["fake"]}

    monkeypatch.setattr(cluster_runner, "run_cluster_eval_suite", fake_run_cluster_eval_suite)

    result = cluster_runner.run_cluster_common_eval(
        preset="common-answer-fast",
        run_id="2026-03-06_eval_fast",
        hosts=["localhost"],
        labels=["localhost"],
        extra_args=["--skip-render-localhost-report"],
    )

    assert result["success"] is True
    assert result["preset"] == "common-answer-fast"
    assert "fast answer bundle" in result["preset_description"].lower()
    assert "vllm_request_rate_sweep" in result["artifact_roles"]
    assert captured["extra_args"] == [
        "--skip-quick-friction",
        "--skip-monitoring-expectations",
        "--disable-fp4",
        "--health-suite",
        "off",
        "--skip-vllm-multinode",
        "--model",
        "openai/gpt-oss-20b",
        "--tp",
        "1",
        "--isl",
        "512",
        "--osl",
        "128",
        "--concurrency-range",
        "8 16 32",
        "--run-vllm-request-rate-sweep",
        "--vllm-request-rate-range",
        "1 2 4",
        "--vllm-request-rate-max-concurrency",
        "32",
        "--vllm-request-rate-num-prompts",
        "128",
        "--fio-runtime",
        "15",
        "--run-nvbandwidth",
        "--nvbandwidth-quick",
        "--skip-render-localhost-report",
    ]


def test_run_cluster_common_eval_core_system_composes_expected_flags(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_run_cluster_eval_suite(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True, "run_id": kwargs.get("run_id"), "command": ["fake"]}

    monkeypatch.setattr(cluster_runner, "run_cluster_eval_suite", fake_run_cluster_eval_suite)

    result = cluster_runner.run_cluster_common_eval(
        preset="core-system",
        run_id="2026-03-05_eval_core",
        hosts=["localhost"],
        labels=["localhost"],
        coverage_baseline_run_id="baseline_r1",
        extra_args=["--foo"],
    )

    assert result["success"] is True
    assert result["preset"] == "core-system"
    assert "nvbandwidth" in result["preset_description"].lower()
    assert "vllm_request_rate_sweep" in result["artifact_roles"]
    assert captured["mode"] == "full"
    assert captured["hosts"] == ["localhost"]
    assert captured["labels"] == ["localhost"]
    assert captured["extra_args"] == [
        "--run-vllm-request-rate-sweep",
        "--run-nvbandwidth",
        "--coverage-baseline-run-id",
        "baseline_r1",
        "--foo",
    ]


def test_run_cluster_common_eval_rejects_unknown_preset() -> None:
    result = cluster_runner.run_cluster_common_eval(preset="not-a-preset", hosts=["localhost"])
    assert result["success"] is False
    assert "Unknown preset" in result["error"]
