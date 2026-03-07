from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path


def _load_renderer_module():
    script_path = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "render_localhost_field_report_package.py"
    spec = importlib.util.spec_from_file_location("render_localhost_field_report_package", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seed_minimal_cluster_root(cluster_root: Path, run_id: str, label: str) -> None:
    run_dir = cluster_root / "runs" / run_id
    structured = run_dir / "structured"
    figures = run_dir / "figures"
    structured.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "manifest.json", {"run_id": run_id})
    _write_json(
        structured / f"{run_id}_suite_steps.json",
        [
            {"name": "preflight_services", "exit_code": 0, "start_time": "2026-03-05T21:08:24+00:00"},
            {"name": "discovery", "exit_code": 0, "start_time": "2026-03-05T21:08:25+00:00"},
            {"name": "hang_triage_bundle", "exit_code": 0, "start_time": "2026-03-05T21:08:28+00:00"},
            {"name": "connectivity_probe", "exit_code": 0, "start_time": "2026-03-05T21:08:29+00:00"},
            {"name": "nccl_env_sensitivity", "exit_code": 0, "start_time": "2026-03-05T21:08:47+00:00"},
            {"name": "vllm_serve_sweep", "exit_code": 0, "start_time": "2026-03-05T21:09:53+00:00"},
            {"name": "validate_required_artifacts", "exit_code": 0, "start_time": "2026-03-05T21:17:25+00:00"},
            {"name": "manifest_refresh", "exit_code": 0, "start_time": "2026-03-05T21:17:26+00:00"},
        ],
    )
    _write_json(
        structured / f"{run_id}_{label}_meta.json",
        {"commands": {"nvidia_smi_l": {"stdout": "GPU 0: Test GPU"}}},
    )
    _write_json(structured / f"{run_id}_{label}_hang_triage_readiness.json", {"status": "ok"})
    _write_json(
        structured / f"{run_id}_torchrun_connectivity_probe.json",
        {
            "status": "ok",
            "world_size": 1,
            "ranks": [{"barrier_ms": [0.08, 0.07], "payload_probe": {"algbw_gbps": 120.288}}],
        },
    )
    _write_json(
        structured / f"{run_id}_nccl_env_sensitivity.json",
        {"status": "ok", "failure_count": 0, "baseline_peak_busbw_gbps": 0.0},
    )
    _write_json(
        structured / f"{run_id}_node1_nccl.json",
        {"results": [{"algbw_gbps": 2246.8, "size_bytes": 67108864}]},
    )
    _write_json(structured / f"{run_id}_preflight_services.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_{label}_meta_nvlink_topology.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_node_parity_summary.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_{label}_fio.json", {"status": "ok"})

    vllm_csv = structured / f"{run_id}_{label}_vllm_serve_sweep.csv"
    vllm_csv.write_text(
        "\n".join(
            [
                "concurrency,total_token_throughput,mean_ttft_ms,p99_ttft_ms,p99_tpot_ms",
                "1,405.615,72.198,88.385,9.128",
                "2,921.609,31.225,35.526,6.354",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (structured / f"{run_id}_{label}_vllm_serve_sweep.jsonl").write_text("{}\n", encoding="utf-8")


def test_render_localhost_report_handles_missing_operator_checks(tmp_path: Path) -> None:
    module = _load_renderer_module()
    run_id = "2026-03-05_localhost_common_eval_r1"
    label = "localhost"
    cluster_root = tmp_path / "cluster"
    _seed_minimal_cluster_root(cluster_root, run_id, label)

    args = Namespace(run_id=run_id, label=label, root=cluster_root, run_dir=cluster_root / "runs" / run_id)

    report = module.render_report(args)
    notes = module.render_notes(args)

    assert "Operator checks | not run in this preset" in report
    assert "operator-friction and monitoring artifacts were not requested in this preset run." in report
    assert "Run localhost common system eval" in report
    assert "Operator checks are optional and skipped in this preset" in notes
    assert "quick_friction | not run" in notes
