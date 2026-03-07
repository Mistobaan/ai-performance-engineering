from __future__ import annotations

import socket
import subprocess
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cluster_root() -> Path:
    return _repo_root() / "cluster"


def _cluster_run_dir(run_id: str) -> Path:
    return _cluster_root() / "runs" / run_id


def _cluster_run_layout(run_id: str) -> Dict[str, str]:
    run_dir = _cluster_run_dir(run_id)
    return {
        "run_dir": str(run_dir),
        "structured_dir": str(run_dir / "structured"),
        "raw_dir": str(run_dir / "raw"),
        "figures_dir": str(run_dir / "figures"),
        "reports_dir": str(run_dir / "reports"),
        "manifest_path": str(run_dir / "manifest.json"),
    }


def _run_cmd(cmd: List[str], *, cwd: Optional[Path] = None, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_ms": duration_ms,
        }
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "command": cmd,
            "returncode": None,
            "stdout": (exc.stdout or ""),
            "stderr": (exc.stderr or ""),
            "error": f"timeout after {timeout_seconds}s",
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
            "duration_ms": duration_ms,
        }


def _default_run_id_prefix() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def _default_primary_label() -> str:
    host = socket.gethostname() or "local"
    return host.split(".")[0] or "local"


_COMMON_EVAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "common-answer-fast": {
        "description": (
            "Agent-friendly fast answer bundle for the benchmark questions people ask first: "
            "NCCL all-reduce, compact vLLM concurrency and request-rate sweeps, GEMM sanity, "
            "GPU STREAM, fio, quick nvbandwidth, scorecard, coverage, and MLPerf-alignment. "
            "Uses a smaller modern LLM profile and skips non-benchmark friction/monitoring extras."
        ),
        "extra_args": [
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
        ],
        "artifact_roles": [
            "meta",
            "manifest",
            "suite_steps",
            "nccl_allreduce",
            "vllm_concurrency_sweep",
            "vllm_request_rate_sweep",
            "gemm_sanity",
            "gpu_stream",
            "fio",
            "nvbandwidth",
            "cluster_scorecard",
            "benchmark_coverage_analysis",
            "mlperf_alignment",
        ],
    },
    "core-system": {
        "description": (
            "Common single-node or cluster evaluation bundle: discovery, NCCL all-reduce, "
            "vLLM concurrency sweep, vLLM request-rate sweep, GEMM sanity, GPU STREAM, fio, "
            "nvbandwidth, scorecard, coverage, and MLPerf-alignment summary."
        ),
        "extra_args": [
            "--run-vllm-request-rate-sweep",
            "--run-nvbandwidth",
        ],
        "artifact_roles": [
            "meta",
            "manifest",
            "suite_steps",
            "nccl_allreduce",
            "vllm_concurrency_sweep",
            "vllm_request_rate_sweep",
            "gemm_sanity",
            "gpu_stream",
            "fio",
            "nvbandwidth",
            "cluster_scorecard",
            "benchmark_coverage_analysis",
            "mlperf_alignment",
        ],
    },
    "modern-llm": {
        "description": (
            "High-signal modern LLM system evaluation: core-system plus request-rate repeats, "
            "allreduce stability, allreduce latency composition, allgather control-plane, "
            "NCCL all-to-all, NCCL algorithm comparison, train-step, and strict completeness gates."
        ),
        "extra_args": ["--modern-llm-profile"],
        "artifact_roles": [
            "meta",
            "manifest",
            "suite_steps",
            "nccl_allreduce",
            "vllm_concurrency_sweep",
            "vllm_request_rate_sweep",
            "gemm_sanity",
            "gpu_stream",
            "fio",
            "nvbandwidth",
            "allreduce_stability",
            "allreduce_latency_comp",
            "allgather_control_plane",
            "nccl_alltoall",
            "nccl_algo_comparison",
            "train_step",
            "cluster_scorecard",
            "benchmark_coverage_analysis",
            "mlperf_alignment",
        ],
    },
    "multinode-readiness": {
        "description": (
            "Fail-fast multi-node contract validation only. Produces readiness evidence without running workloads."
        ),
        "extra_args": [
            "--modern-llm-profile",
            "--multinode-readiness-check-only",
        ],
        "artifact_roles": [
            "multinode_readiness",
        ],
    },
}


def _smoke_eval(run_id: str, *, primary_label: Optional[str] = None, timeout_seconds: int = 120) -> Dict[str, Any]:
    cluster_root = _cluster_root()
    scripts_dir = cluster_root / "scripts"
    run_dir = _cluster_run_dir(run_id)
    struct_dir = run_dir / "structured"
    raw_dir = run_dir / "raw"
    reports_dir = run_dir / "reports"
    figures_dir = run_dir / "figures"
    for path in (struct_dir, raw_dir, reports_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    label = primary_label or _default_primary_label()
    meta_path = struct_dir / f"{run_id}_{label}_meta.json"

    collect_script = scripts_dir / "collect_system_info.sh"
    manifest_script = scripts_dir / "write_manifest.py"

    if not collect_script.exists():
        return {"success": False, "error": f"Missing script: {collect_script}"}
    if not manifest_script.exists():
        return {"success": False, "error": f"Missing script: {manifest_script}"}

    collect_cmd = ["bash", str(collect_script), "--output", str(meta_path), "--label", str(label)]
    collect = _run_cmd(collect_cmd, cwd=cluster_root, timeout_seconds=timeout_seconds)
    if collect.get("returncode") != 0:
        return {
            "success": False,
            "mode": "smoke",
            "run_id": run_id,
            "primary_label": label,
            "meta_path": str(meta_path),
            "collect": collect,
            "error": "collect_system_info failed",
        }

    manifest_cmd = [
        "python3",
        str(manifest_script),
        "--root",
        str(cluster_root),
        "--run-id",
        str(run_id),
        "--run-dir",
        str(run_dir),
    ]
    manifest = _run_cmd(manifest_cmd, cwd=cluster_root, timeout_seconds=timeout_seconds)
    manifest_path = (manifest.get("stdout") or "").strip().splitlines()[-1] if manifest.get("stdout") else ""

    return {
        "success": bool(manifest.get("returncode") == 0),
        "mode": "smoke",
        "run_id": run_id,
        "primary_label": label,
        "meta_path": str(meta_path),
        "manifest_path": manifest_path or None,
        "collect": collect,
        "manifest": manifest,
        **_cluster_run_layout(run_id),
    }


def run_cluster_eval_suite(
    *,
    mode: str = "smoke",
    run_id: Optional[str] = None,
    hosts: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    ssh_user: Optional[str] = None,
    ssh_key: Optional[str] = None,
    oob_if: Optional[str] = None,
    socket_ifname: Optional[str] = None,
    nccl_ib_hca: Optional[str] = None,
    primary_label: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the cluster eval suite (full) or a fast local smoke run.

    - mode=smoke:
      Writes under `cluster/runs/<run_id>/...` via collect_system_info.sh and write_manifest.py.

    - mode=full:
      Invokes `cluster/scripts/run_cluster_eval_suite.sh` (requires --hosts).
    """
    run_id_value = run_id or _default_run_id_prefix()
    mode_value = (mode or "smoke").strip().lower()
    timeout = int(timeout_seconds) if timeout_seconds is not None else None

    if mode_value in {"smoke", "local", "mini"}:
        return _smoke_eval(run_id_value, primary_label=primary_label, timeout_seconds=timeout or 120)

    if mode_value not in {"full", "eval", "suite"}:
        return {"success": False, "error": f"Unknown mode: {mode!r} (expected smoke|full)"}

    cluster_root = _cluster_root()
    script = cluster_root / "scripts" / "run_cluster_eval_suite.sh"
    if not script.exists():
        return {"success": False, "error": f"Missing script: {script}"}

    hosts_list = hosts or []
    hosts_list = [h.strip() for h in hosts_list if isinstance(h, str) and h.strip()]
    if not hosts_list:
        return {"success": False, "error": "--hosts is required in full mode", "mode": "full", "run_id": run_id_value}

    cmd: List[str] = ["bash", str(script), "--hosts", ",".join(hosts_list), "--run-id", str(run_id_value)]
    if labels:
        labels_list = [l.strip() for l in labels if isinstance(l, str) and l.strip()]
        if labels_list:
            cmd.extend(["--labels", ",".join(labels_list)])
    if ssh_user:
        cmd.extend(["--ssh-user", str(ssh_user)])
    if ssh_key:
        cmd.extend(["--ssh-key", str(ssh_key)])
    if oob_if:
        cmd.extend(["--oob-if", str(oob_if)])
    if socket_ifname:
        cmd.extend(["--socket-ifname", str(socket_ifname)])
    if nccl_ib_hca:
        cmd.extend(["--nccl-ib-hca", str(nccl_ib_hca)])
    if primary_label:
        cmd.extend(["--primary-label", str(primary_label)])
    if extra_args:
        cmd.extend([str(x) for x in extra_args if str(x).strip()])

    result = _run_cmd(cmd, cwd=cluster_root, timeout_seconds=timeout)
    success = result.get("returncode") == 0
    return {
        "success": bool(success),
        "mode": "full",
        "run_id": run_id_value,
        "command": cmd,
        **_cluster_run_layout(run_id_value),
        **result,
    }


def run_cluster_common_eval(
    *,
    preset: str = "core-system",
    run_id: Optional[str] = None,
    hosts: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    ssh_user: Optional[str] = None,
    ssh_key: Optional[str] = None,
    oob_if: Optional[str] = None,
    socket_ifname: Optional[str] = None,
    nccl_ib_hca: Optional[str] = None,
    primary_label: Optional[str] = None,
    coverage_baseline_run_id: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    preset_name = (preset or "core-system").strip().lower()
    preset_payload = _COMMON_EVAL_PRESETS.get(preset_name)
    if preset_payload is None:
        return {
            "success": False,
            "error": f"Unknown preset: {preset!r} (expected one of {', '.join(sorted(_COMMON_EVAL_PRESETS))})",
        }

    merged_extra_args = [str(arg) for arg in preset_payload["extra_args"]]
    if coverage_baseline_run_id:
        merged_extra_args.extend(["--coverage-baseline-run-id", str(coverage_baseline_run_id)])
    if extra_args:
        merged_extra_args.extend(str(arg) for arg in extra_args if str(arg).strip())

    result = run_cluster_eval_suite(
        mode="full",
        run_id=run_id,
        hosts=hosts,
        labels=labels,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        oob_if=oob_if,
        socket_ifname=socket_ifname,
        nccl_ib_hca=nccl_ib_hca,
        primary_label=primary_label,
        extra_args=merged_extra_args,
        timeout_seconds=timeout_seconds,
    )
    return {
        "preset": preset_name,
        "preset_description": preset_payload["description"],
        "artifact_roles": list(preset_payload["artifact_roles"]),
        "coverage_baseline_run_id": coverage_baseline_run_id,
        **result,
    }


def validate_field_report_requirements(
    *,
    report: Optional[str] = None,
    notes: Optional[str] = None,
    template: Optional[str] = None,
    runbook: Optional[str] = None,
    canonical_run_id: Optional[str] = None,
    allow_run_id: Optional[List[str]] = None,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """Run the field-report validator script and return stdout/stderr."""
    root = _repo_root()
    script = root / "cluster" / "scripts" / "validate_field_report_requirements.sh"
    if not script.exists():
        return {"success": False, "error": f"Missing validator script: {script}"}

    cmd: List[str] = ["bash", str(script)]
    if report:
        cmd.extend(["--report", str(report)])
    if notes:
        cmd.extend(["--notes", str(notes)])
    if template:
        cmd.extend(["--template", str(template)])
    if runbook:
        cmd.extend(["--runbook", str(runbook)])
    if canonical_run_id:
        cmd.extend(["--canonical-run-id", str(canonical_run_id)])
    if allow_run_id:
        for rid in allow_run_id:
            if rid:
                cmd.extend(["--allow-run-id", str(rid)])

    result = _run_cmd(cmd, cwd=root, timeout_seconds=int(timeout_seconds))
    success = result.get("returncode") == 0
    payload: Dict[str, Any] = {"success": bool(success), **result}
    if not success and not payload.get("error"):
        rc = payload.get("returncode")
        payload["error"] = f"validator failed (returncode={rc})"
    return payload


def build_canonical_package(
    *,
    canonical_run_id: str,
    comparison_run_ids: Optional[List[str]] = None,
    historical_run_ids: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    """Materialize a clean, non-destructive cluster package for one canonical run family."""
    canonical_run = (canonical_run_id or "").strip()
    if not canonical_run:
        return {"success": False, "error": "canonical_run_id is required"}

    root = _repo_root()
    cluster_root = _cluster_root()
    script = cluster_root / "scripts" / "build_canonical_package.py"
    if not script.exists():
        return {"success": False, "error": f"Missing script: {script}"}

    comparison = [rid.strip() for rid in (comparison_run_ids or []) if isinstance(rid, str) and rid.strip()]
    historical = [rid.strip() for rid in (historical_run_ids or []) if isinstance(rid, str) and rid.strip()]

    output_dir_value = (output_dir or "cluster/canonical_package").strip()
    cmd: List[str] = [
        "python3",
        str(script),
        "--canonical-run-id",
        canonical_run,
        "--output-dir",
        output_dir_value,
    ]
    for rid in comparison:
        cmd.extend(["--comparison-run-id", rid])
    for rid in historical:
        cmd.extend(["--historical-run-id", rid])

    result = _run_cmd(cmd, cwd=root, timeout_seconds=int(timeout_seconds))
    success = result.get("returncode") == 0
    package_root = (root / output_dir_value).resolve()

    payload: Dict[str, Any] = {
        "success": bool(success),
        "canonical_run_id": canonical_run,
        "comparison_run_ids": comparison,
        "historical_run_ids": historical,
        "output_dir": str(package_root),
        "package_readme_path": str(package_root / "README.md"),
        "package_manifest_path": str(package_root / "package_manifest.json"),
        "cleanup_keep_run_ids_path": str(package_root / "cleanup_keep_run_ids.txt"),
        "historical_reference_path": str(package_root / "historical-multinode-reference.md"),
        **result,
    }
    if not success and not payload.get("error"):
        rc = payload.get("returncode")
        payload["error"] = f"canonical package build failed (returncode={rc})"
    return payload


def promote_cluster_run(
    *,
    run_id: str,
    label: str = "localhost",
    allow_run_ids: Optional[List[str]] = None,
    publish_report_path: Optional[str] = None,
    publish_notes_path: Optional[str] = None,
    skip_render_localhost_report: bool = False,
    skip_validate_localhost_report: bool = False,
    cleanup: bool = False,
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    """Promote one run-local result tree into the published cluster package."""
    run_id_value = (run_id or "").strip()
    if not run_id_value:
        return {"success": False, "error": "run_id is required"}

    root = _repo_root()
    cluster_root = _cluster_root()
    script = cluster_root / "scripts" / "promote_run.py"
    if not script.exists():
        return {"success": False, "error": f"Missing script: {script}"}

    publish_report_value = (publish_report_path or "cluster/field-report-localhost.md").strip()
    publish_notes_value = (publish_notes_path or "cluster/field-report-localhost-notes.md").strip()
    allow = [rid.strip() for rid in (allow_run_ids or []) if isinstance(rid, str) and rid.strip()]

    cmd: List[str] = [
        "python3",
        str(script),
        "--run-id",
        run_id_value,
        "--label",
        str(label or "localhost"),
        "--repo-root",
        str(root),
        "--publish-report-path",
        publish_report_value,
        "--publish-notes-path",
        publish_notes_value,
    ]
    for rid in allow:
        cmd.extend(["--allow-run-id", rid])
    if skip_render_localhost_report:
        cmd.append("--skip-render-localhost-report")
    if skip_validate_localhost_report:
        cmd.append("--skip-validate-localhost-report")
    if cleanup:
        cmd.append("--cleanup")

    result = _run_cmd(cmd, cwd=root, timeout_seconds=int(timeout_seconds))
    success = result.get("returncode") == 0
    payload: Dict[str, Any] = {
        "success": bool(success),
        "run_id": run_id_value,
        "label": label or "localhost",
        **_cluster_run_layout(run_id_value),
        "published_root": str(cluster_root / "published" / "current"),
        "published_structured_dir": str(cluster_root / "published" / "current" / "structured"),
        "published_raw_dir": str(cluster_root / "published" / "current" / "raw"),
        "published_figures_dir": str(cluster_root / "published" / "current" / "figures"),
        "published_reports_dir": str(cluster_root / "published" / "current" / "reports"),
        "published_manifest_path": str(cluster_root / "published" / "current" / "manifest.json"),
        "published_localhost_report_path": str((root / publish_report_value).resolve()),
        "published_localhost_notes_path": str((root / publish_notes_value).resolve()),
        "allow_run_ids": allow,
        **result,
    }
    stdout = result.get("stdout") or ""
    try:
        parsed = json.loads(stdout)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        payload.update(parsed)
    if not success and not payload.get("error"):
        rc = payload.get("returncode")
        payload["error"] = f"promote run failed (returncode={rc})"
    return payload
