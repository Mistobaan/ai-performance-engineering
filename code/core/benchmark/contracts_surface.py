"""Shared benchmark methodology and warehouse contract surfaces.

This module provides one small summary object that can be reused by CLI, MCP,
and dashboard handlers without duplicating path or schema logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

_SURFACES = {
    "methodology": {
        "path": REPO_ROOT / "docs" / "benchmark_methodology.md",
        "kind": "doc",
        "description": "Repo-wide benchmark methodology, evidence policy, and bottleneck model.",
    },
    "warehouse": {
        "path": REPO_ROOT / "docs" / "performance_warehouse.md",
        "kind": "doc",
        "description": "Planet-scale warehouse design, schema, telemetry joins, and retention policy.",
    },
    "workload_spec": {
        "path": REPO_ROOT / "templates" / "benchmark_workload_spec.yaml",
        "kind": "yaml",
        "description": "Frozen workload definition and serving fairness contract.",
    },
    "benchmark_run": {
        "path": REPO_ROOT / "templates" / "benchmark_run.yaml",
        "kind": "yaml",
        "description": "Declarative BenchmarkRun contract with observability and sinks.",
    },
    "warehouse_contract": {
        "path": REPO_ROOT / "templates" / "performance_warehouse_contract.yaml",
        "kind": "yaml",
        "description": "Raw-vs-curated warehouse contract, telemetry sources, and retention tiers.",
    },
    "kubernetes_service": {
        "path": REPO_ROOT / "cluster" / "docs" / "kubernetes_benchmark_service.md",
        "kind": "doc",
        "description": "Kubernetes-native operator and control-loop design for BenchmarkRun.",
    },
    "benchmark_run_crd": {
        "path": REPO_ROOT / "cluster" / "configs" / "benchmarkrun-crd.yaml",
        "kind": "yaml",
        "description": "CRD sketch for BenchmarkRun.",
    },
}


def _summarize_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    summary: Dict[str, Any] = {"top_level_keys": sorted(payload.keys())}
    spec = payload.get("spec")
    if isinstance(spec, dict):
        summary["spec_keys"] = sorted(spec.keys())
        if "layers" in spec and isinstance(spec["layers"], list):
            summary["enabled_layers"] = [
                layer.get("name")
                for layer in spec["layers"]
                if isinstance(layer, dict) and layer.get("enabled")
            ]
        if "observability" in spec:
            summary["has_observability"] = True
        if "sinks" in spec:
            summary["has_sinks"] = True
    return summary


def _surface_entry(name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(meta["path"])
    exists = path.exists()
    entry: Dict[str, Any] = {
        "name": name,
        "path": str(path),
        "exists": exists,
        "kind": meta["kind"],
        "description": meta["description"],
    }
    if exists and meta["kind"] == "yaml":
        entry["summary"] = _summarize_yaml(path)
    return entry


def get_benchmark_contracts_summary() -> Dict[str, Any]:
    """Return the repo-exposed benchmark methodology and warehouse surfaces."""
    contracts = {name: _surface_entry(name, meta) for name, meta in _SURFACES.items()}
    return {
        "available": True,
        "repo_root": str(REPO_ROOT),
        "contracts": contracts,
        "interfaces": {
            "cli": "python -m cli.aisp tools benchmark-contracts",
            "dashboard_api": "/api/benchmark/contracts",
            "mcp_tool": "benchmark_contracts",
        },
    }
