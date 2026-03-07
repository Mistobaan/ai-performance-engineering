#!/usr/bin/env python3
"""Generate a compact operator-check dashboard (quick friction + monitoring expectations).

This figure is intended for stakeholder-facing field reports: it provides a quick view of
missing tools/permissions, per-check durations, and (for monitoring expectations) whether
Kubernetes control-plane checks were enabled or intentionally skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style


STATUS_COLORS = {
    "ok": "#2CA02C",  # green
    "degraded": "#FF7F0E",  # orange
    "error": "#D62728",  # red
    "failed": "#D62728",  # red
    "not_applicable": "#8C8C8C",  # gray
    "skipped": "#8C8C8C",  # gray
    "missing": "#8C8C8C",  # gray
    "unknown": "#8C8C8C",  # gray
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _status_color(status: str) -> str:
    return STATUS_COLORS.get(status, STATUS_COLORS["unknown"])


def _first_json_obj(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    # speedtest tools usually print a single JSON object; try a few robust parses.
    for candidate in (text.strip().splitlines()[:1] + [text.strip()]):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    m = re.search(r"(\{.*\})", text.strip(), flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _parse_speedtest_mbps(stdout_excerpt: str) -> dict[str, float] | None:
    payload = _first_json_obj(stdout_excerpt)
    if not isinstance(payload, dict):
        return None

    # speedtest-cli: download/upload are bits/sec; ping is ms.
    if "download" in payload and "upload" in payload and "ping" in payload:
        dl = _as_float(payload.get("download"))
        ul = _as_float(payload.get("upload"))
        ping = _as_float(payload.get("ping"))
        if dl > 0 and ul > 0:
            return {
                "download_mbps": dl / 1e6,
                "upload_mbps": ul / 1e6,
                "ping_ms": ping,
            }

    # Ookla speedtest: bandwidth is bytes/sec.
    if isinstance(payload.get("download"), dict) and isinstance(payload.get("upload"), dict):
        dl_bps = _as_float((payload.get("download") or {}).get("bandwidth")) * 8.0
        ul_bps = _as_float((payload.get("upload") or {}).get("bandwidth")) * 8.0
        ping = _as_float(((payload.get("ping") or {}).get("latency")))
        if dl_bps > 0 and ul_bps > 0:
            return {
                "download_mbps": dl_bps / 1e6,
                "upload_mbps": ul_bps / 1e6,
                "ping_ms": ping,
            }

    return None


@dataclass(frozen=True)
class CheckRow:
    name: str
    duration_sec: float
    status: str
    note: str = ""


def _load_quick_friction(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    checks = payload.get("checks") or []
    by_name = {c.get("name"): c for c in checks if isinstance(c, dict) and c.get("name")}
    order = payload.get("selected_checks") or [c.get("name") for c in checks if isinstance(c, dict)]
    rows: list[CheckRow] = []
    speedtest_summary = None
    for name in order:
        rec = by_name.get(name) or {}
        status = str(rec.get("status") or "missing")
        dur = _as_float(rec.get("duration_sec"))
        note = ""
        if name == "speedtest":
            speed = _parse_speedtest_mbps(str(rec.get("stdout_excerpt") or ""))
            if speed:
                speedtest_summary = speed
                note = f"{speed['download_mbps']:.0f}/{speed['upload_mbps']:.0f} Mbps, {speed['ping_ms']:.1f} ms"
        rows.append(CheckRow(name=name, duration_sec=dur, status=status, note=note))

    return {
        "path": str(path),
        "status": payload.get("status", "unknown"),
        "failed_checks": payload.get("failed_checks") or [],
        "rows": rows,
        "speedtest": speedtest_summary,
        "tool_paths": payload.get("tool_paths") or {},
    }


def _load_monitoring_expectations(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    checks = payload.get("checks") or []
    by_name = {c.get("name"): c for c in checks if isinstance(c, dict) and c.get("name")}

    selected = payload.get("selected_checks") or []
    skipped = set(payload.get("skipped_checks") or [])
    effective = payload.get("effective_checks") or []
    if selected:
        order = [x for x in selected if x]
    elif effective:
        order = [x for x in effective if x]
    else:
        order = [c.get("name") for c in checks if isinstance(c, dict) and c.get("name")]

    rows: list[CheckRow] = []
    for name in order:
        if name in skipped and name not in by_name:
            rows.append(CheckRow(name=name, duration_sec=0.0, status="not_applicable", note="skipped"))
            continue
        rec = by_name.get(name) or {}
        rows.append(
            CheckRow(
                name=name,
                duration_sec=_as_float(rec.get("duration_sec")),
                status=str(rec.get("status") or "missing"),
            )
        )

    return {
        "path": str(path),
        "status": payload.get("status", "unknown"),
        "k8s": payload.get("k8s") or {},
        "categories": payload.get("categories") or {},
        "rows": rows,
        "skipped_checks": payload.get("skipped_checks") or [],
        "effective_checks": payload.get("effective_checks") or [],
    }


def _plot_check_rows(ax: Any, title: str, rows: list[CheckRow] | None) -> None:
    ax.set_title(title)
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "Missing artifact / no checks", ha="center", va="center")
        return

    y = np.arange(len(rows))
    durs = np.array([max(0.0, r.duration_sec) for r in rows], dtype=float)
    colors = [_status_color(r.status) for r in rows]

    ax.barh(y, durs, color=colors, edgecolor="#333333", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([r.name for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Duration (s)")

    max_d = float(np.max(durs)) if len(durs) else 0.0
    ax.set_xlim(0.0, max(1.0, max_d * 1.35))

    xoff = max(0.02, max_d * 0.02)
    for i, r in enumerate(rows):
        label = r.status
        if r.note:
            label = f"{label} ({r.note})"
        ax.text(durs[i] + xoff, i, label, va="center", fontsize=8)


def main() -> int:
    apply_plot_style()

    ap = argparse.ArgumentParser(description="Plot operator-check dashboard for a run id.")
    ap.add_argument("--run-id", required=True, help="Run id prefix (example: 2026-02-10_full_suite_e2e_wire_qf_mon)")
    ap.add_argument(
        "--structured-dir",
        default="results/structured",
        help="Path to structured results directory (default: results/structured)",
    )
    ap.add_argument(
        "--node-labels",
        default="node1,node2",
        help="Comma-separated node labels used in structured filenames (default: node1,node2)",
    )
    ap.add_argument(
        "--fig-out",
        default="",
        help="Output PNG path (default: docs/figures/<run_id>_operator_checks_dashboard.png)",
    )
    ap.add_argument(
        "--summary-out",
        default="",
        help="Output summary JSON path (default: results/structured/<run_id>_operator_checks_dashboard.json)",
    )
    args = ap.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    structured_dir = Path(args.structured_dir).resolve()
    labels = [x.strip() for x in args.node_labels.split(",") if x.strip()]
    if not labels:
        raise SystemExit("ERROR: --node-labels resolved to an empty set")

    default_fig_dir = Path(os.environ.get("CLUSTER_FIGURES_DIR", root_dir / "docs/figures"))
    default_structured_dir = Path(os.environ.get("CLUSTER_RESULTS_STRUCTURED_DIR", root_dir / "results/structured"))
    fig_out = Path(args.fig_out).resolve() if args.fig_out else (default_fig_dir / f"{args.run_id}_operator_checks_dashboard.png")
    summary_out = (
        Path(args.summary_out).resolve()
        if args.summary_out
        else (default_structured_dir / f"{args.run_id}_operator_checks_dashboard.json")
    )
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    quick: dict[str, Any] = {}
    mon: dict[str, Any] = {}
    for label in labels:
        q_path = structured_dir / f"{args.run_id}_{label}_quick_friction.json"
        m_path = structured_dir / f"{args.run_id}_{label}_monitoring_expectations.json"
        quick[label] = _load_quick_friction(q_path)
        mon[label] = _load_monitoring_expectations(m_path)

    summary = {
        "run_id": args.run_id,
        "node_labels": labels,
        "sources": {
            "quick_friction": {k: (v or {}).get("path") for k, v in quick.items()},
            "monitoring_expectations": {k: (v or {}).get("path") for k, v in mon.items()},
        },
        "quick_friction": {
            k: {
                "status": (v or {}).get("status", "missing"),
                "failed_checks": (v or {}).get("failed_checks", []),
                "speedtest": (v or {}).get("speedtest"),
                "tool_paths": (v or {}).get("tool_paths", {}),
            }
            for k, v in quick.items()
        },
        "monitoring_expectations": {
            k: {
                "status": (v or {}).get("status", "missing"),
                "k8s": (v or {}).get("k8s", {}),
                "skipped_checks": (v or {}).get("skipped_checks", []),
                "effective_checks": (v or {}).get("effective_checks", []),
                "categories": (v or {}).get("categories", {}),
            }
            for k, v in mon.items()
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    n = len(labels)
    fig_h = max(6.5, 2.9 * n + 2.0)
    fig, axes = plt.subplots(n, 2, figsize=(14.2, fig_h), squeeze=False)
    fig.suptitle(f"Operator Checks Dashboard ({args.run_id})", fontsize=14, y=0.995)

    for i, label in enumerate(labels):
        q = quick.get(label) or {}
        m = mon.get(label) or {}

        q_status = q.get("status", "missing")
        m_status = m.get("status", "missing")
        k8s = m.get("k8s") or {}
        k8s_mode = k8s.get("mode", "unknown")
        k8s_detected = k8s.get("detected", "unknown")
        k8s_cp = k8s.get("control_plane_enabled", "unknown")

        q_title = f"{label}: quick_friction (status={q_status})"
        m_title = f"{label}: monitoring_expectations (status={m_status}, k8s_mode={k8s_mode}, detected={k8s_detected}, cp_enabled={k8s_cp})"
        _plot_check_rows(axes[i][0], q_title, q.get("rows"))
        _plot_check_rows(axes[i][1], m_title, m.get("rows"))

    legend = [
        mpatches.Patch(color=_status_color("ok"), label="ok"),
        mpatches.Patch(color=_status_color("degraded"), label="degraded"),
        mpatches.Patch(color=_status_color("error"), label="error"),
        mpatches.Patch(color=_status_color("not_applicable"), label="not_applicable/skipped"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
    fig.savefig(fig_out)
    plt.close(fig)

    print(fig_out)
    print(summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
