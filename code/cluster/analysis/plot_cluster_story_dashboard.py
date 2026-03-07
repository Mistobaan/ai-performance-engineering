#!/usr/bin/env python3
"""Generate a single dashboard figure and node parity summary for a run."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
from plot_style import COLOR_CYCLE, apply_plot_style


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_nccl_curve(structured_dir: Path, run_id: str) -> tuple[list[float], list[float], str | None]:
    summary_path = structured_dir / f"{run_id}_health_suite_extended_node1node2_cluster_health_suite_summary.json"
    nccl_json: Path | None = None
    if summary_path.exists():
        summary = _read_json(summary_path)
        nccl_path = (
            ((summary.get("nccl") or {}).get("all_reduce_perf") or {}).get("structured_json")
        )
        if nccl_path:
            maybe = Path(str(nccl_path))
            if maybe.exists():
                nccl_json = maybe

    if nccl_json is None:
        fallback = structured_dir / f"{run_id}_2nodes_nccl.json"
        if fallback.exists():
            nccl_json = fallback

    if nccl_json is None:
        return [], [], None

    payload = _read_json(nccl_json)
    rows = sorted(payload.get("results") or [], key=lambda r: r.get("size_bytes", 0))
    sizes_gib = [float(r["size_bytes"]) / (1024.0**3) for r in rows if _as_float(r.get("busbw_gbps")) is not None]
    busbw = [_as_float(r.get("busbw_gbps")) for r in rows if _as_float(r.get("busbw_gbps")) is not None]
    return sizes_gib, [v for v in busbw if v is not None], str(nccl_json)


def _load_vllm_curve(csv_path: Path) -> tuple[list[int], list[float], list[float]]:
    if not csv_path.exists():
        return [], [], []
    rows = _read_csv(csv_path)
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"tok": [], "p99_ttft": []})
    for row in rows:
        conc = _as_float(row.get("concurrency"))
        tok = _as_float(row.get("total_token_throughput"))
        p99_ttft = _as_float(row.get("p99_ttft_ms"))
        if conc is None:
            continue
        key = int(conc)
        if tok is not None:
            grouped[key]["tok"].append(tok)
        if p99_ttft is not None:
            grouped[key]["p99_ttft"].append(p99_ttft)

    xs = sorted(grouped.keys())
    tok_y = [mean(grouped[x]["tok"]) if grouped[x]["tok"] else float("nan") for x in xs]
    ttft_y = [mean(grouped[x]["p99_ttft"]) if grouped[x]["p99_ttft"] else float("nan") for x in xs]
    return xs, tok_y, ttft_y


def _load_gemm_summary(csv_path: Path) -> dict[str, Any] | None:
    if not csv_path.exists():
        return None
    rows = _read_csv(csv_path)
    per_gpu: dict[int, float] = {}
    values: list[float] = []
    for row in rows:
        try:
            gpu = int(row.get("physical_gpu", "-1"))
        except (TypeError, ValueError):
            gpu = -1
        val = _as_float(row.get("avg_tflops"))
        if gpu < 0 or val is None:
            continue
        per_gpu[gpu] = val
        values.append(val)
    if not values:
        return None
    return {
        "per_gpu_tflops": per_gpu,
        "mean_tflops": mean(values),
        "min_tflops": min(values),
        "max_tflops": max(values),
    }


def _load_numa_local_bw(json_path: Path) -> float | None:
    if not json_path.exists():
        return None
    payload = _read_json(json_path)
    vals = [_as_float(r.get("bw_gbps")) for r in payload.get("results", [])]
    valid = [v for v in vals if v is not None]
    if not valid:
        return None
    return max(valid)


def _load_fio_seq_read(json_path: Path) -> float | None:
    if not json_path.exists():
        return None
    payload = _read_json(json_path)
    return _as_float((((payload.get("results") or {}).get("seq_read") or {}).get("bw_mb_s")))


def _plot_missing(ax: Any, title: str, reason: str) -> None:
    ax.set_title(title)
    ax.axis("off")
    ax.text(0.5, 0.5, reason, ha="center", va="center", fontsize=10)


def main() -> int:
    apply_plot_style()

    ap = argparse.ArgumentParser(description="Create a 2x2 dashboard and node parity summary for one run.")
    ap.add_argument("--run-id", required=True, help="Run id prefix (for example: 2026-02-09_gb200_fullflags_all_0117)")
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
    ap.add_argument("--fig-out", default="", help="Output PNG path (default: docs/figures/<run_id>_cluster_story_dashboard.png)")
    ap.add_argument(
        "--summary-out",
        default="",
        help="Output summary JSON path (default: results/structured/<run_id>_node_parity_summary.json)",
    )
    args = ap.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    structured_dir = Path(args.structured_dir).resolve()
    labels = [x.strip() for x in args.node_labels.split(",") if x.strip()]
    if not labels:
        raise SystemExit("ERROR: --node-labels resolved to an empty set")

    default_fig_dir = Path(os.environ.get("CLUSTER_FIGURES_DIR", root_dir / "docs/figures"))
    default_structured_dir = Path(os.environ.get("CLUSTER_RESULTS_STRUCTURED_DIR", root_dir / "results/structured"))
    fig_out = Path(args.fig_out).resolve() if args.fig_out else (default_fig_dir / f"{args.run_id}_cluster_story_dashboard.png")
    summary_out = (
        Path(args.summary_out).resolve()
        if args.summary_out
        else (default_structured_dir / f"{args.run_id}_node_parity_summary.json")
    )
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    # Networking arc input.
    nccl_x, nccl_y, nccl_source = _load_nccl_curve(structured_dir, args.run_id)
    iperf_path = structured_dir / f"{args.run_id}_iperf3_oob_tcp.json"
    oob_fwd = None
    oob_rev = None
    if iperf_path.exists():
        iperf = _read_json(iperf_path)
        oob_fwd = _as_float(((iperf.get("iperf3") or {}).get("fwd_gbps")))
        oob_rev = _as_float(((iperf.get("iperf3") or {}).get("rev_gbps")))

    # Inference arc input.
    vllm_csv = structured_dir / f"{args.run_id}_{labels[0]}_vllm_serve_sweep.csv"
    vllm_x, vllm_tok, vllm_p99_ttft = _load_vllm_curve(vllm_csv)

    # Node-level parity input.
    node_metrics: list[dict[str, Any]] = []
    for label in labels:
        gemm_path = structured_dir / f"{args.run_id}_{label}_gemm_gpu_sanity.csv"
        numa_path = structured_dir / f"{args.run_id}_{label}_numa_mem_bw.json"
        fio_path = structured_dir / f"{args.run_id}_{label}_fio.json"
        node_metrics.append(
            {
                "label": label,
                "paths": {"gemm_csv": str(gemm_path), "numa_json": str(numa_path), "fio_json": str(fio_path)},
                "gemm": _load_gemm_summary(gemm_path),
                "numa_local_bw_gbps": _load_numa_local_bw(numa_path),
                "fio_seq_read_mb_s": _load_fio_seq_read(fio_path),
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    ax_nccl, ax_vllm, ax_gemm, ax_ratio = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel 1: NCCL all-reduce bus bandwidth curve.
    if nccl_x and nccl_y:
        ax_nccl.plot(nccl_x, nccl_y, marker="o", linewidth=2.0, color=COLOR_CYCLE[0], label="all_reduce busbw")
        ax_nccl.set_xscale("log", base=2)
        ax_nccl.set_xlabel("Message size (GiB, log2)")
        ax_nccl.set_ylabel("Bus bandwidth (GB/s)")
        ax_nccl.set_title("Networking Arc: NCCL all-reduce")
        ax_nccl.legend(loc="lower right")
    else:
        _plot_missing(ax_nccl, "Networking Arc: NCCL all-reduce", "Missing NCCL all-reduce artifact")

    # Panel 2: vLLM throughput and p99 TTFT.
    if vllm_x:
        line_tok = ax_vllm.plot(vllm_x, vllm_tok, marker="o", linewidth=2.0, color=COLOR_CYCLE[2], label="Total tok/s")
        ax_vllm.set_xlabel("Concurrency")
        ax_vllm.set_ylabel("Total tok/s", color=COLOR_CYCLE[2])
        ax_vllm.tick_params(axis="y", labelcolor=COLOR_CYCLE[2])
        ax_vllm.set_title("Inference Arc: throughput vs p99 TTFT")

        ax_vllm_t = ax_vllm.twinx()
        line_ttft = ax_vllm_t.plot(
            vllm_x,
            vllm_p99_ttft,
            marker="s",
            linewidth=2.0,
            color=COLOR_CYCLE[3],
            label="p99 TTFT (ms)",
        )
        ax_vllm_t.set_ylabel("p99 TTFT (ms)", color=COLOR_CYCLE[3])
        ax_vllm_t.tick_params(axis="y", labelcolor=COLOR_CYCLE[3])

        lines = line_tok + line_ttft
        labels_legend = [l.get_label() for l in lines]
        ax_vllm.legend(lines, labels_legend, loc="upper left")
    else:
        _plot_missing(ax_vllm, "Inference Arc: throughput vs p99 TTFT", "Missing vLLM sweep CSV")

    # Panel 3: per-GPU GEMM by node.
    gemm_nodes = [node for node in node_metrics if node.get("gemm")]
    if gemm_nodes:
        gpu_ids = sorted(
            {
                gpu
                for node in gemm_nodes
                for gpu in ((node.get("gemm") or {}).get("per_gpu_tflops") or {}).keys()
            }
        )
        width = 0.8 / max(len(gemm_nodes), 1)
        x_pos = list(range(len(gpu_ids)))
        for idx, node in enumerate(gemm_nodes):
            vals = [((node["gemm"]["per_gpu_tflops"]).get(gpu)) for gpu in gpu_ids]
            offsets = [x - 0.4 + width / 2.0 + idx * width for x in x_pos]
            ax_gemm.bar(offsets, vals, width=width, label=node["label"])
        ax_gemm.set_xticks(x_pos)
        ax_gemm.set_xticklabels([f"GPU{gpu}" for gpu in gpu_ids])
        ax_gemm.set_ylabel("avg TFLOPS (BF16)")
        ax_gemm.set_title("Node parity: GEMM per-GPU")
        ax_gemm.legend(loc="lower right")
    else:
        _plot_missing(ax_gemm, "Node parity: GEMM per-GPU", "Missing GEMM per-GPU CSVs")

    # Panel 4: node2/node1 parity ratios for available metrics.
    ratio_payload: dict[str, float] = {}
    ratio_labels: list[str] = []
    ratio_values: list[float] = []
    ratio_note = ""
    if len(node_metrics) >= 2:
        n1 = node_metrics[0]
        n2 = node_metrics[1]

        g1 = ((n1.get("gemm") or {}).get("mean_tflops"))
        g2 = ((n2.get("gemm") or {}).get("mean_tflops"))
        if g1 and g2:
            ratio = g2 / g1
            ratio_labels.append("GEMM mean")
            ratio_values.append(ratio)
            ratio_payload["gemm_mean_tflops"] = ratio

        numa1 = n1.get("numa_local_bw_gbps")
        numa2 = n2.get("numa_local_bw_gbps")
        if numa1 and numa2:
            ratio = numa2 / numa1
            ratio_labels.append("NUMA local BW")
            ratio_values.append(ratio)
            ratio_payload["numa_local_bw_gbps"] = ratio

        fio1 = n1.get("fio_seq_read_mb_s")
        fio2 = n2.get("fio_seq_read_mb_s")
        if fio1 and fio2:
            ratio = fio2 / fio1
            ratio_labels.append("FIO seq read")
            ratio_values.append(ratio)
            ratio_payload["fio_seq_read_mb_s"] = ratio
        elif fio1 is not None and fio2 is None:
            ratio_note = f"{n2['label']} fio seq-read missing"

    if ratio_labels:
        bars = ax_ratio.bar(ratio_labels, ratio_values, color=[COLOR_CYCLE[0], COLOR_CYCLE[1], COLOR_CYCLE[2]][: len(ratio_labels)])
        ax_ratio.axhline(1.0, color="#555555", linewidth=1.2, linestyle=":")
        ax_ratio.set_ylabel("node2 / node1 ratio")
        ax_ratio.set_title("Parity ratios (first two nodes)")
        ymax = max(max(ratio_values) * 1.15, 1.05)
        ax_ratio.set_ylim(0.0, ymax)
        for bar, val in zip(bars, ratio_values, strict=True):
            ax_ratio.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + ymax * 0.02,
                f"{val:.3f}x",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if ratio_note:
            ax_ratio.text(0.5, 0.02, ratio_note, transform=ax_ratio.transAxes, ha="center", va="bottom", fontsize=8)
    else:
        missing_reason = "Insufficient node parity data"
        if ratio_note:
            missing_reason = f"{missing_reason}\n({ratio_note})"
        _plot_missing(ax_ratio, "Parity ratios (first two nodes)", missing_reason)

    fig.suptitle(f"Cluster Story Dashboard: {args.run_id}", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(fig_out)
    plt.close(fig)

    nccl_peak = max(nccl_y) if nccl_y else None
    max_tok = None
    max_tok_conc = None
    max_tok_p99_ttft = None
    if vllm_x and vllm_tok:
        pairs = [(tok, conc, ttft) for conc, tok, ttft in zip(vllm_x, vllm_tok, vllm_p99_ttft, strict=True) if tok == tok]
        if pairs:
            tok, conc, ttft = max(pairs, key=lambda x: x[0])
            max_tok = tok
            max_tok_conc = conc
            max_tok_p99_ttft = ttft

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": args.run_id,
        "inputs": {
            "structured_dir": str(structured_dir),
            "nccl_source_json": nccl_source,
            "iperf3_oob_json": str(iperf_path) if iperf_path.exists() else None,
            "vllm_csv": str(vllm_csv) if vllm_csv.exists() else None,
            "node_labels": labels,
        },
        "networking": {
            "nccl_all_reduce_peak_busbw_gbps": nccl_peak,
            "oob_tcp_fwd_gbps": oob_fwd,
            "oob_tcp_rev_gbps": oob_rev,
        },
        "inference": {
            "max_total_token_throughput": max_tok,
            "concurrency_at_max_throughput": max_tok_conc,
            "p99_ttft_ms_at_max_throughput": max_tok_p99_ttft,
        },
        "nodes": node_metrics,
        "node2_over_node1_ratios": ratio_payload,
    }
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote dashboard figure: {fig_out}")
    print(f"Wrote parity summary: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
