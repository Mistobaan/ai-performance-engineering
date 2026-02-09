#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def load_data(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases") or data.get("results") or []
    if not cases:
        raise SystemExit("No cases/results found in JSON. Expected key 'cases' or 'results'.")
    return data, _enrich_cases(data, cases)


def _enrich_cases(payload, cases):
    hosts = payload.get("hosts") or []
    default_nodes = len(hosts) if hosts else None
    default_gpus = payload.get("total_ranks")
    if default_gpus is None and payload.get("gpus_per_node") and default_nodes:
        try:
            default_gpus = int(payload.get("gpus_per_node")) * int(default_nodes)
        except (TypeError, ValueError):
            default_gpus = None
    default_label = payload.get("run_id")

    enriched = []
    for row in cases:
        if not isinstance(row, dict):
            continue
        out = dict(row)
        if out.get("nodes") in (None, "") and default_nodes is not None:
            out["nodes"] = int(default_nodes)
        if out.get("gpus") in (None, "") and default_gpus is not None:
            out["gpus"] = int(default_gpus)
        if out.get("label") in (None, "") and default_label:
            out["label"] = str(default_label)
        enriched.append(out)
    return enriched


def group_cases(cases, key_fields):
    groups = {}
    for row in cases:
        key = tuple(row.get(k) for k in key_fields)
        groups.setdefault(key, []).append(row)
    return groups


def _get_nbytes(row):
    return row.get("nbytes") or row.get("size_bytes") or 0


def plot_bw_vs_msg(cases, out_path: Path, title: str):
    key_fields = ["label", "nodes", "gpus"]
    groups = group_cases(cases, key_fields)

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, rows in groups.items():
        rows = sorted(rows, key=_get_nbytes)
        x_mb = [_get_nbytes(r) / (1024 * 1024) for r in rows]
        y = [r.get("busbw_gbps") or r.get("busbw") or r.get("algbw_gbps") for r in rows]
        label_parts = []
        if key[0]:
            label_parts.append(str(key[0]))
        if key[1]:
            label_parts.append(f"{key[1]} nodes")
        if key[2]:
            label_parts.append(f"{key[2]} gpus")
        label = ", ".join(label_parts) or "case"
        ax.plot(x_mb, y, marker="o", label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Message size (MB, log scale)")
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_scaling_efficiency(cases, out_path: Path, title: str, target_sizes_mb):
    # Expect entries with fields: nbytes/size_bytes, busbw_gbps/busbw, gpus, nodes.
    # Efficiency is normalized to the smallest available GPU-count point.
    fig, ax = plt.subplots(figsize=(8, 5))

    any_series = False
    for target_mb in target_sizes_mb:
        target_bytes = target_mb * 1024 * 1024
        # pick closest size sample per GPU count
        by_gpu = {}
        for r in cases:
            if _get_nbytes(r) in (None, 0):
                continue
            gpus = r.get("gpus")
            if gpus in (None, ""):
                continue
            try:
                gpus = int(gpus)
            except (TypeError, ValueError):
                continue
            by_gpu.setdefault(gpus, []).append(r)

        points = []
        for gpus, rows in by_gpu.items():
            rows = sorted(rows, key=lambda r: abs(_get_nbytes(r) - target_bytes))
            chosen = dict(rows[0])
            chosen["gpus"] = gpus
            points.append(chosen)

        if not points:
            continue

        baseline = sorted(points, key=lambda r: int(r.get("gpus") or 0))[0]
        base_bw = baseline.get("busbw_gbps") or baseline.get("busbw") or baseline.get("algbw_gbps")
        base_gpus = baseline.get("gpus") or 1
        if not base_bw:
            continue

        xs = []
        ys = []
        for p in sorted(points, key=lambda r: (r.get("nodes") or 1, r.get("gpus") or 1)):
            gpus = p.get("gpus") or 1
            bw = p.get("busbw_gbps") or p.get("busbw") or p.get("algbw_gbps")
            if not bw:
                continue
            scale = float(gpus) / float(base_gpus)
            if scale <= 0:
                continue
            eff = bw / (base_bw * scale)
            xs.append(gpus)
            ys.append(eff)

        if xs:
            any_series = True
            ax.plot(xs, ys, marker="o", label=f"~{target_mb}MB")

    ax.set_xlabel("GPU count")
    ax.set_ylabel("Scaling efficiency vs smallest GPU-count sample")
    ax.set_title(title)
    ax.set_ylim(0, 1.2)
    ax.grid(True, linestyle="--", alpha=0.4)
    if any_series:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No scaling points available", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    apply_plot_style()

    parser = argparse.ArgumentParser(description="Plot NCCL results.")
    parser.add_argument("--input", required=True, help="Path to structured NCCL JSON")
    parser.add_argument(
        "--baseline-input",
        default="",
        help="Optional single-node NCCL JSON used to build multi-point scaling efficiency curves",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for output figures")
    parser.add_argument("--run-id", default="run", help="Run id prefix for file names")
    parser.add_argument("--sizes-mb", default="1,16,64", help="Comma-separated message sizes for efficiency plot")
    args = parser.parse_args()

    data_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, cases = load_data(data_path)
    if args.baseline_input:
        base_path = Path(args.baseline_input)
        if base_path.exists():
            _, baseline_cases = load_data(base_path)
            cases = cases + baseline_cases

    bw_out = out_dir / f"{args.run_id}_nccl_bw_vs_msg.png"
    eff_out = out_dir / f"{args.run_id}_nccl_scaling_efficiency.png"

    plot_bw_vs_msg(cases, bw_out, "NCCL all-reduce bus bandwidth vs message size")

    sizes = [int(s.strip()) for s in args.sizes_mb.split(",") if s.strip()]
    plot_scaling_efficiency(cases, eff_out, "NCCL scaling efficiency", sizes)


if __name__ == "__main__":
    main()
