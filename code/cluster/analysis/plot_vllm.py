#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        raise SystemExit("No rows found in CSV.")
    return rows


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def filter_rows(rows, prompt_len, gen_len):
    if prompt_len is None and gen_len is None:
        return rows
    filtered = []
    for r in rows:
        p = as_float(r.get("prompt_len"))
        g = as_float(r.get("gen_len"))
        if prompt_len is not None and p != prompt_len:
            continue
        if gen_len is not None and g != gen_len:
            continue
        filtered.append(r)
    return filtered


def group_by_concurrency(rows):
    groups = {}
    for r in rows:
        conc = as_float(r.get("concurrency") or r.get("concurrent"))
        if conc is None:
            continue
        groups.setdefault(conc, []).append(r)
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def extract_metric(row, keys):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return as_float(row[k])
    return None


def plot_throughput(groups, out_path: Path, title: str):
    xs, ys = [], []
    for conc, rows in groups.items():
        vals = []
        for r in rows:
            v = extract_metric(
                r,
                [
                    "tok_per_s",
                    "toks_per_s",
                    "tokens_per_s",
                    "throughput_tok_s",
                    "output_throughput",
                    "total_token_throughput",
                ],
            )
            if v is not None:
                vals.append(v)
        if vals:
            xs.append(conc)
            ys.append(mean(vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Tokens/sec")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_latency(groups, out_path: Path, title: str):
    xs, p50s, p99s = [], [], []
    for conc, rows in groups.items():
        p50_vals, p99_vals = [], []
        for r in rows:
            p50 = extract_metric(r, ["p50_ms", "p50", "latency_p50_ms", "median_ttft_ms", "mean_ttft_ms"])
            p99 = extract_metric(r, ["p99_ms", "p99", "latency_p99_ms", "p99_ttft_ms"])
            if p50 is not None:
                p50_vals.append(p50)
            if p99 is not None:
                p99_vals.append(p99)
        if p50_vals or p99_vals:
            xs.append(conc)
            p50s.append(mean(p50_vals) if p50_vals else None)
            p99s.append(mean(p99_vals) if p99_vals else None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, p50s, marker="o", label="p50")
    ax.plot(xs, p99s, marker="o", label="p99")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    apply_plot_style()
    parser = argparse.ArgumentParser(description="Plot vLLM benchmark results.")
    parser.add_argument("--input", required=True, help="Path to structured vLLM CSV")
    parser.add_argument("--out-dir", required=True, help="Directory for output figures")
    parser.add_argument("--run-id", default="run", help="Run id prefix for file names")
    parser.add_argument("--prompt-len", type=int, default=None, help="Filter by prompt length")
    parser.add_argument("--gen-len", type=int, default=None, help="Filter by generation length")
    args = parser.parse_args()

    data_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(data_path)
    rows = filter_rows(rows, args.prompt_len, args.gen_len)
    groups = group_by_concurrency(rows)

    thr_out = out_dir / f"{args.run_id}_vllm_toks_vs_concurrency.png"
    lat_out = out_dir / f"{args.run_id}_vllm_latency_vs_concurrency.png"

    plot_throughput(groups, thr_out, "vLLM throughput vs concurrency")
    plot_latency(groups, lat_out, "vLLM latency vs concurrency")


if __name__ == "__main__":
    main()
