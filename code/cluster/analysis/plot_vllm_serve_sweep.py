#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_mean(rows, x_key: str, y_key: str):
    groups = {}
    for r in rows:
        x = as_float(r.get(x_key))
        y = as_float(r.get(y_key))
        if x is None or y is None:
            continue
        groups.setdefault(x, []).append(y)
    xs = sorted(groups.keys())
    ys = [mean(groups[x]) for x in xs]
    return xs, ys


def plot_line(xs, series, out: Path, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    if not xs:
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes)
    elif len(xs) == 1:
        x0 = xs[0]
        for label, ys in series:
            if not ys:
                continue
            ax.scatter([x0], [ys[0]], label=label, s=55)
        span = max(1.0, abs(float(x0)) * 0.1)
        ax.set_xlim(x0 - span, x0 + span)
        ax.text(
            0.02,
            0.98,
            "single concurrency sample\n(run a concurrency range for a curve)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
        )
    else:
        for label, ys in series:
            ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    if len(series) > 1:
        ax.legend(fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> int:
    apply_plot_style()

    p = argparse.ArgumentParser(description="Plot vLLM bench serve concurrency sweep CSV.")
    p.add_argument("--input", required=True, help="Structured CSV from vLLM serve sweep")
    p.add_argument("--out-dir", required=True, help="Directory for output figures")
    p.add_argument("--run-id", default="run", help="Prefix for output filenames")
    args = p.parse_args()

    rows = load_rows(Path(args.input))
    out_dir = Path(args.out_dir)

    xs, total_tok = group_mean(rows, "concurrency", "total_token_throughput")
    plot_line(
        xs,
        [("total tok/s", total_tok)],
        out_dir / f"{args.run_id}_vllm_serve_total_tok_s_vs_concurrency.png",
        "vLLM serving throughput vs concurrency",
        "Tokens/sec",
    )

    xs, p50_tpot = group_mean(rows, "concurrency", "median_tpot_ms")
    _, p99_tpot = group_mean(rows, "concurrency", "p99_tpot_ms")
    plot_line(
        xs,
        [("p50 TPOT", p50_tpot), ("p99 TPOT", p99_tpot)],
        out_dir / f"{args.run_id}_vllm_serve_tpot_vs_concurrency.png",
        "vLLM serving TPOT vs concurrency",
        "Latency (ms)",
    )

    xs, p50_ttft = group_mean(rows, "concurrency", "median_ttft_ms")
    _, p99_ttft = group_mean(rows, "concurrency", "p99_ttft_ms")
    plot_line(
        xs,
        [("p50 TTFT", p50_ttft), ("p99 TTFT", p99_ttft)],
        out_dir / f"{args.run_id}_vllm_serve_ttft_vs_concurrency.png",
        "vLLM serving TTFT vs concurrency",
        "Latency (ms)",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
