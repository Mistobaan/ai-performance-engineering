#!/usr/bin/env python3
"""Build a unified cluster scorecard + bottleneck classification artifact."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _num_or_none(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _extract_label(path: Path, run_id: str, suffix: str) -> str | None:
    name = path.name
    prefix = f"{run_id}_"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix) : -len(suffix)]


def _detect_labels(structured_dir: Path, run_id: str) -> List[str]:
    suffixes = [
        "_gemm_gpu_sanity.csv",
        "_vllm_serve_sweep.csv",
        "_vllm_serve_slo_goodput.json",
        "_vllm_serve_sweep_stability.json",
        "_vllm_serve_request_rate_sweep.csv",
        "_vllm_serve_request_rate_sweep_stability.json",
        "_nvbandwidth.json",
        "_gpu_stream.json",
        "_fio.json",
        "_fio_stability.json",
    ]
    labels: set[str] = set()
    for suffix in suffixes:
        for p in structured_dir.glob(f"{run_id}_*{suffix}"):
            lbl = _extract_label(p, run_id, suffix)
            if lbl:
                labels.add(lbl)
    return sorted(labels)


def _peak_nccl(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {"peak_algbw_gbps": 0.0, "peak_busbw_gbps": 0.0}
    payload = _load_json(path)
    rows = payload.get("results") or []
    peak_algbw = max((_float(r.get("algbw_gbps")) for r in rows), default=0.0)
    peak_busbw = max((_float(r.get("busbw_gbps")) for r in rows), default=0.0)
    return {"peak_algbw_gbps": peak_algbw, "peak_busbw_gbps": peak_busbw}


def _allreduce_stability_metrics(path: Path) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "allreduce_applicable": False,
        "allreduce_busbw_mean_gbps": None,
        "allreduce_busbw_cv_pct": None,
        "allreduce_p99_p50_ratio": None,
        "allreduce_jitter_assessment": "n/a",
    }
    if not path.exists():
        return base
    payload = _load_json(path)
    world_size_raw = payload.get("world_size")
    if world_size_raw is not None and int(_float(world_size_raw, 0.0)) <= 1:
        return {**base, "allreduce_jitter_assessment": "n/a (world_size<=1)"}
    summary = payload.get("summary") or {}
    return {
        "allreduce_applicable": True,
        "allreduce_busbw_mean_gbps": _float(summary.get("busbw_mean_gbps")),
        "allreduce_busbw_cv_pct": _float(summary.get("busbw_cv_pct")),
        "allreduce_p99_p50_ratio": _float(summary.get("p99_p50_ratio")),
        "allreduce_jitter_assessment": str(summary.get("jitter_assessment") or "unknown"),
    }


def _nccl_algo_metrics(path: Path) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "nccl_algo_applicable": False,
        "nccl_algo_best": None,
        "nccl_algo_peak_busbw_gbps": None,
        "nccl_algo_spread_pct": None,
        "nccl_algo_auto_gap_pct": None,
    }
    if not path.exists():
        return base
    payload = _load_json(path)
    total_ranks_raw = payload.get("total_ranks")
    if total_ranks_raw is not None and int(_float(total_ranks_raw, 0.0)) <= 1:
        return {**base, "nccl_algo_best": "n/a (single-rank)"}
    rows = payload.get("algorithms_tested") or []
    ok_rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    if not ok_rows:
        return {**base, "nccl_algo_applicable": True}
    peaks: List[Tuple[str, float]] = []
    for row in ok_rows:
        algo = str(row.get("algo") or "")
        peak = _float(row.get("peak_busbw_gbps"))
        if algo and peak > 0:
            peaks.append((algo, peak))
    if not peaks:
        return {**base, "nccl_algo_applicable": True}
    peaks.sort(key=lambda x: x[1], reverse=True)
    best_algo, best_peak = peaks[0]
    worst_peak = peaks[-1][1]
    spread_pct = ((best_peak - worst_peak) / best_peak * 100.0) if best_peak > 0 else 0.0

    auto_peak = 0.0
    for algo, peak in peaks:
        if algo.lower() == "auto":
            auto_peak = peak
            break
    auto_gap_pct = ((best_peak - auto_peak) / best_peak * 100.0) if best_peak > 0 and auto_peak > 0 else 0.0

    return {
        "nccl_algo_applicable": True,
        "nccl_algo_best": best_algo,
        "nccl_algo_peak_busbw_gbps": best_peak,
        "nccl_algo_spread_pct": spread_pct,
        "nccl_algo_auto_gap_pct": auto_gap_pct,
    }


def _coverage_metrics(path: Path) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "coverage_score_pct": None,
        "advanced_coverage_score_pct": None,
        "coverage_maturity": None,
    }
    if not path.exists():
        return base
    payload = _load_json(path)
    return {
        "coverage_score_pct": int(_float(payload.get("coverage_score_pct"), 0.0)),
        "advanced_coverage_score_pct": int(_float(payload.get("advanced_coverage_score_pct"), 0.0)),
        "coverage_maturity": str(payload.get("coverage_maturity") or ""),
    }


def _mlperf_metrics(path: Path) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "overall_status": "",
        "inference_track_ready": False,
        "training_track_ready": False,
    }
    if not path.exists():
        return base
    payload = _load_json(path)
    return {
        "overall_status": str(payload.get("overall_status") or ""),
        "inference_track_ready": bool(payload.get("inference_track_ready")),
        "training_track_ready": bool(payload.get("training_track_ready")),
    }


def _canonical_gates(
    coverage_score_pct: int | None,
    advanced_coverage_score_pct: int | None,
    mlperf_overall_status: str,
    min_coverage_pct: int = 100,
    min_advanced_pct: int = 85,
) -> Dict[str, Any]:
    cov = int(coverage_score_pct or 0)
    adv = int(advanced_coverage_score_pct or 0)
    status = (mlperf_overall_status or "").strip().lower()
    mlperf_ok = status in {"aligned", "inference_ready_only", "training_ready_only"}
    coverage_ok = cov >= min_coverage_pct
    advanced_ok = adv >= min_advanced_pct
    return {
        "coverage_min_pct": min_coverage_pct,
        "advanced_min_pct": min_advanced_pct,
        "coverage_min_met": coverage_ok,
        "advanced_min_met": advanced_ok,
        "mlperf_alignment_min_met": mlperf_ok,
        "canonical_complete": coverage_ok and advanced_ok and mlperf_ok,
    }


def _allreduce_latency_comp_metrics(path: Path) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "allreduce_latency_comp_ratio": None,
        "allreduce_latency_comp_one_large_ms": None,
        "allreduce_latency_comp_many_small_ms": None,
    }
    if not path.exists():
        return base
    payload = _load_json(path)
    comp = payload.get("comparison") or {}
    return {
        "allreduce_latency_comp_ratio": _num_or_none(comp.get("duration_ratio_small_over_large")),
        "allreduce_latency_comp_one_large_ms": _num_or_none(comp.get("one_large_duration_ms_mean")),
        "allreduce_latency_comp_many_small_ms": _num_or_none(comp.get("many_small_duration_ms_mean")),
    }


def _allgather_control_plane_metrics(path: Path) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "allgather_obj_vs_tensor_speedup": None,
        "allgather_obj_vs_allreduce_speedup": None,
        "allgather_fastest_method": None,
        "allgather_fastest_latency_ms": None,
    }
    if not path.exists():
        return base
    payload = _load_json(path)
    comp = payload.get("comparison") or {}
    return {
        "allgather_obj_vs_tensor_speedup": _num_or_none(comp.get("all_gather_object_over_all_gather_tensor_speedup")),
        "allgather_obj_vs_allreduce_speedup": _num_or_none(comp.get("all_gather_object_over_all_reduce_tensor_speedup")),
        "allgather_fastest_method": str(comp.get("fastest_method") or ""),
        "allgather_fastest_latency_ms": _num_or_none(comp.get("fastest_latency_ms")),
    }


def _label_metrics(structured_dir: Path, run_id: str, label: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"label": label}

    gemm_csv = structured_dir / f"{run_id}_{label}_gemm_gpu_sanity.csv"
    if gemm_csv.exists():
        rows = _read_csv_rows(gemm_csv)
        tflops = [_float(r.get("avg_tflops")) for r in rows if _float(r.get("avg_tflops")) > 0]
        out["gemm_max_tflops"] = max(tflops) if tflops else 0.0
        out["gemm_avg_tflops"] = (sum(tflops) / len(tflops)) if tflops else 0.0

    gpu_stream_json = structured_dir / f"{run_id}_{label}_gpu_stream.json"
    if gpu_stream_json.exists():
        payload = _load_json(gpu_stream_json)
        ops = {str(r.get("operation")): _float(r.get("bandwidth_gbps")) for r in (payload.get("operations") or [])}
        out["gpu_stream_peak_gbps"] = _float(payload.get("peak_bandwidth_gbps"))
        out["gpu_stream_triad_gbps"] = ops.get("triad", 0.0)
        out["gpu_stream_copy_gbps"] = ops.get("copy", 0.0)

    nvbw_json = structured_dir / f"{run_id}_{label}_nvbandwidth.json"
    if nvbw_json.exists():
        payload = _load_json(nvbw_json)
        key = payload.get("key_sum_gbps") or {}
        out["nvbandwidth_hbm_gbps"] = max(
            _float(key.get("device_to_device_memcpy_read_ce")),
            _float(key.get("device_to_device_memcpy_write_ce")),
        )
        out["nvbandwidth_pcie_h2d_gbps"] = _float(key.get("host_to_device_memcpy_ce"))
        out["nvbandwidth_pcie_d2h_gbps"] = _float(key.get("device_to_host_memcpy_ce"))
        out["nvbandwidth_peak_sum_gbps"] = _float(payload.get("peak_sum_gbps"))

    vllm_csv = structured_dir / f"{run_id}_{label}_vllm_serve_sweep.csv"
    if vllm_csv.exists():
        rows = _read_csv_rows(vllm_csv)
        parsed = []
        for row in rows:
            conc = int(_float(row.get("concurrency"), 0))
            parsed.append(
                {
                    "concurrency": conc,
                    "tp": int(_float(row.get("tp"), 0)),
                    "total_tok_s": _float(row.get("total_token_throughput")),
                    "p99_ttft_ms": _float(row.get("p99_ttft_ms")),
                    "p99_tpot_ms": _float(row.get("p99_tpot_ms")),
                    "gpu_power_mean_w": _float(row.get("gpu_power_mean_w")),
                }
            )
        parsed = [r for r in parsed if r["concurrency"] > 0 and r["total_tok_s"] > 0]
        parsed.sort(key=lambda r: r["concurrency"])
        if parsed:
            first = parsed[0]
            last = parsed[-1]
            best = max(parsed, key=lambda r: r["total_tok_s"])
            out["vllm_min_concurrency"] = first["concurrency"]
            out["vllm_max_concurrency"] = last["concurrency"]
            out["vllm_tok_s_min_concurrency"] = first["total_tok_s"]
            out["vllm_tok_s_max_concurrency"] = last["total_tok_s"]
            out["vllm_throughput_gain_ratio"] = (
                (last["total_tok_s"] / first["total_tok_s"]) if first["total_tok_s"] > 0 else 0.0
            )
            out["vllm_p99_ttft_min_ms"] = first["p99_ttft_ms"]
            out["vllm_p99_ttft_max_ms"] = last["p99_ttft_ms"]
            out["vllm_p99_ttft_ratio"] = (
                (last["p99_ttft_ms"] / first["p99_ttft_ms"]) if first["p99_ttft_ms"] > 0 else 0.0
            )
            out["vllm_best_total_tok_s"] = best["total_tok_s"]
            out["vllm_best_gpu_power_mean_w"] = best["gpu_power_mean_w"]
            out["vllm_best_tp"] = best["tp"]
            out["vllm_best_tok_per_joule"] = (
                (best["total_tok_s"] / best["gpu_power_mean_w"]) if best["gpu_power_mean_w"] > 0 else 0.0
            )

    vllm_slo_json = structured_dir / f"{run_id}_{label}_vllm_serve_slo_goodput.json"
    if vllm_slo_json.exists():
        payload = _load_json(vllm_slo_json)
        summary = payload.get("summary") or {}
        out["vllm_slo_max_goodput_tok_s"] = _float(summary.get("max_goodput_tok_s"))
        out["vllm_slo_max_goodput_req_s"] = _float(summary.get("max_goodput_req_s"))
        out["vllm_slo_goodput_efficiency_tok_ratio"] = _float(summary.get("goodput_efficiency_tok_ratio"))
        out["vllm_slo_knee_concurrency"] = _float(summary.get("knee_concurrency"))

    vllm_stability_json = structured_dir / f"{run_id}_{label}_vllm_serve_sweep_stability.json"
    if vllm_stability_json.exists():
        payload = _load_json(vllm_stability_json)
        summary = payload.get("summary") or {}
        out["vllm_conc_total_tok_cv_pct_p95"] = _num_or_none(summary.get("total_token_throughput_cv_pct_p95"))
        out["vllm_conc_p99_ttft_cv_pct_p95"] = _num_or_none(summary.get("p99_ttft_cv_pct_p95"))
        out["vllm_conc_p99_tpot_cv_pct_p95"] = _num_or_none(summary.get("p99_tpot_cv_pct_p95"))

    vllm_rate_csv = structured_dir / f"{run_id}_{label}_vllm_serve_request_rate_sweep.csv"
    if vllm_rate_csv.exists():
        rows = _read_csv_rows(vllm_rate_csv)
        parsed = []
        for row in rows:
            rate = _float(row.get("request_rate"), 0.0)
            parsed.append(
                {
                    "request_rate": rate,
                    "tp": int(_float(row.get("tp"), 0)),
                    "total_tok_s": _float(row.get("total_token_throughput")),
                    "p99_ttft_ms": _float(row.get("p99_ttft_ms")),
                    "p99_tpot_ms": _float(row.get("p99_tpot_ms")),
                    "gpu_power_mean_w": _float(row.get("gpu_power_mean_w")),
                }
            )
        parsed = [r for r in parsed if r["request_rate"] > 0 and r["total_tok_s"] > 0]
        parsed.sort(key=lambda r: r["request_rate"])
        if parsed:
            best = max(parsed, key=lambda r: r["total_tok_s"])
            out["vllm_rate_max_total_tok_s"] = best["total_tok_s"]
            out["vllm_rate_at_max_total_tok_s"] = best["request_rate"]
            out["vllm_rate_p99_ttft_ms_at_max_total_tok_s"] = best["p99_ttft_ms"]
            out["vllm_rate_p99_tpot_ms_at_max_total_tok_s"] = best["p99_tpot_ms"]
            out["vllm_rate_best_gpu_power_mean_w"] = best["gpu_power_mean_w"]
            out["vllm_rate_best_tp"] = best["tp"]
            out["vllm_rate_best_tok_per_joule"] = (
                (best["total_tok_s"] / best["gpu_power_mean_w"]) if best["gpu_power_mean_w"] > 0 else 0.0
            )

    vllm_rate_stability_json = structured_dir / f"{run_id}_{label}_vllm_serve_request_rate_sweep_stability.json"
    if vllm_rate_stability_json.exists():
        payload = _load_json(vllm_rate_stability_json)
        summary = payload.get("summary") or {}
        out["vllm_rate_total_tok_cv_pct_p95"] = _num_or_none(summary.get("total_token_throughput_cv_pct_p95"))
        out["vllm_rate_p99_ttft_cv_pct_p95"] = _num_or_none(summary.get("p99_ttft_cv_pct_p95"))
        out["vllm_rate_p99_tpot_cv_pct_p95"] = _num_or_none(summary.get("p99_tpot_cv_pct_p95"))

    fio_json = structured_dir / f"{run_id}_{label}_fio.json"
    if fio_json.exists():
        payload = _load_json(fio_json).get("results") or {}
        seq_read = payload.get("seq_read") or {}
        seq_write = payload.get("seq_write") or {}
        out["fio_seq_read_mb_s"] = _float(seq_read.get("bw_mb_s"))
        out["fio_seq_write_mb_s"] = _float(seq_write.get("bw_mb_s"))

    fio_stability_json = structured_dir / f"{run_id}_{label}_fio_stability.json"
    if fio_stability_json.exists():
        payload = _load_json(fio_stability_json)
        summary = payload.get("summary") or {}
        out["fio_seq_read_bw_cv_pct"] = _num_or_none(summary.get("seq_read_bw_cv_pct"))
        out["fio_seq_write_bw_cv_pct"] = _num_or_none(summary.get("seq_write_bw_cv_pct"))
        out["fio_rand_read_iops_cv_pct"] = _num_or_none(summary.get("rand_read_iops_cv_pct"))
        out["fio_rand_write_iops_cv_pct"] = _num_or_none(summary.get("rand_write_iops_cv_pct"))

    return out


def _classify_bottleneck(summary: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    recommendations: List[str] = []
    bottleneck_type = "mixed"
    confidence = "low"

    comm_scale = _float(summary.get("nccl_multi_to_single_busbw_ratio"))
    alltoall_scale = _float(summary.get("nccl_alltoall_multi_to_single_busbw_ratio"))
    vllm_ttft_ratio = _float(summary.get("vllm_p99_ttft_ratio"))
    vllm_gain = _float(summary.get("vllm_throughput_gain_ratio"))
    stream_hbm_ratio = _float(summary.get("gpu_stream_to_hbm_ratio"))
    hbm_gbps = _float(summary.get("nvbandwidth_hbm_gbps"))
    gemm_tflops = _float(summary.get("gemm_max_tflops"))
    pcie_h2d = _float(summary.get("nvbandwidth_pcie_h2d_gbps"))
    goodput_eff = _float(summary.get("vllm_goodput_efficiency_tok_ratio"))
    vllm_conc_tok_cv = _num_or_none(summary.get("vllm_conc_total_tok_cv_pct_p95"))
    vllm_conc_ttft_cv = _num_or_none(summary.get("vllm_conc_p99_ttft_cv_pct_p95"))
    vllm_rate_tok_cv = _num_or_none(summary.get("vllm_rate_total_tok_cv_pct_p95"))
    allreduce_applicable = bool(summary.get("allreduce_applicable"))
    allreduce_cv = _float(summary.get("allreduce_busbw_cv_pct"))
    allreduce_p99_p50 = _float(summary.get("allreduce_p99_p50_ratio"))
    nccl_algo_applicable = bool(summary.get("nccl_algo_applicable"))
    nccl_algo_spread = _float(summary.get("nccl_algo_spread_pct"))
    nccl_algo_auto_gap = _float(summary.get("nccl_algo_auto_gap_pct"))

    if (
        (vllm_conc_tok_cv is not None and vllm_conc_tok_cv >= 10.0)
        or (vllm_conc_ttft_cv is not None and vllm_conc_ttft_cv >= 12.0)
        or (vllm_rate_tok_cv is not None and vllm_rate_tok_cv >= 10.0)
    ):
        bottleneck_type = "host-bound"
        confidence = "high"
        reasons.append(
            "vLLM sweep repeat variance is elevated "
            f"(conc tok/s CV p95={vllm_conc_tok_cv if vllm_conc_tok_cv is not None else 'n/a'}%, "
            f"conc p99 TTFT CV p95={vllm_conc_ttft_cv if vllm_conc_ttft_cv is not None else 'n/a'}%, "
            f"rate tok/s CV p95={vllm_rate_tok_cv if vllm_rate_tok_cv is not None else 'n/a'}%)."
        )
        recommendations.extend(
            [
                "Stabilize host scheduling path (CPU pinning, background load isolation, queue controls).",
                "Re-run workload sweeps with the same repeat count and verify CV is below 10% before promoting conclusions.",
            ]
        )
    elif allreduce_applicable and (allreduce_cv >= 5.0 or allreduce_p99_p50 >= 1.15):
        bottleneck_type = "communication-bound"
        confidence = "high" if allreduce_cv >= 5.0 else "medium"
        reasons.append(
            f"All-reduce stability indicates high jitter (CV={allreduce_cv:.2f}%, p99/p50={allreduce_p99_p50:.3f})."
        )
        recommendations.extend(
            [
                "Stabilize interconnect pathing (NIC affinity, routing, congestion controls) and rerun stability profile.",
                "Track per-iteration collective jitter during peak workload windows.",
            ]
        )
    elif nccl_algo_applicable and ((nccl_algo_spread >= 8.0) or (nccl_algo_auto_gap >= 5.0)):
        bottleneck_type = "communication-bound"
        confidence = "medium"
        reasons.append(
            f"NCCL algorithm sensitivity is high (spread={nccl_algo_spread:.2f}%, auto gap={nccl_algo_auto_gap:.2f}%)."
        )
        recommendations.extend(
            [
                "Pin the winning NCCL algorithm for production profile and validate across message-size bands.",
                "Investigate topology- or transport-specific algorithm regressions.",
            ]
        )
    elif (comm_scale > 0 and comm_scale < 0.65) or (alltoall_scale > 0 and alltoall_scale < 0.65):
        bottleneck_type = "communication-bound"
        confidence = "high"
        if comm_scale > 0 and comm_scale < 0.65:
            reasons.append(
                f"Multi-node NCCL all-reduce bus bandwidth ratio is low ({comm_scale:.2f} vs single-node), indicating interconnect pressure."
            )
        if alltoall_scale > 0 and alltoall_scale < 0.65:
            reasons.append(
                f"Multi-node NCCL all-to-all bus bandwidth ratio is low ({alltoall_scale:.2f} vs single-node), indicating MoE-style communication pressure."
            )
        recommendations.extend(
            [
                "Tune NCCL environment (CROSS_NIC, QPs, CTAs) and verify topology affinity.",
                "Profile collective overlap with compute using Nsight Systems NVTX ranges.",
            ]
        )
    elif (goodput_eff > 0 and goodput_eff < 0.60) or (vllm_ttft_ratio >= 3.0 and vllm_gain <= 2.2):
        bottleneck_type = "host-bound"
        confidence = "high" if goodput_eff > 0 and goodput_eff < 0.60 else "medium"
        if goodput_eff > 0:
            reasons.append(
                f"Only {goodput_eff:.2f} of peak token throughput meets SLO (goodput efficiency), indicating host/scheduler latency pressure."
            )
        if vllm_ttft_ratio > 0:
            reasons.append(
                f"vLLM tail latency growth is steep (p99 TTFT ratio={vllm_ttft_ratio:.2f}) with limited throughput gain ({vllm_gain:.2f}x)."
            )
        recommendations.extend(
            [
                "Inspect scheduler/data-path overhead and CPU-side batching limits.",
                "Run Nsight Systems + PyTorch trace to attribute host gaps and launch overhead.",
            ]
        )
    elif stream_hbm_ratio > 0 and stream_hbm_ratio < 0.55:
        bottleneck_type = "memory-bound"
        confidence = "medium"
        reasons.append(
            f"STREAM-like triad bandwidth is low relative to nvbandwidth HBM copy ({stream_hbm_ratio:.2f} ratio)."
        )
        recommendations.extend(
            [
                "Focus on memory access efficiency (coalescing, cache reuse, fusion, vectorization).",
                "Use Nsight Compute to inspect DRAM throughput and memory stalls on hot kernels.",
            ]
        )
    elif pcie_h2d > 0 and pcie_h2d < 20.0:
        bottleneck_type = "host-transfer-bound"
        confidence = "medium"
        reasons.append(f"Host-to-device transfer bandwidth is low ({pcie_h2d:.1f} GB/s).")
        recommendations.extend(
            [
                "Check PCIe generation/link width and NUMA placement.",
                "Reduce H2D pressure using pinned memory, prefetching, and overlap.",
            ]
        )
    elif gemm_tflops > 0 and hbm_gbps > 0 and gemm_tflops < 250.0 and hbm_gbps > 1500.0:
        bottleneck_type = "compute-bound"
        confidence = "medium"
        reasons.append(
            f"GEMM throughput is comparatively low ({gemm_tflops:.1f} TFLOPS) while memory bandwidth is healthy ({hbm_gbps:.1f} GB/s)."
        )
        recommendations.extend(
            [
                "Inspect occupancy, tensor-core utilization, and launch configuration.",
                "Use Nsight Compute to optimize math pipeline utilization.",
            ]
        )
    else:
        reasons.append("No single dominant subsystem bottleneck exceeded heuristic trigger thresholds.")
        recommendations.extend(
            [
                "Treat this run as mixed-bound; prioritize top workload hotspots.",
                "Run targeted deep-dive profiling on the slowest end-to-end workloads.",
            ]
        )

    return {
        "bottleneck_type": bottleneck_type,
        "confidence": confidence,
        "reasons": reasons,
        "recommendations": recommendations,
    }


def _fmt(v: Any, prec: int = 2) -> str:
    f = _float(v, default=float("nan"))
    if f != f:  # NaN
        return "n/a"
    return f"{f:.{prec}f}"


def _fmt_text(v: Any) -> str:
    if v is None:
        return "n/a"
    s = str(v).strip()
    return s if s else "n/a"


def _build_markdown(payload: Dict[str, Any]) -> str:
    run_id = payload["run_id"]
    summary = payload["summary"]
    bottleneck = payload["bottleneck"]
    lines: List[str] = []
    lines.append(f"# Cluster Scorecard: `{run_id}`")
    lines.append("")
    lines.append(f"Generated: `{payload['generated_at_utc']}`")
    lines.append(f"Workload KPI label: `{payload.get('resolved_primary_label') or 'n/a'}`")
    lines.append("")
    lines.append("## Canonical Completeness")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Overall score | `{_fmt(payload.get('overall_score'), 1)}` |")
    lines.append(f"| Pass/fail | `{_fmt_text(payload.get('pass_fail'))}` |")
    lines.append(f"| Coverage score | `{_fmt(payload.get('coverage_score_pct'), 0)}%` |")
    lines.append(f"| Advanced coverage score | `{_fmt(payload.get('advanced_coverage_score_pct'), 0)}%` |")
    lines.append(f"| Coverage maturity | `{_fmt_text(payload.get('coverage_maturity'))}` |")
    lines.append(f"| MLPerf overall status | `{_fmt_text(payload.get('mlperf_overall_status'))}` |")
    lines.append(
        f"| MLPerf inference track ready | `{_fmt_text(payload.get('mlperf_inference_track_ready'))}` |"
    )
    lines.append(
        f"| MLPerf training track ready | `{_fmt_text(payload.get('mlperf_training_track_ready'))}` |"
    )
    gates = payload.get("canonical_gates") or {}
    lines.append(f"| Gate: coverage >= min | `{_fmt_text(gates.get('coverage_min_met'))}` |")
    lines.append(f"| Gate: advanced >= min | `{_fmt_text(gates.get('advanced_min_met'))}` |")
    lines.append(f"| Gate: MLPerf alignment minimum | `{_fmt_text(gates.get('mlperf_alignment_min_met'))}` |")
    lines.append(f"| Gate: canonical complete | `{_fmt_text(gates.get('canonical_complete'))}` |")
    lines.append("")
    lines.append("## Unified KPIs")
    lines.append("")
    lines.append("| Domain | KPI | Value |")
    lines.append("|---|---|---:|")
    lines.append(f"| Compute | GEMM max TFLOPS | `{_fmt(summary.get('gemm_max_tflops'), 1)}` |")
    lines.append(f"| Memory | nvbandwidth HBM GB/s | `{_fmt(summary.get('nvbandwidth_hbm_gbps'), 1)}` |")
    lines.append(f"| Memory | STREAM-like triad GB/s | `{_fmt(summary.get('gpu_stream_triad_gbps'), 1)}` |")
    lines.append(f"| Communication | NCCL single-node peak busbw GB/s | `{_fmt(summary.get('nccl_single_peak_busbw_gbps'), 1)}` |")
    lines.append(f"| Communication | NCCL multi-node peak busbw GB/s | `{_fmt(summary.get('nccl_multi_peak_busbw_gbps'), 1)}` |")
    lines.append(f"| Communication | Multi/single busbw ratio | `{_fmt(summary.get('nccl_multi_to_single_busbw_ratio'), 2)}` |")
    lines.append(f"| Communication | NCCL all-to-all single-node peak busbw GB/s | `{_fmt(summary.get('nccl_alltoall_single_peak_busbw_gbps'), 1)}` |")
    lines.append(f"| Communication | NCCL all-to-all multi-node peak busbw GB/s | `{_fmt(summary.get('nccl_alltoall_multi_peak_busbw_gbps'), 1)}` |")
    lines.append(f"| Communication | NCCL all-to-all multi/single busbw ratio | `{_fmt(summary.get('nccl_alltoall_multi_to_single_busbw_ratio'), 2)}` |")
    lines.append(f"| Communication | NCCL algo winner | `{_fmt_text(summary.get('nccl_algo_best'))}` |")
    lines.append(f"| Communication | NCCL algo spread % | `{_fmt(summary.get('nccl_algo_spread_pct'), 2)}` |")
    lines.append(f"| Communication | NCCL auto gap % | `{_fmt(summary.get('nccl_algo_auto_gap_pct'), 2)}` |")
    lines.append(f"| Communication | Allreduce stability CV % | `{_fmt(summary.get('allreduce_busbw_cv_pct'), 2)}` |")
    lines.append(f"| Communication | Allreduce stability p99/p50 | `{_fmt(summary.get('allreduce_p99_p50_ratio'), 3)}` |")
    lines.append(f"| Communication | Allreduce jitter assessment | `{_fmt_text(summary.get('allreduce_jitter_assessment'))}` |")
    lines.append(
        f"| Communication | Allreduce latency comp (small/large duration ratio) | `{_fmt(summary.get('allreduce_latency_comp_ratio'), 2)}` |"
    )
    lines.append(
        f"| Communication | Allreduce latency comp one-large duration ms | `{_fmt(summary.get('allreduce_latency_comp_one_large_ms'), 4)}` |"
    )
    lines.append(
        f"| Communication | Allreduce latency comp many-small duration ms | `{_fmt(summary.get('allreduce_latency_comp_many_small_ms'), 4)}` |"
    )
    lines.append(
        f"| Communication | all_gather_object vs tensor speedup | `{_fmt(summary.get('allgather_obj_vs_tensor_speedup'), 2)}x` |"
    )
    lines.append(
        f"| Communication | all_gather_object vs all_reduce speedup | `{_fmt(summary.get('allgather_obj_vs_allreduce_speedup'), 2)}x` |"
    )
    lines.append(f"| Communication | Control-plane fastest method | `{_fmt_text(summary.get('allgather_fastest_method'))}` |")
    lines.append(
        f"| Communication | Control-plane fastest latency ms | `{_fmt(summary.get('allgather_fastest_latency_ms'), 4)}` |"
    )
    lines.append(f"| Host transfer | nvbandwidth H2D GB/s | `{_fmt(summary.get('nvbandwidth_pcie_h2d_gbps'), 1)}` |")
    lines.append(f"| Workload | vLLM throughput gain ratio | `{_fmt(summary.get('vllm_throughput_gain_ratio'), 2)}` |")
    lines.append(f"| Workload | vLLM p99 TTFT ratio | `{_fmt(summary.get('vllm_p99_ttft_ratio'), 2)}` |")
    lines.append(f"| Workload | vLLM max SLO goodput tok/s | `{_fmt(summary.get('vllm_max_goodput_tok_s'), 2)}` |")
    lines.append(f"| Workload | vLLM goodput efficiency ratio | `{_fmt(summary.get('vllm_goodput_efficiency_tok_ratio'), 2)}` |")
    lines.append(f"| Workload | vLLM knee concurrency | `{_fmt(summary.get('vllm_knee_concurrency'), 0)}` |")
    lines.append(f"| Workload | vLLM request-rate max tok/s | `{_fmt(summary.get('vllm_rate_max_total_tok_s'), 2)}` |")
    lines.append(f"| Workload | vLLM request-rate at max tok/s | `{_fmt(summary.get('vllm_rate_at_max_total_tok_s'), 2)}` |")
    lines.append(f"| Efficiency | vLLM tok/J @ max tok/s | `{_fmt(summary.get('vllm_tok_per_joule_at_max_total_tok_s'), 3)}` |")
    lines.append(
        f"| Efficiency | vLLM request-rate tok/J @ max tok/s | `{_fmt(summary.get('vllm_rate_tok_per_joule_at_max_total_tok_s'), 3)}` |"
    )
    lines.append(
        f"| Efficiency | Cost USD / 1M tok (concurrency) | `{_fmt(summary.get('vllm_cost_per_mtok_usd_at_max_total_tok_s'), 4)}` |"
    )
    lines.append(
        f"| Efficiency | Cost USD / 1M tok (request-rate) | `{_fmt(summary.get('vllm_rate_cost_per_mtok_usd_at_max_total_tok_s'), 4)}` |"
    )
    lines.append(f"| Workload Stability | vLLM conc tok/s CV p95 % | `{_fmt(summary.get('vllm_conc_total_tok_cv_pct_p95'), 2)}` |")
    lines.append(f"| Workload Stability | vLLM conc p99 TTFT CV p95 % | `{_fmt(summary.get('vllm_conc_p99_ttft_cv_pct_p95'), 2)}` |")
    lines.append(f"| Workload Stability | vLLM rate tok/s CV p95 % | `{_fmt(summary.get('vllm_rate_total_tok_cv_pct_p95'), 2)}` |")
    lines.append(f"| Storage Stability | fio seq-read BW CV % | `{_fmt(summary.get('fio_seq_read_bw_cv_pct'), 2)}` |")
    lines.append(f"| Storage Stability | fio seq-write BW CV % | `{_fmt(summary.get('fio_seq_write_bw_cv_pct'), 2)}` |")
    lines.append("")
    lines.append("## Bottleneck Classification")
    lines.append("")
    lines.append("| Classifier | Value |")
    lines.append("|---|---|")
    lines.append(f"| Dominant bottleneck | `{bottleneck['bottleneck_type']}` |")
    lines.append(f"| Confidence | `{bottleneck['confidence']}` |")
    lines.append("")
    lines.append("| Evidence |")
    lines.append("|---|")
    for reason in bottleneck.get("reasons", []):
        lines.append(f"| {reason} |")
    lines.append("")
    lines.append("| Recommended next actions |")
    lines.append("|---|")
    for rec in bottleneck.get("recommendations", []):
        lines.append(f"| {rec} |")
    lines.append("")
    lines.append("## Per-Node Metrics")
    lines.append("")
    lines.append(
        "| Label | GEMM max TFLOPS | nvbandwidth HBM GB/s | STREAM triad GB/s | vLLM tok/s gain | vLLM p99 TTFT ratio | vLLM max SLO goodput tok/s | vLLM knee concurrency | vLLM conc tok/s CV p95 % | fio seq read MB/s | fio seq read CV % | fio seq write MB/s | fio seq write CV % |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload.get("per_label_metrics", []):
        lines.append(
            f"| `{row.get('label','')}` | `{_fmt(row.get('gemm_max_tflops'), 1)}` | `{_fmt(row.get('nvbandwidth_hbm_gbps'), 1)}` | `{_fmt(row.get('gpu_stream_triad_gbps'), 1)}` | `{_fmt(row.get('vllm_throughput_gain_ratio'), 2)}` | `{_fmt(row.get('vllm_p99_ttft_ratio'), 2)}` | `{_fmt(row.get('vllm_slo_max_goodput_tok_s'), 2)}` | `{_fmt(row.get('vllm_slo_knee_concurrency'), 0)}` | `{_fmt(row.get('vllm_conc_total_tok_cv_pct_p95'), 2)}` | `{_fmt(row.get('fio_seq_read_mb_s'), 1)}` | `{_fmt(row.get('fio_seq_read_bw_cv_pct'), 2)}` | `{_fmt(row.get('fio_seq_write_mb_s'), 1)}` | `{_fmt(row.get('fio_seq_write_bw_cv_pct'), 2)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cluster scorecard and bottleneck classification.")
    p.add_argument("--run-id", required=True, help="Run ID prefix")
    p.add_argument("--primary-label", default="", help="Preferred label for workload KPI selection")
    p.add_argument(
        "--structured-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "structured"),
        help="Structured artifact directory",
    )
    p.add_argument("--output-json", default="", help="Output JSON path (default: <structured>/<run_id>_cluster_scorecard.json)")
    p.add_argument("--output-md", default="", help="Output markdown path (default: <structured>/<run_id>_cluster_scorecard.md)")
    p.add_argument(
        "--gpu-hourly-cost-usd",
        default="",
        help="Optional single-GPU hourly cost in USD to compute cost-per-1M-token metrics.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    structured_dir = Path(args.structured_dir)
    run_id = args.run_id
    output_json = Path(args.output_json) if args.output_json else structured_dir / f"{run_id}_cluster_scorecard.json"
    output_md = Path(args.output_md) if args.output_md else structured_dir / f"{run_id}_cluster_scorecard.md"

    labels = _detect_labels(structured_dir, run_id)
    per_label = [_label_metrics(structured_dir, run_id, lbl) for lbl in labels]

    single_nccl = _peak_nccl(structured_dir / f"{run_id}_node1_nccl.json")
    multi_nccl = _peak_nccl(structured_dir / f"{run_id}_2nodes_nccl.json")
    single_alltoall = _peak_nccl(structured_dir / f"{run_id}_node1_alltoall_nccl_alltoall.json")
    multi_alltoall = _peak_nccl(structured_dir / f"{run_id}_2nodes_alltoall_nccl_alltoall.json")
    allreduce_stability = _allreduce_stability_metrics(structured_dir / f"{run_id}_allreduce_stability.json")
    allreduce_latency_comp = _allreduce_latency_comp_metrics(structured_dir / f"{run_id}_allreduce_latency_comp.json")
    allgather_control_plane = _allgather_control_plane_metrics(structured_dir / f"{run_id}_allgather_control_plane.json")
    nccl_algo = _nccl_algo_metrics(structured_dir / f"{run_id}_nccl_algo_comparison.json")
    coverage = _coverage_metrics(structured_dir / f"{run_id}_benchmark_coverage_analysis.json")
    mlperf = _mlperf_metrics(structured_dir / f"{run_id}_mlperf_alignment.json")
    canonical_gates = _canonical_gates(
        coverage_score_pct=coverage.get("coverage_score_pct"),
        advanced_coverage_score_pct=coverage.get("advanced_coverage_score_pct"),
        mlperf_overall_status=str(mlperf.get("overall_status") or ""),
    )
    nccl_ratio = 0.0
    if single_nccl["peak_busbw_gbps"] > 0:
        nccl_ratio = multi_nccl["peak_busbw_gbps"] / single_nccl["peak_busbw_gbps"]
    nccl_alltoall_ratio = 0.0
    if single_alltoall["peak_busbw_gbps"] > 0:
        nccl_alltoall_ratio = multi_alltoall["peak_busbw_gbps"] / single_alltoall["peak_busbw_gbps"]

    def _max_metric(key: str) -> float:
        return max((_float(row.get(key)) for row in per_label), default=0.0)

    # Prefer explicitly requested label. Otherwise choose the first row with workload KPIs.
    requested_primary = (args.primary_label or "").strip()
    primary_row: Dict[str, Any] = {}
    if per_label:
        if requested_primary:
            for row in per_label:
                if str(row.get("label", "")).strip() == requested_primary:
                    primary_row = row
                    break
        if not primary_row:
            for row in per_label:
                if _float(row.get("vllm_throughput_gain_ratio")) > 0 or _float(row.get("vllm_rate_max_total_tok_s")) > 0:
                    primary_row = row
                    break
        if not primary_row:
            primary_row = per_label[0]
    resolved_primary_label = str(primary_row.get("label", requested_primary or ""))
    summary = {
        "gemm_max_tflops": _max_metric("gemm_max_tflops"),
        "nvbandwidth_hbm_gbps": _max_metric("nvbandwidth_hbm_gbps"),
        "nvbandwidth_pcie_h2d_gbps": _max_metric("nvbandwidth_pcie_h2d_gbps"),
        "nvbandwidth_pcie_d2h_gbps": _max_metric("nvbandwidth_pcie_d2h_gbps"),
        "gpu_stream_peak_gbps": _max_metric("gpu_stream_peak_gbps"),
        "gpu_stream_triad_gbps": _max_metric("gpu_stream_triad_gbps"),
        "nccl_single_peak_algbw_gbps": single_nccl["peak_algbw_gbps"],
        "nccl_single_peak_busbw_gbps": single_nccl["peak_busbw_gbps"],
        "nccl_multi_peak_algbw_gbps": multi_nccl["peak_algbw_gbps"],
        "nccl_multi_peak_busbw_gbps": multi_nccl["peak_busbw_gbps"],
        "nccl_multi_to_single_busbw_ratio": nccl_ratio if nccl_ratio > 0 else 0.0,
        "nccl_alltoall_single_peak_algbw_gbps": single_alltoall["peak_algbw_gbps"],
        "nccl_alltoall_single_peak_busbw_gbps": single_alltoall["peak_busbw_gbps"],
        "nccl_alltoall_multi_peak_algbw_gbps": multi_alltoall["peak_algbw_gbps"],
        "nccl_alltoall_multi_peak_busbw_gbps": multi_alltoall["peak_busbw_gbps"],
        "nccl_alltoall_multi_to_single_busbw_ratio": nccl_alltoall_ratio if nccl_alltoall_ratio > 0 else 0.0,
        "allreduce_applicable": bool(allreduce_stability.get("allreduce_applicable")),
        "allreduce_busbw_mean_gbps": allreduce_stability.get("allreduce_busbw_mean_gbps"),
        "allreduce_busbw_cv_pct": allreduce_stability.get("allreduce_busbw_cv_pct"),
        "allreduce_p99_p50_ratio": allreduce_stability.get("allreduce_p99_p50_ratio"),
        "allreduce_jitter_assessment": allreduce_stability.get("allreduce_jitter_assessment"),
        "allreduce_latency_comp_ratio": allreduce_latency_comp.get("allreduce_latency_comp_ratio"),
        "allreduce_latency_comp_one_large_ms": allreduce_latency_comp.get("allreduce_latency_comp_one_large_ms"),
        "allreduce_latency_comp_many_small_ms": allreduce_latency_comp.get("allreduce_latency_comp_many_small_ms"),
        "allgather_obj_vs_tensor_speedup": allgather_control_plane.get("allgather_obj_vs_tensor_speedup"),
        "allgather_obj_vs_allreduce_speedup": allgather_control_plane.get("allgather_obj_vs_allreduce_speedup"),
        "allgather_fastest_method": allgather_control_plane.get("allgather_fastest_method"),
        "allgather_fastest_latency_ms": allgather_control_plane.get("allgather_fastest_latency_ms"),
        "nccl_algo_applicable": bool(nccl_algo.get("nccl_algo_applicable")),
        "nccl_algo_best": nccl_algo.get("nccl_algo_best"),
        "nccl_algo_peak_busbw_gbps": nccl_algo.get("nccl_algo_peak_busbw_gbps"),
        "nccl_algo_spread_pct": nccl_algo.get("nccl_algo_spread_pct"),
        "nccl_algo_auto_gap_pct": nccl_algo.get("nccl_algo_auto_gap_pct"),
        "vllm_throughput_gain_ratio": _float(primary_row.get("vllm_throughput_gain_ratio")),
        "vllm_p99_ttft_ratio": _float(primary_row.get("vllm_p99_ttft_ratio")),
        "vllm_max_goodput_tok_s": _float(primary_row.get("vllm_slo_max_goodput_tok_s")),
        "vllm_max_goodput_req_s": _float(primary_row.get("vllm_slo_max_goodput_req_s")),
        "vllm_goodput_efficiency_tok_ratio": _float(primary_row.get("vllm_slo_goodput_efficiency_tok_ratio")),
        "vllm_knee_concurrency": _float(primary_row.get("vllm_slo_knee_concurrency")),
        "vllm_rate_max_total_tok_s": _float(primary_row.get("vllm_rate_max_total_tok_s")),
        "vllm_rate_at_max_total_tok_s": _float(primary_row.get("vllm_rate_at_max_total_tok_s")),
        "vllm_tok_per_joule_at_max_total_tok_s": _float(primary_row.get("vllm_best_tok_per_joule")),
        "vllm_rate_tok_per_joule_at_max_total_tok_s": _float(primary_row.get("vllm_rate_best_tok_per_joule")),
        "vllm_conc_total_tok_cv_pct_p95": primary_row.get("vllm_conc_total_tok_cv_pct_p95"),
        "vllm_conc_p99_ttft_cv_pct_p95": primary_row.get("vllm_conc_p99_ttft_cv_pct_p95"),
        "vllm_conc_p99_tpot_cv_pct_p95": primary_row.get("vllm_conc_p99_tpot_cv_pct_p95"),
        "vllm_rate_total_tok_cv_pct_p95": primary_row.get("vllm_rate_total_tok_cv_pct_p95"),
        "vllm_rate_p99_ttft_cv_pct_p95": primary_row.get("vllm_rate_p99_ttft_cv_pct_p95"),
        "vllm_rate_p99_tpot_cv_pct_p95": primary_row.get("vllm_rate_p99_tpot_cv_pct_p95"),
        "fio_seq_read_bw_cv_pct": primary_row.get("fio_seq_read_bw_cv_pct"),
        "fio_seq_write_bw_cv_pct": primary_row.get("fio_seq_write_bw_cv_pct"),
        "gpu_hourly_cost_usd": None,
        "vllm_cost_per_mtok_usd_at_max_total_tok_s": None,
        "vllm_rate_cost_per_mtok_usd_at_max_total_tok_s": None,
    }
    hourly_cost_raw = (args.gpu_hourly_cost_usd or "").strip()
    hourly_cost = _num_or_none(hourly_cost_raw) if hourly_cost_raw else None
    if hourly_cost is not None and hourly_cost > 0:
        summary["gpu_hourly_cost_usd"] = hourly_cost
        conc_tp = int(_float(primary_row.get("vllm_best_tp"), 0.0))
        conc_tok_s = _float(primary_row.get("vllm_best_total_tok_s"))
        if conc_tp > 0 and conc_tok_s > 0:
            summary["vllm_cost_per_mtok_usd_at_max_total_tok_s"] = ((hourly_cost * conc_tp) / 3600.0) * (
                1_000_000.0 / conc_tok_s
            )
        rate_tp = int(_float(primary_row.get("vllm_rate_best_tp"), 0.0))
        rate_tok_s = _float(primary_row.get("vllm_rate_max_total_tok_s"))
        if rate_tp > 0 and rate_tok_s > 0:
            summary["vllm_rate_cost_per_mtok_usd_at_max_total_tok_s"] = ((hourly_cost * rate_tp) / 3600.0) * (
                1_000_000.0 / rate_tok_s
            )
    if summary["nvbandwidth_hbm_gbps"] > 0:
        summary["gpu_stream_to_hbm_ratio"] = summary["gpu_stream_triad_gbps"] / summary["nvbandwidth_hbm_gbps"]
    else:
        summary["gpu_stream_to_hbm_ratio"] = 0.0

    coverage_score = int(coverage.get("coverage_score_pct") or 0)
    advanced_score = int(coverage.get("advanced_coverage_score_pct") or 0)
    mlperf_status = str(mlperf.get("overall_status") or "").strip().lower()
    if mlperf_status == "aligned":
        mlperf_score = 100.0
    elif mlperf_status in {"inference_ready_only", "training_ready_only"}:
        mlperf_score = 80.0
    elif mlperf_status == "partial":
        mlperf_score = 40.0
    else:
        mlperf_score = 0.0
    overall_score = round((0.35 * coverage_score) + (0.35 * advanced_score) + (0.30 * mlperf_score), 1)
    pass_fail = "pass" if bool(canonical_gates.get("canonical_complete")) else "fail"

    bottleneck = _classify_bottleneck(summary)
    payload = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_score": overall_score,
        "pass_fail": pass_fail,
        "coverage_score_pct": coverage.get("coverage_score_pct"),
        "advanced_coverage_score_pct": coverage.get("advanced_coverage_score_pct"),
        "coverage_maturity": coverage.get("coverage_maturity"),
        "mlperf_overall_status": mlperf.get("overall_status"),
        "mlperf_inference_track_ready": mlperf.get("inference_track_ready"),
        "mlperf_training_track_ready": mlperf.get("training_track_ready"),
        "canonical_gates": canonical_gates,
        "labels": labels,
        "requested_primary_label": requested_primary,
        "resolved_primary_label": resolved_primary_label,
        "per_label_metrics": per_label,
        "summary": summary,
        "bottleneck": bottleneck,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_build_markdown(payload), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
