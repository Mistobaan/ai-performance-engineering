#!/usr/bin/env bash
set -euo pipefail

MODEL="$1"
TP="$2"
ISL="$3"
OSL="$4"
MAX_MODEL_LEN="$5"
PORT="$6"
SWEEP_DIR="$7"
shift 7
CONCURRENCY_RANGE="$@"

export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
VLLM_SERVE_ENFORCE_EAGER="${VLLM_SERVE_ENFORCE_EAGER:-1}"
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_SERVER_READY_TIMEOUT="${VLLM_SERVER_READY_TIMEOUT:-1200}"
VLLM_SWEEP_NUM_PROMPTS_MULTIPLIER="${VLLM_SWEEP_NUM_PROMPTS_MULTIPLIER:-10}"
VLLM_SWEEP_MAX_POINTS_PER_RUN="${VLLM_SWEEP_MAX_POINTS_PER_RUN:-0}"
VLLM_SWEEP_STRICT_POINT_VALIDATION="${VLLM_SWEEP_STRICT_POINT_VALIDATION:-1}"

if ! python3 - "$VLLM_GPU_MEMORY_UTILIZATION" <<'PY'
import sys
v = float(sys.argv[1])
if v <= 0.0 or v > 1.0:
    raise SystemExit(1)
PY
then
  echo "ERROR: VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1], got '${VLLM_GPU_MEMORY_UTILIZATION}'." >&2
  exit 2
fi
if ! [[ "$VLLM_SERVER_READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: VLLM_SERVER_READY_TIMEOUT must be a positive integer, got '${VLLM_SERVER_READY_TIMEOUT}'." >&2
  exit 2
fi
if ! [[ "$VLLM_SWEEP_NUM_PROMPTS_MULTIPLIER" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: VLLM_SWEEP_NUM_PROMPTS_MULTIPLIER must be a positive integer, got '${VLLM_SWEEP_NUM_PROMPTS_MULTIPLIER}'." >&2
  exit 2
fi
if ! [[ "$VLLM_SWEEP_MAX_POINTS_PER_RUN" =~ ^[0-9]+$ ]]; then
  echo "ERROR: VLLM_SWEEP_MAX_POINTS_PER_RUN must be a non-negative integer, got '${VLLM_SWEEP_MAX_POINTS_PER_RUN}'." >&2
  exit 2
fi

effective_gpu_mem_util="$VLLM_GPU_MEMORY_UTILIZATION"
total_mem_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
free_mem_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if [[ "$total_mem_mib" =~ ^[0-9]+$ ]] && [[ "$free_mem_mib" =~ ^[0-9]+$ ]] && [[ "$total_mem_mib" -gt 0 ]]; then
  effective_gpu_mem_util="$(python3 - "$VLLM_GPU_MEMORY_UTILIZATION" "$free_mem_mib" "$total_mem_mib" <<'PY'
import sys
requested = float(sys.argv[1])
free_mib = float(sys.argv[2])
total_mib = float(sys.argv[3])
headroom_mib = 3072.0
safe = max(0.05, min(0.95, ((free_mib - headroom_mib) / total_mib) * 0.995))
print(f"{min(requested, safe):.4f}")
PY
)"
  if [[ "$effective_gpu_mem_util" != "$VLLM_GPU_MEMORY_UTILIZATION" ]]; then
    echo "Lowering --gpu-memory-utilization from ${VLLM_GPU_MEMORY_UTILIZATION} to ${effective_gpu_mem_util} based on free VRAM (${free_mem_mib} MiB / ${total_mem_mib} MiB)."
  fi
fi

if [[ "$MODEL" == *"gpt-oss"* ]]; then
  export VLLM_MXFP4_USE_MARLIN=1
fi

mkdir -p "$SWEEP_DIR"

SERVER_LOG="${SWEEP_DIR}/server.log"
SUMMARY_FILE="${SWEEP_DIR}/summary.txt"
SUMMARY_CSV="${SWEEP_DIR}/sweep_summary.csv"
SUMMARY_JSONL="${SWEEP_DIR}/sweep_summary.jsonl"

CSV_HEADER="model,tp,isl,osl,concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,gpu_util_mean_pct,gpu_util_p95_pct,mem_used_mean_mb,mem_used_max_mb,gpu_power_mean_w,gpu_power_p95_w,completed,failed"
if [[ ! -f "$SUMMARY_CSV" ]] || [[ ! -s "$SUMMARY_CSV" ]]; then
  echo "$CSV_HEADER" >"$SUMMARY_CSV"
fi
if [[ ! -f "$SUMMARY_JSONL" ]]; then
  : >"$SUMMARY_JSONL"
fi

rewrite_summary_from_csv() {
  python3 - "$MODEL" "$TP" "$ISL" "$OSL" "$SUMMARY_CSV" "$SUMMARY_FILE" <<'PY'
import csv
import sys
from pathlib import Path

model, tp, isl, osl, csv_path, summary_path = sys.argv[1:]
rows = []
with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            conc = int(float(r.get("concurrency", "0") or 0))
        except ValueError:
            continue
        rows.append((conc, r))
rows.sort(key=lambda x: x[0])
with Path(summary_path).open("w", encoding="utf-8") as out:
    out.write("========================================\n")
    out.write("vLLM Concurrency Sweep Results\n")
    out.write("========================================\n")
    out.write(f"Model: {model}\n")
    out.write(f"TP: {tp}\n")
    out.write(f"ISL: {isl}, OSL: {osl}\n\n")
    out.write("Concurrency | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | P99 TPOT\n")
    out.write("------------|--------------|-------------|-----------|-----------|----------\n")
    for conc, row in rows:
        def metric(key):
            try:
                return float(row.get(key, "0") or 0.0)
            except ValueError:
                return 0.0
        out.write(
            f"{conc:<11d} | {metric('output_throughput'):<12.2f} | {metric('total_token_throughput'):<11.2f} | "
            f"{metric('mean_ttft_ms'):<9.2f} | {metric('mean_tpot_ms'):<9.3f} | {metric('p99_tpot_ms'):<9.3f}\n"
        )
PY
}

declare -A COMPLETED_CONC=()
while IFS=, read -r conc completed failed total_tok _rest; do
  [[ -n "$conc" ]] || continue
  if [[ "$conc" == "concurrency" ]]; then
    continue
  fi
  if [[ "$completed" =~ ^[0-9]+$ ]] && [[ "$failed" =~ ^[0-9]+$ ]]; then
    if [[ "$completed" -gt 0 ]] && [[ "$failed" -eq 0 ]]; then
      if python3 - "$total_tok" <<'PY'
import sys
v = float(sys.argv[1] or "0")
raise SystemExit(0 if v > 0 else 1)
PY
      then
        COMPLETED_CONC["$conc"]=1
      fi
    fi
  fi
done < <(python3 - "$SUMMARY_CSV" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
with p.open("r", encoding="utf-8", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        print(",".join([
            (row.get("concurrency") or "").strip(),
            (row.get("completed") or "").strip(),
            (row.get("failed") or "").strip(),
            (row.get("total_token_throughput") or "").strip(),
            "",
        ]))
PY
)

declare -a PENDING_CONCURRENCY=()
for CONC in $CONCURRENCY_RANGE; do
  if [[ -n "${COMPLETED_CONC[$CONC]:-}" ]]; then
    echo "Skipping completed concurrency point ${CONC} (resume)."
  else
    PENDING_CONCURRENCY+=("$CONC")
  fi
done

if [[ "${#PENDING_CONCURRENCY[@]}" -eq 0 ]]; then
  rewrite_summary_from_csv
  echo "All concurrency points already completed; nothing to run."
  echo
  cat "$SUMMARY_FILE"
  exit 0
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ensure_server_healthy() {
  local context="$1"
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: vLLM server process is not running (${context})." >&2
    tail -120 "$SERVER_LOG" || true
    exit 3
  fi
  if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "ERROR: vLLM server health check failed (${context})." >&2
    tail -120 "$SERVER_LOG" || true
    exit 3
  fi
  if grep -qE "Engine core proc .* died unexpectedly|EngineDeadError" "$SERVER_LOG"; then
    echo "ERROR: vLLM engine death detected in server log (${context})." >&2
    tail -120 "$SERVER_LOG" || true
    exit 3
  fi
}

echo "=== Starting vLLM Server ==="
SERVE_ARGS=(
  "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --gpu-memory-utilization "$effective_gpu_mem_util"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs 1024
  --disable-log-requests
)

if [[ -n "$VLLM_KV_CACHE_MEMORY_BYTES" ]]; then
  # Optional explicit override. By default we let vLLM size KV cache from
  # --gpu-memory-utilization to avoid stale fixed-byte OOM failures.
  echo "Using --kv-cache-memory-bytes=${VLLM_KV_CACHE_MEMORY_BYTES}"
  SERVE_ARGS+=(--kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES")
fi

if [[ "$VLLM_SERVE_ENFORCE_EAGER" == "1" ]]; then
  echo "Enabling --enforce-eager for startup robustness."
  SERVE_ARGS+=(--enforce-eager)
fi

vllm serve "${SERVE_ARGS[@]}" >"$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "Server log: $SERVER_LOG"

echo "Waiting for server to be ready..."
MAX_WAIT="$VLLM_SERVER_READY_TIMEOUT"
WAITED=0
while ! curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  if [[ -f "$SERVER_LOG" ]] && grep -qE "Engine core initialization failed|AssertionError: Error in memory profiling|RuntimeError: Engine core initialization failed" "$SERVER_LOG"; then
    echo "ERROR: Server reported a fatal initialization error before becoming healthy"
    tail -120 "$SERVER_LOG" || true
    exit 1
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: Server died before becoming healthy"
    tail -100 "$SERVER_LOG" || true
    exit 1
  fi
  if [[ "$WAITED" -ge "$MAX_WAIT" ]]; then
    echo "ERROR: Server failed to start within ${MAX_WAIT}s"
    tail -100 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 5
  WAITED=$((WAITED + 5))
  echo "  Waiting... (${WAITED}s)"
done

echo "Server is ready!"
echo
points_run=0
partial_resume=0
for CONC in "${PENDING_CONCURRENCY[@]}"; do
  if [[ "$VLLM_SWEEP_MAX_POINTS_PER_RUN" -gt 0 && "$points_run" -ge "$VLLM_SWEEP_MAX_POINTS_PER_RUN" ]]; then
    partial_resume=1
    break
  fi
  points_run=$((points_run + 1))
  echo
  echo "========================================"
  echo "=== Running Benchmark: Concurrency $CONC ==="
  echo "========================================"

  NUM_PROMPTS=$((CONC * VLLM_SWEEP_NUM_PROMPTS_MULTIPLIER))
  RESULT_JSON="conc${CONC}_isl${ISL}_osl${OSL}_tp${TP}.json"
  RESULT_TXT="conc${CONC}_bench.txt"
  TELEMETRY_CSV="conc${CONC}_telemetry.csv"

  ensure_server_healthy "before concurrency=${CONC}"

  # Capture GPU util/memory telemetry while the benchmark is running.
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,nounits >"${SWEEP_DIR}/${TELEMETRY_CSV}" || true

  # Run bench in background so we can sample telemetry concurrently.
  set +e
  vllm bench serve \
    --model "$MODEL" \
    --backend vllm \
    --base-url "http://localhost:$PORT" \
    --dataset-name random \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --save-result \
    --result-dir "$SWEEP_DIR" \
    --result-filename "$RESULT_JSON" > >(tee "${SWEEP_DIR}/${RESULT_TXT}") 2>&1 &
  BENCH_PID=$!

  while kill -0 "$BENCH_PID" >/dev/null 2>&1; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits >>"${SWEEP_DIR}/${TELEMETRY_CSV}" || true
    sleep 1
  done

  wait "$BENCH_PID"
  BENCH_RC=$?
  set -e

  if [[ "$BENCH_RC" -ne 0 ]]; then
    echo "ERROR: vllm bench serve failed for concurrency ${CONC} (rc=${BENCH_RC})" >&2
    tail -120 "$SERVER_LOG" || true
    exit "$BENCH_RC"
  fi

  ensure_server_healthy "after concurrency=${CONC}"

  set +e
  python3 - <<'PY' \
    "$MODEL" "$TP" "$ISL" "$OSL" "$CONC" "$NUM_PROMPTS" \
    "${SWEEP_DIR}/${RESULT_JSON}" "${SWEEP_DIR}/${TELEMETRY_CSV}" \
    "$SUMMARY_CSV" "$SUMMARY_JSONL" "$VLLM_SWEEP_STRICT_POINT_VALIDATION"
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

model, tp, isl, osl, conc, num_prompts = sys.argv[1:7]
result_path = Path(sys.argv[7])
telemetry_path = Path(sys.argv[8])
csv_out = Path(sys.argv[9])
jsonl_out = Path(sys.argv[10])
strict = (sys.argv[11] or "").strip() != "0"

def pctl(vals, q):
    if not vals:
        return None
    xs = sorted(vals)
    # Nearest-rank percentile.
    k = max(0, min(len(xs) - 1, int(math.ceil((q / 100.0) * len(xs))) - 1))
    return xs[k]

data = json.loads(result_path.read_text())

# Parse telemetry (one row per GPU per sample).
tele = {}
if telemetry_path.exists():
    with telemetry_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
            idx = row.get("index", "")
            if idx == "":
                continue
            try:
                gpu = int(idx)
            except ValueError:
                continue
            def to_f(key):
                v = row.get(key, "")
                try:
                    return float(v)
                except ValueError:
                    return None
            util = to_f("utilization.gpu [%]")
            mem_used = to_f("memory.used [MiB]")
            power_draw = to_f("power.draw [W]")
            if power_draw is None:
                power_draw = to_f("power.draw")
            if util is None and mem_used is None and power_draw is None:
                continue
            tele.setdefault(gpu, {"util_gpu": [], "mem_used": [], "power_draw": []})
            if util is not None:
                tele[gpu]["util_gpu"].append(util)
            if mem_used is not None:
                tele[gpu]["mem_used"].append(mem_used)
            if power_draw is not None:
                tele[gpu]["power_draw"].append(power_draw)

per_gpu = []
util_means = []
util_p95s = []
mem_means = []
mem_maxs = []
power_means = []
power_p95s = []
for gpu, series in sorted(tele.items()):
    util = series.get("util_gpu", [])
    mem = series.get("mem_used", [])
    power = series.get("power_draw", [])
    u_mean = mean(util) if util else None
    u_p95 = pctl(util, 95) if util else None
    m_mean = mean(mem) if mem else None
    m_max = max(mem) if mem else None
    p_mean = mean(power) if power else None
    p_p95 = pctl(power, 95) if power else None
    per_gpu.append(
        {
            "gpu": gpu,
            "util_gpu_mean_pct": u_mean,
            "util_gpu_p95_pct": u_p95,
            "mem_used_mean_mib": m_mean,
            "mem_used_max_mib": m_max,
            "power_mean_w": p_mean,
            "power_p95_w": p_p95,
        }
    )
    if u_mean is not None:
        util_means.append(u_mean)
    if u_p95 is not None:
        util_p95s.append(u_p95)
    if m_mean is not None:
        mem_means.append(m_mean)
    if m_max is not None:
        mem_maxs.append(m_max)
    if p_mean is not None:
        power_means.append(p_mean)
    if p_p95 is not None:
        power_p95s.append(p_p95)

util_mean = mean(util_means) if util_means else None
util_p95 = mean(util_p95s) if util_p95s else None
mem_mean = mean(mem_means) if mem_means else None
mem_max = max(mem_maxs) if mem_maxs else None
power_mean = mean(power_means) if power_means else None
power_p95 = mean(power_p95s) if power_p95s else None

row = {
    "model": model,
    "tp": int(tp),
    "isl": int(isl),
    "osl": int(osl),
    "concurrency": int(conc),
    "num_prompts": int(num_prompts),
    "request_throughput": float(data.get("request_throughput", 0.0) or 0.0),
    "output_throughput": float(data.get("output_throughput", 0.0) or 0.0),
    "total_token_throughput": float(data.get("total_token_throughput", 0.0) or 0.0),
    "mean_ttft_ms": float(data.get("mean_ttft_ms", 0.0) or 0.0),
    "median_ttft_ms": float(data.get("median_ttft_ms", 0.0) or 0.0),
    "p99_ttft_ms": float(data.get("p99_ttft_ms", 0.0) or 0.0),
    "mean_tpot_ms": float(data.get("mean_tpot_ms", 0.0) or 0.0),
    "median_tpot_ms": float(data.get("median_tpot_ms", 0.0) or 0.0),
    "p99_tpot_ms": float(data.get("p99_tpot_ms", 0.0) or 0.0),
    "completed": int(data.get("completed", 0) or 0),
    "failed": int(data.get("failed", 0) or 0),
    "gpu_telemetry": {
        "per_gpu": per_gpu,
        "util_gpu_mean_pct": util_mean,
        "util_gpu_p95_pct": util_p95,
        "mem_used_mean_mib": mem_mean,
        "mem_used_max_mib": mem_max,
        "power_mean_w": power_mean,
        "power_p95_w": power_p95,
    },
    "result_json": str(result_path),
    "telemetry_csv": str(telemetry_path),
}

if strict:
    if row["completed"] <= 0:
        raise SystemExit(
            f"Invalid concurrency point {row['concurrency']}: completed={row['completed']} (must be > 0)."
        )
    if row["failed"] > 0:
        raise SystemExit(
            f"Invalid concurrency point {row['concurrency']}: failed={row['failed']} (must be 0)."
        )
    if row["total_token_throughput"] <= 0.0:
        raise SystemExit(
            f"Invalid concurrency point {row['concurrency']}: total_token_throughput={row['total_token_throughput']:.6f} (must be > 0)."
        )

with jsonl_out.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")

def fnum(v):
    return "" if v is None else f"{v:.3f}"

with csv_out.open("a", encoding="utf-8", newline="") as f:
    f.write(",".join(
        [
            model,
            str(int(tp)),
            str(int(isl)),
            str(int(osl)),
            str(int(conc)),
            str(int(num_prompts)),
            fnum(row["request_throughput"]),
            fnum(row["output_throughput"]),
            fnum(row["total_token_throughput"]),
            fnum(row["mean_ttft_ms"]),
            fnum(row["median_ttft_ms"]),
            fnum(row["p99_ttft_ms"]),
            fnum(row["mean_tpot_ms"]),
            fnum(row["median_tpot_ms"]),
            fnum(row["p99_tpot_ms"]),
            fnum(util_mean),
            fnum(util_p95),
            fnum(mem_mean),
            fnum(mem_max),
            fnum(power_mean),
            fnum(power_p95),
            str(row["completed"]),
            str(row["failed"]),
        ]
    ) + "\n")
PY
  POINT_RC=$?
  set -e
  if [[ "$POINT_RC" -ne 0 ]]; then
    echo "ERROR: invalid vLLM concurrency sweep point at concurrency=${CONC}" >&2
    tail -120 "$SERVER_LOG" || true
    exit "$POINT_RC"
  fi

  echo
  echo "Results saved to:"
  echo "  - ${SWEEP_DIR}/${RESULT_JSON}"
  echo "  - ${SWEEP_DIR}/${RESULT_TXT}"
  echo "  - ${SWEEP_DIR}/${TELEMETRY_CSV}"
  COMPLETED_CONC["$CONC"]=1
done

rewrite_summary_from_csv
if [[ "$partial_resume" -eq 1 ]]; then
  echo
  echo "Reached VLLM_SWEEP_MAX_POINTS_PER_RUN=${VLLM_SWEEP_MAX_POINTS_PER_RUN}; exiting with resumable status."
  exit 75
fi

echo
echo "========================================"
echo "=== Sweep Complete ==="
echo "========================================"
cat "$SUMMARY_FILE"
