#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_vllm_bench.sh

Notes:
  - GPU clock lock is REQUIRED. Run via:
      scripts/run_with_gpu_clocks.sh -- scripts/run_vllm_bench.sh
EOF
}

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PROMPT_LEN="${PROMPT_LEN:-512}"
GEN_LEN="${GEN_LEN:-256}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1,2,4,8,16}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-15}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=./lib_host_runtime_env.sh
source "${ROOT_DIR}/scripts/lib_host_runtime_env.sh"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
source_host_runtime_env_if_present "$ROOT_DIR"
resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_STRUCT="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
OUT_RAW="${CLUSTER_RAW_DIR_EFFECTIVE}"
VENV_PY="${ROOT_DIR}/env/venv/bin/python"
VLLM_BIN="${ROOT_DIR}/env/venv/bin/vllm"

mkdir -p "$OUT_STRUCT" "$OUT_RAW"

if [[ "${AISP_CLOCK_LOCKED:-}" != "1" ]]; then
  export RUN_ID
  export LABEL="${LABEL:-$(hostname)}"
  LOCK_META_OUT="${OUT_STRUCT}/${RUN_ID}_${LABEL}_vllm_bench_clock_lock.json"
  exec "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --lock-meta-out "$LOCK_META_OUT" \
    -- "$0" "$@"
fi

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing venv python at $VENV_PY" >&2
  exit 1
fi

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "Missing vllm CLI at $VLLM_BIN" >&2
  exit 1
fi

clock_csv="$("$VENV_PY" - <<'PY'
import torch

try:
    import pynvml
except ImportError as exc:
    raise SystemExit(f"pynvml not installed ({exc}); cannot capture app clocks")

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available; cannot capture app clocks")

pynvml.nvmlInit()
try:
    idx = int(torch.cuda.current_device())
    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
    app_sm = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM))
    app_mem = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM))
    cur_sm = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
    cur_mem = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
    print(f"{idx},{app_sm},{app_mem},{cur_sm},{cur_mem}")
finally:
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
PY
)"

IFS=',' read -r PHYSICAL_GPU APP_SM_MHZ APP_MEM_MHZ CUR_SM_MHZ CUR_MEM_MHZ <<<"$clock_csv"

CSV_OUT="${OUT_STRUCT}/${RUN_ID}_vllm.csv"
echo "run_id,node,model,prompt_len,gen_len,concurrency,tok_per_s,p50_ms,p99_ms,physical_gpu,app_sm_mhz,app_mem_mhz,cur_sm_mhz,cur_mem_mhz" > "$CSV_OUT"

IFS=',' read -r -a CONC_ARR <<<"$CONCURRENCY_LIST"

for conc in "${CONC_ARR[@]}"; do
  thr_json="${OUT_RAW}/${RUN_ID}_vllm_thr_c${conc}.json"
  lat_json="${OUT_RAW}/${RUN_ID}_vllm_lat_c${conc}.json"

  echo "== vLLM throughput (concurrency=${conc}) =="
  "$VLLM_BIN" bench throughput \
    --backend vllm \
    --dataset-name random \
    --random-input-len "$PROMPT_LEN" \
    --random-output-len "$GEN_LEN" \
    --random-prefix-len 0 \
    --num-prompts "$conc" \
    --model "$MODEL" \
    --output-json "$thr_json"

  echo "== vLLM latency (concurrency=${conc}) =="
  "$VLLM_BIN" bench latency \
    --model "$MODEL" \
    --input-len "$PROMPT_LEN" \
    --output-len "$GEN_LEN" \
    --batch-size "$conc" \
    --num-iters-warmup "$WARMUP" \
    --num-iters "$ITERS" \
    --output-json "$lat_json"

  "$VENV_PY" - <<'PY' "$RUN_ID" "$MODEL" "$PROMPT_LEN" "$GEN_LEN" "$conc" "$thr_json" "$lat_json" "$(hostname)" "$CSV_OUT" "$PHYSICAL_GPU" "$APP_SM_MHZ" "$APP_MEM_MHZ" "$CUR_SM_MHZ" "$CUR_MEM_MHZ"
import json
import sys

run_id, model, prompt_len, gen_len, conc, thr_path, lat_path, node, csv_out, physical_gpu, app_sm_mhz, app_mem_mhz, cur_sm_mhz, cur_mem_mhz = sys.argv[1:]

with open(thr_path, "r", encoding="utf-8") as f:
    thr = json.load(f)
with open(lat_path, "r", encoding="utf-8") as f:
    lat = json.load(f)

tok_per_s = thr.get("tokens_per_second")
percentiles = lat.get("percentiles", {})
p50 = percentiles.get("50")
p99 = percentiles.get("99")

def to_ms(val):
    return None if val is None else float(val) * 1000.0

row = [
    run_id,
    node,
    model,
    prompt_len,
    gen_len,
    conc,
    f"{tok_per_s:.6f}" if tok_per_s is not None else "",
    f"{to_ms(p50):.6f}" if p50 is not None else "",
    f"{to_ms(p99):.6f}" if p99 is not None else "",
    physical_gpu,
    app_sm_mhz,
    app_mem_mhz,
    cur_sm_mhz,
    cur_mem_mhz,
]

with open(csv_out, "a", encoding="utf-8") as f:
    f.write(",".join(map(str, row)) + "\n")
PY
done

echo "Wrote ${CSV_OUT}"
