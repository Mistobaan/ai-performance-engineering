#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a vLLM online-serving concurrency sweep in the official vLLM container.

This reproduces the "vLLM Concurrency Sweep Results" and per-concurrency
"Serving Benchmark Result" screenshots (vllm bench serve).

Usage:
  scripts/repro/run_vllm_serve_sweep_container.sh [options]

Options:
  --run-id <id>              RUN_ID prefix (default: YYYY-MM-DD)
  --label <label>            Label for output paths (default: hostname)
  --model <hf_model_id>      (default: openai/gpt-oss-120b)
  --tp <n>                   Tensor parallel size (default: all visible GPUs)
  --isl <n>                  Input sequence length (default: 1024)
  --osl <n>                  Output sequence length (default: 1024)
  --concurrency-range "..."  Space-separated concurrencies (default: "32 64 128 256 512")
  --repeats <n>              Number of full sweep repetitions (default: 1)
  --port <port>              (default: 8888)
  --image <docker_image>     (default: auto by architecture)
  --detach                   Start the sweep container in detached mode.
                             (Default: run attached and stream logs.)
  --allow-existing-vllm-procs
                             Allow pre-existing VLLM::EngineCore GPU processes.
                             (Default: fail-fast for canonical benchmark hygiene.)

Env:
  HF_TOKEN can be set to enable gated model downloads.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="$(date +%Y-%m-%d)"
LABEL="$(hostname)"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
TP="${TP:-}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
PORT="${PORT:-8888}"
CONCURRENCY_RANGE="${CONCURRENCY_RANGE:-32 64 128 256 512}"
REPEATS="${REPEATS:-1}"
DETACH=0
ALLOW_EXISTING_VLLM_PROCS=0

ORIG_ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --isl) ISL="$2"; shift 2 ;;
    --osl) OSL="$2"; shift 2 ;;
    --concurrency-range) CONCURRENCY_RANGE="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --image) CONTAINER_IMAGE="$2"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    --allow-existing-vllm-procs) ALLOW_EXISTING_VLLM_PROCS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found" >&2
  exit 1
fi
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --repeats must be a positive integer (got: ${REPEATS})" >&2
  exit 2
fi

DOCKER_CMD=(docker)
if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    DOCKER_CMD=(sudo -n docker)
    echo "INFO: docker socket requires elevated access; using non-interactive sudo docker."
  else
    echo "ERROR: unable to access docker daemon as current user, and sudo -n docker is unavailable." >&2
    echo "Fix: grant docker group access or configure non-interactive sudo for docker." >&2
    exit 1
  fi
fi

docker_exec() {
  "${DOCKER_CMD[@]}" "$@"
}

ARCH="$(uname -m)"
if [[ -z "$CONTAINER_IMAGE" ]]; then
  if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    CONTAINER_IMAGE="vllm/vllm-openai:cu130-nightly-aarch64"
  else
    CONTAINER_IMAGE="vllm/vllm-openai:cu130-nightly"
  fi
fi

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ -z "$TP" ]]; then
  TP="$GPU_COUNT"
fi
if [[ "$TP" -gt "$GPU_COUNT" ]]; then
  echo "WARNING: Requested TP=$TP but only $GPU_COUNT GPUs available. Using TP=$GPU_COUNT" >&2
  TP="$GPU_COUNT"
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"

OUT_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}/${RUN_ID}_${LABEL}_vllm_serve_sweep"
mkdir -p "$OUT_DIR"

MAX_MODEL_LEN=$((ISL + OSL + 256))

STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$STRUCT_DIR"
LOCK_META_OUT="${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep_clock_lock.json"

# vLLM is executed via Docker + NVIDIA runtime. Ensure nvidia-persistenced is
# running so its socket exists for container mount hooks.
if [[ ! -S "/run/nvidia-persistenced/socket" ]]; then
  if command -v systemctl >/dev/null 2>&1; then
    if sudo -n true >/dev/null 2>&1; then
      sudo systemctl start nvidia-persistenced >/dev/null 2>&1 || true
    fi
  fi
fi
if [[ ! -S "/run/nvidia-persistenced/socket" ]]; then
  echo "ERROR: /run/nvidia-persistenced/socket is missing." >&2
  echo "This is required for running the vLLM container with GPU access." >&2
  echo "Fix: sudo systemctl start nvidia-persistenced" >&2
  exit 1
fi

if [[ "$ALLOW_EXISTING_VLLM_PROCS" -ne 1 ]]; then
  preexisting_vllm="$(python3 - <<'PY'
import subprocess

try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
except Exception:
    raise SystemExit(0)

rows = []
for line in out.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        continue
    pid, name, used_mem = parts[0], parts[1], parts[2]
    if name == "VLLM::EngineCore":
        rows.append(f"pid={pid} used_memory_mib={used_mem}")

if rows:
    print("\n".join(rows))
PY
)"
  if [[ -n "$preexisting_vllm" ]]; then
    echo "ERROR: detected pre-existing VLLM::EngineCore GPU processes before vLLM sweep." >&2
    echo "$preexisting_vllm" >&2
    echo "Fix: terminate stale processes (for example: pkill -f VLLM::EngineCore) and rerun." >&2
    echo "Override this guard only when intentionally co-running workloads: --allow-existing-vllm-procs" >&2
    exit 3
  fi
fi

# Enforce strict GPU clock locking for the entire sweep (server + benchmarks).
if [[ "${AISP_CLOCK_LOCKED:-}" != "1" ]]; then
  export RUN_ID LABEL
  exec "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --lock-meta-out "$LOCK_META_OUT" \
    -- "$0" "${ORIG_ARGS[@]}"
fi

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
TIKTOKEN_CACHE="${HOME}/.cache/tiktoken_rs"
VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$HOME/.cache/vllm}"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-$HOME/.cache/flashinfer}"
mkdir -p "$HF_CACHE_DIR" "$TIKTOKEN_CACHE" "$VLLM_CACHE_DIR" "$FLASHINFER_CACHE_DIR"

# Harmony vocab (for gpt-oss / o200k_base) used by tiktoken-rs.
TIKTOKEN_VOCAB_FILE="${TIKTOKEN_CACHE}/fb374d419588a4632f3f557e76b4b70aebbca790"
if [[ ! -f "$TIKTOKEN_VOCAB_FILE" ]]; then
  echo "Downloading harmony tiktoken vocab file to ${TIKTOKEN_VOCAB_FILE}..."
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "$TIKTOKEN_VOCAB_FILE" https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken || true
  else
    curl -fsSL https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken -o "$TIKTOKEN_VOCAB_FILE" || true
  fi
fi

HF_MOUNT=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_MOUNT+=(-e "HF_TOKEN=${HF_TOKEN}" -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
fi
VLLM_ENV=()
for k in VLLM_SERVE_ENFORCE_EAGER VLLM_KV_CACHE_MEMORY_BYTES VLLM_GPU_MEMORY_UTILIZATION VLLM_SERVER_READY_TIMEOUT VLLM_SWEEP_NUM_PROMPTS_MULTIPLIER VLLM_SWEEP_MAX_POINTS_PER_RUN; do
  if [[ -n "${!k:-}" ]]; then
    VLLM_ENV+=(-e "${k}=${!k}")
  fi
done

read -r -a CONC_ARR <<<"$CONCURRENCY_RANGE"

echo "========================================"
echo "vLLM Concurrency Sweep (Containerized)"
echo "========================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Architecture: $(uname -m)"
echo "Container: $CONTAINER_IMAGE"
echo ""
echo "Model: $MODEL"
echo "TP: $TP"
echo "ISL: $ISL"
echo "OSL: $OSL"
echo "Max model len: $MAX_MODEL_LEN"
echo "Concurrency range: ${CONC_ARR[*]}"
echo "Repeats: $REPEATS"
echo "Output dir: $OUT_DIR"
echo ""

if docker_exec image inspect "$CONTAINER_IMAGE" >/dev/null 2>&1; then
  echo "Using cached container image: ${CONTAINER_IMAGE}"
else
  echo "Pulling container (best-effort)..."
  docker_exec pull "$CONTAINER_IMAGE" 2>/dev/null || echo "WARNING: docker pull failed; attempting to continue with local cache"
fi

INNER="${ROOT_DIR}/scripts/repro/vllm_serve_sweep_inner.sh"
if [[ ! -f "$INNER" ]]; then
  echo "Missing inner script at ${INNER}" >&2
  exit 1
fi

LOG_PATH="${OUT_DIR}/sweep_log.txt"
AGG_CSV="${OUT_DIR}/sweep_summary.csv"
AGG_JSONL="${OUT_DIR}/sweep_summary.jsonl"
AGG_SUMMARY_TXT="${OUT_DIR}/summary.txt"
AGG_STABILITY_JSON="${OUT_DIR}/sweep_stability.json"
PROGRESS_JSON="${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep_progress.json"
REPEAT_CSVS=()

if [[ "$DETACH" -eq 1 && "$REPEATS" -gt 1 ]]; then
  echo "ERROR: --detach is only supported with --repeats 1." >&2
  exit 2
fi

write_progress_json() {
  local status="$1"
  local current_repeat="${2:-0}"
  local current_csv="${3:-}"
  python3 - "$PROGRESS_JSON" "$RUN_ID" "$LABEL" "$status" "$REPEATS" "$current_repeat" "$current_csv" "$OUT_DIR" "${CONC_ARR[@]}" <<'PY'
import csv
import json
import pathlib
import sys
from datetime import datetime, timezone

out_path, run_id, label, status, repeats_total, current_repeat, current_csv, out_dir, *points = sys.argv[1:]
repeats_total = int(repeats_total)
current_repeat = int(current_repeat)
points = [str(p) for p in points]
out_dir_path = pathlib.Path(out_dir)

repeats_completed = 0
points_completed_total = 0
current_repeat_points_completed = 0
last_completed_point = None

for rep_dir in sorted(out_dir_path.glob("repeat_*")):
    csv_path = rep_dir / "sweep_summary.csv"
    if not csv_path.exists():
        continue
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    row_count = len(rows)
    rep_num = int(rep_dir.name.split("_")[-1])
    points_completed_total += row_count
    if row_count >= len(points):
        repeats_completed += 1
    if rep_num == current_repeat:
        current_repeat_points_completed = row_count
    if rows:
        last_row = rows[-1]
        last_completed_point = {
            "repeat": rep_num,
            "concurrency": last_row.get("concurrency"),
            "completed": int(float(last_row.get("completed", "0") or 0)),
            "failed": int(float(last_row.get("failed", "0") or 0)),
            "total_token_throughput": float(last_row.get("total_token_throughput", "0") or 0),
        }

payload = {
    "run_id": run_id,
    "label": label,
    "mode": "vllm_serve_sweep",
    "status": status,
    "repeats_total": repeats_total,
    "repeats_completed": repeats_completed,
    "current_repeat": current_repeat,
    "points_expected_per_repeat": len(points),
    "point_values": points,
    "points_completed_total": points_completed_total,
    "current_repeat_points_completed": current_repeat_points_completed,
    "last_completed_point": last_completed_point,
    "raw_output_dir": str(out_dir_path),
    "current_repeat_csv": current_csv or None,
    "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
pathlib.Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

write_progress_json "starting" 0 ""

DOCKER_ARGS=(
  --gpus all
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --network host
  -e TIKTOKEN_RS_CACHE_DIR=/root/.cache/tiktoken_rs
  "${HF_MOUNT[@]}"
  "${VLLM_ENV[@]}"
  -v "$INNER":/sweep.sh:ro
  -v "$HF_CACHE_DIR":/root/.cache/huggingface
  -v "$TIKTOKEN_CACHE":/root/.cache/tiktoken_rs
  -v "$VLLM_CACHE_DIR":/root/.cache/vllm
  -v "$FLASHINFER_CACHE_DIR":/root/.cache/flashinfer
  --entrypoint bash
)

rm -f "$LOG_PATH"

for rep in $(seq 1 "$REPEATS"); do
  REP_DIR="${OUT_DIR}/repeat_${rep}"
  mkdir -p "$REP_DIR"
  REP_LOG="${REP_DIR}/sweep_log.txt"
  write_progress_json "running" "$rep" "${REP_DIR}/sweep_summary.csv"
  if [[ "$DETACH" -eq 1 ]]; then
    safe_name="$(echo "vllm_sweep_${RUN_ID}_${LABEL}_r${rep}" | tr -c '[:alnum:]_.' '_' )_$(date +%s)"
    echo "Starting detached container (repeat ${rep}/${REPEATS}): ${safe_name}"
    docker_exec run -d --name "$safe_name" \
      "${DOCKER_ARGS[@]}" \
      -v "$REP_DIR":/results \
      "$CONTAINER_IMAGE" \
      -lc "/sweep.sh \"$MODEL\" \"$TP\" \"$ISL\" \"$OSL\" \"$MAX_MODEL_LEN\" \"$PORT\" \"/results\" ${CONC_ARR[*]} > /results/sweep_log.txt 2>&1"
    tail -n +1 -F "$REP_LOG" &
    TAIL_PID=$!
    rc="$(docker_exec wait "$safe_name")"
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
    docker_exec rm "$safe_name" >/dev/null 2>&1 || true
    if [[ "$rc" -ne 0 ]]; then
      write_progress_json "failed" "$rep" "${REP_DIR}/sweep_summary.csv"
      echo "ERROR: vLLM sweep container exited with code ${rc} on repeat ${rep}" >&2
      exit "$rc"
    fi
  else
    echo "=== Repeat ${rep}/${REPEATS}: vLLM concurrency sweep ==="
    set +e
    docker_exec run --rm \
      "${DOCKER_ARGS[@]}" \
      -v "$REP_DIR":/results \
      "$CONTAINER_IMAGE" \
      /sweep.sh \
        "$MODEL" "$TP" "$ISL" "$OSL" "$MAX_MODEL_LEN" "$PORT" \
        "/results" "${CONC_ARR[@]}" 2>&1 | tee "$REP_LOG"
    rc=${PIPESTATUS[0]}
    set -e
    if [[ "$rc" -ne 0 ]]; then
      write_progress_json "failed" "$rep" "${REP_DIR}/sweep_summary.csv"
      echo "ERROR: vLLM sweep container failed on repeat ${rep} (rc=${rc})." >&2
      if [[ -f "${REP_DIR}/server.log" ]]; then
        echo "---- ${REP_DIR}/server.log (last 120 lines) ----" >&2
        tail -n 120 "${REP_DIR}/server.log" >&2 || true
      fi
      exit "$rc"
    fi
  fi

  if [[ ! -f "${REP_DIR}/sweep_summary.csv" ]]; then
    write_progress_json "failed" "$rep" "${REP_DIR}/sweep_summary.csv"
    echo "ERROR: missing repeat CSV output: ${REP_DIR}/sweep_summary.csv" >&2
    exit 1
  fi
  REPEAT_CSVS+=("${REP_DIR}/sweep_summary.csv")
  write_progress_json "running" "$rep" "${REP_DIR}/sweep_summary.csv"
  {
    echo "========================================"
    echo "Repeat ${rep}/${REPEATS}"
    echo "========================================"
    cat "$REP_LOG"
    echo
  } >>"$LOG_PATH"
done

write_progress_json "aggregating" "$REPEATS" "${OUT_DIR}/repeat_${REPEATS}/sweep_summary.csv"
python3 "${ROOT_DIR}/analysis/aggregate_vllm_repeat_csv.py" \
  --mode concurrency \
  --inputs "${REPEAT_CSVS[@]}" \
  --output-csv "$AGG_CSV" \
  --output-jsonl "$AGG_JSONL" \
  --output-stability-json "$AGG_STABILITY_JSON" \
  --output-summary-txt "$AGG_SUMMARY_TXT"

if [[ -f "$AGG_SUMMARY_TXT" ]]; then
  cp -f "$AGG_SUMMARY_TXT" "${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_concurrency_sweep_summary.txt"
  echo "Wrote ${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_concurrency_sweep_summary.txt"
fi
if [[ -f "$AGG_CSV" ]]; then
  cp -f "$AGG_CSV" "${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep.csv"
  echo "Wrote ${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep.csv"
fi
if [[ -f "$AGG_JSONL" ]]; then
  cp -f "$AGG_JSONL" "${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep.jsonl"
  echo "Wrote ${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep.jsonl"
fi
if [[ -f "$AGG_STABILITY_JSON" ]]; then
  cp -f "$AGG_STABILITY_JSON" "${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep_stability.json"
  echo "Wrote ${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep_stability.json"
fi

write_progress_json "complete" "$REPEATS" "$AGG_CSV"

echo "Wrote ${LOG_PATH}"
