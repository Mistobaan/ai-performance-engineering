#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_nccl_algo_comparison.sh --hosts <h1,h2,...> [options]

Compares NCCL algorithms (Ring, Tree, NVLS) explicitly by running
nccl-tests all_reduce_perf with NCCL_ALGO forced for each algorithm.

This reveals:
- Whether NCCL's auto-selection is optimal for your topology
- Which algorithm wins at different message sizes (Tree at small, Ring at large)
- The NVLS speedup (NVLink SHARP) vs traditional algorithms
- Potential routing or topology issues that favor one algorithm

Outputs:
  runs/<run_id>/structured/<run_id>_nccl_algo_<algo>.json  (per algorithm)
  runs/<run_id>/structured/<run_id>_nccl_algo_comparison.json  (summary)

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --gpus-per-node <n>    GPUs per node (default: all visible GPUs)
  --min-bytes <size>     Minimum message size (default: 1K)
  --max-bytes <size>     Maximum message size (default: 16G)
  --warmup <n>           Warmup iterations (default: 5)
  --iters <n>            Measurement iterations (default: 20)
  --algos <list>         Comma-separated algos to test (default: Ring,Tree,NVLS,auto)
  --socket-ifname <if>   NCCL_SOCKET_IFNAME
  --nccl-ib-hca <list>   NCCL_IB_HCA
  --ssh-key <path>       SSH key
  --oob-if <iface>       OOB interface
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_host_runtime_env.sh
source "${ROOT_DIR}/scripts/lib_host_runtime_env.sh"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
source_host_runtime_env_if_present "$ROOT_DIR"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
HOSTS=""
GPUS_PER_NODE=""
MIN_BYTES="1K"
MAX_BYTES="16G"
WARMUP=5
ITERS=20
ALGOS="Ring,Tree,NVLS,auto"
SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
IB_HCA="${NCCL_IB_HCA:-}"
SSH_KEY="${SSH_KEY:-}"
OOB_IF="${OOB_IF:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --min-bytes) MIN_BYTES="$2"; shift 2 ;;
    --max-bytes) MAX_BYTES="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --algos) ALGOS="$2"; shift 2 ;;
    --socket-ifname) SOCKET_IFNAME="$2"; shift 2 ;;
    --nccl-ib-hca) IB_HCA="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --oob-if) OOB_IF="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
IFS=',' read -r -a ALGO_ARR <<<"$ALGOS"
TOTAL_RANKS=$((GPUS_PER_NODE * ${#HOST_ARR[@]}))

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}"
OUT_STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_RAW_DIR" "$OUT_STRUCT_DIR"

HOSTFILE="${OUT_RAW_DIR}/${RUN_ID}_nccl_algo_comparison_hosts.txt"
printf "%s\n" "${HOST_ARR[@]}" > "$HOSTFILE"

MPI_PY="${ROOT_DIR}/env/venv/bin/python"
WRAPPER="${ROOT_DIR}/scripts/nccl_lock_wrapper.py"
NCCL_BIN="${ROOT_DIR}/tools/nccl-tests/build/all_reduce_perf"

if [[ ! -x "$MPI_PY" ]]; then
  echo "Missing python venv at $MPI_PY" >&2; exit 1
fi
if [[ ! -x "$NCCL_BIN" ]]; then
  echo "Missing nccl-tests binary at $NCCL_BIN" >&2; exit 1
fi

echo "========================================"
echo "NCCL Algorithm Comparison"
echo "  RUN_ID: ${RUN_ID}"
echo "  Hosts: ${HOSTS}"
echo "  GPUs/node: ${GPUS_PER_NODE}, Total ranks: ${TOTAL_RANKS}"
echo "  Algorithms: ${ALGOS}"
echo "  Message range: ${MIN_BYTES} - ${MAX_BYTES}"
echo "========================================"

SUMMARY_FILE="${OUT_STRUCT_DIR}/${RUN_ID}_nccl_algo_comparison.json"
ALGO_RESULTS=()

for algo in "${ALGO_ARR[@]}"; do
  algo_clean="$(echo "$algo" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
  algo_env="$(echo "$algo" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"
  LABEL="nccl_algo_${algo_clean}"
  RAW_LOG="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}.log"
  STRUCT_OUT="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}.json"

  echo ""
  echo "== Testing algorithm: ${algo} =="

  MPIRUN_CMD=(
    mpirun
    --hostfile "$HOSTFILE"
    --map-by "ppr:${GPUS_PER_NODE}:node"
    --bind-to none
    --mca routed direct
  )

  if [[ -n "$SSH_KEY" ]]; then
    MPIRUN_CMD+=(
      --mca plm_rsh_args
      "-i ${SSH_KEY} -o BatchMode=yes -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=8 -o ConnectionAttempts=3"
    )
  fi
  if [[ -n "$OOB_IF" ]]; then
    MPIRUN_CMD+=(--mca oob_tcp_if_include "$OOB_IF" --mca btl_tcp_if_include "$OOB_IF")
  fi

  MPIRUN_CMD+=(
    -np "$TOTAL_RANKS"
    -x "NCCL_DEBUG=WARN"
    -x "PATH=$PATH"
    -x "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    -x "REQUIRE_CLOCK_LOCK=1"
  )

  # Force algorithm (unless "auto")
  if [[ "$algo_clean" != "auto" ]]; then
    MPIRUN_CMD+=(-x "NCCL_ALGO=${algo_env}")
  fi

  if [[ -n "$SOCKET_IFNAME" ]]; then
    MPIRUN_CMD+=(-x "NCCL_SOCKET_IFNAME=${SOCKET_IFNAME}")
  fi
  if [[ -n "$IB_HCA" ]]; then
    MPIRUN_CMD+=(-x "NCCL_IB_HCA=${IB_HCA}")
  fi

  MPIRUN_CMD+=(
    "$MPI_PY"
    "$WRAPPER"
    --
    "$NCCL_BIN"
    -b "$MIN_BYTES"
    -e "$MAX_BYTES"
    -f 2
    -g 1
    -w "$WARMUP"
    -n "$ITERS"
  )

  set +e
  "${MPIRUN_CMD[@]}" 2>&1 | tee "$RAW_LOG"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "$rc" -ne 0 ]]; then
    echo "WARNING: Algorithm ${algo} failed (rc=${rc}). Some algorithms may not be supported." >&2
    ALGO_RESULTS+=("{\"algo\": \"${algo}\", \"status\": \"failed\", \"exit_code\": ${rc}}")
    continue
  fi

  CMD_STRING=$(printf "%q " "${MPIRUN_CMD[@]}")
  "${ROOT_DIR}/scripts/parse_nccl_log.py" \
    --input "$RAW_LOG" \
    --output "$STRUCT_OUT" \
    --run-id "$RUN_ID" \
    --hosts "$HOSTS" \
    --gpus-per-node "$GPUS_PER_NODE" \
    --command "$CMD_STRING" || {
    echo "WARNING: Failed to parse NCCL log for ${algo}" >&2
    ALGO_RESULTS+=("{\"algo\": \"${algo}\", \"status\": \"parse_failed\"}")
    continue
  }

  # Extract peak busbw from structured output
  peak_busbw=$(python3 -c "
import json
d = json.load(open('${STRUCT_OUT}'))
results = d.get('results', [])
if results:
    peak = max(r.get('busbw_gbps', 0) for r in results)
    print(f'{peak:.2f}')
else:
    print('0.00')
" 2>/dev/null || echo "0.00")

  echo "  ${algo}: peak busbw = ${peak_busbw} GB/s"
  ALGO_RESULTS+=("{\"algo\": \"${algo}\", \"status\": \"ok\", \"peak_busbw_gbps\": ${peak_busbw}, \"output\": \"${STRUCT_OUT}\"}")
done

# Write comparison summary
python3 - "$SUMMARY_FILE" "$RUN_ID" "$HOSTS" "$GPUS_PER_NODE" "$TOTAL_RANKS" "${ALGO_RESULTS[@]}" <<'PYEOF'
import json
import sys
out_path = sys.argv[1]
run_id = sys.argv[2]
hosts = [h.strip() for h in sys.argv[3].split(",") if h.strip()]
gpus_per_node = int(sys.argv[4])
total_ranks = int(sys.argv[5])
algo_entries = [json.loads(x) for x in sys.argv[6:]]
summary = {
    "run_id": run_id,
    "hosts": hosts,
    "gpus_per_node": gpus_per_node,
    "total_ranks": total_ranks,
    "algorithms_tested": algo_entries,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote: {out_path}")
PYEOF

echo ""
echo "========================================"
echo "NCCL Algorithm Comparison Complete"
echo "  Summary: ${SUMMARY_FILE}"
echo "========================================"
