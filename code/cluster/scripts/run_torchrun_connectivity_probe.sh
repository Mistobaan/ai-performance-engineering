#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_torchrun_connectivity_probe.sh [options]

Fast distributed connectivity gate for NCCL readiness.

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --ssh-user <user>      SSH user (default: ubuntu)
  --gpus-per-node <n>    GPUs per node (default: all visible GPUs)
  --cuda-visible-devices <list>  Optional CUDA_VISIBLE_DEVICES list for torchrun
  --master-addr <ip>     Rendezvous address (default: first host)
  --master-port <port>   Rendezvous port (default: 29504)
  --oob-if <iface>       Socket interface for torchrun OOB (sets NCCL/GLOO *_SOCKET_IFNAME)
  --nccl-ib-hca <list>   NCCL_IB_HCA allowlist
  --nccl-debug <level>   NCCL_DEBUG for probe ranks (default: WARN)
  --nccl-debug-subsys <s>  NCCL_DEBUG_SUBSYS for probe ranks (default: INIT,NET)
  --barrier-iters <n>    Barrier timing iterations per rank (default: 5)
  --payload-bytes <n>    Payload bytes for all-reduce sanity (default: 8388608)
  --timeout-sec <n>      Distributed timeout in seconds (default: 120)
  --ssh-key <path>       SSH key for remote launch (default: $SSH_KEY)
  -h, --help             Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_host_runtime_env.sh
source "${ROOT_DIR}/scripts/lib_host_runtime_env.sh"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="$(date +%Y-%m-%d)"
HOSTS=""
SSH_USER="ubuntu"
GPUS_PER_NODE=""
CUDA_VISIBLE_DEVICES_LIST=""
MASTER_ADDR=""
MASTER_PORT="29504"
OOB_IF="${OOB_IF:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
NCCL_DEBUG_LEVEL="${NCCL_DEBUG:-WARN}"
NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
BARRIER_ITERS="5"
PAYLOAD_BYTES="8388608"
TIMEOUT_SEC="120"
SSH_KEY="${SSH_KEY:-}"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_LIST="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --oob-if) OOB_IF="$2"; shift 2 ;;
    --nccl-ib-hca) NCCL_IB_HCA="$2"; shift 2 ;;
    --nccl-debug) NCCL_DEBUG_LEVEL="$2"; shift 2 ;;
    --nccl-debug-subsys) NCCL_DEBUG_SUBSYS="$2"; shift 2 ;;
    --barrier-iters) BARRIER_ITERS="$2"; shift 2 ;;
    --payload-bytes) PAYLOAD_BYTES="$2"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi
if ! [[ "$BARRIER_ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --barrier-iters must be a positive integer (got: ${BARRIER_ITERS})" >&2
  exit 2
fi
if ! [[ "$PAYLOAD_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --payload-bytes must be a positive integer (got: ${PAYLOAD_BYTES})" >&2
  exit 2
fi
if ! [[ "$TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --timeout-sec must be a positive integer (got: ${TIMEOUT_SEC})" >&2
  exit 2
fi
if ! [[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --master-port must be a positive integer (got: ${MASTER_PORT})" >&2
  exit 2
fi

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
if [[ "${#HOST_ARR[@]}" -eq 0 ]]; then
  echo "ERROR: --hosts is required" >&2
  exit 2
fi

if [[ -z "$MASTER_ADDR" ]]; then
  MASTER_ADDR="${HOST_ARR[0]}"
fi
if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi
source_host_runtime_env_if_present "$ROOT_DIR"
resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
if ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --gpus-per-node must be a positive integer (got: ${GPUS_PER_NODE})" >&2
  exit 2
fi

OUT_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}"
OUT_STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_RAW_DIR" "$OUT_STRUCT_DIR"
OUTPUT_JSON="${OUT_STRUCT_DIR}/${RUN_ID}_torchrun_connectivity_probe.json"

TORCHRUN_BIN="${ROOT_DIR}/env/venv/bin/torchrun"
SCRIPT_PATH="${ROOT_DIR}/scripts/torchrun_connectivity_probe.py"
if [[ ! -x "$TORCHRUN_BIN" ]]; then
  echo "ERROR: missing torchrun binary at ${TORCHRUN_BIN}" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: missing connectivity probe script: ${SCRIPT_PATH}" >&2
  exit 1
fi

SSH_OPTS=(
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o IdentityAgent=none
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=8
  -o ConnectionAttempts=3
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
)
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

echo "== torchrun connectivity probe =="
echo "RUN_ID=${RUN_ID}"
echo "HOSTS=${HOSTS}"
echo "MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
if [[ -n "$CUDA_VISIBLE_DEVICES_LIST" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST}"
fi
if [[ -n "$OOB_IF" ]]; then
  echo "SOCKET_IFNAME=${OOB_IF}"
fi
if [[ -n "$NCCL_IB_HCA" ]]; then
  echo "NCCL_IB_HCA=${NCCL_IB_HCA}"
fi

REMOTE_ENV=(
  "RUN_ID=${RUN_ID}"
  "NCCL_DEBUG=${NCCL_DEBUG_LEVEL}"
  "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
)
if [[ -n "$CUDA_VISIBLE_DEVICES_LIST" ]]; then
  REMOTE_ENV+=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST}")
fi
if [[ -n "$OOB_IF" ]]; then
  REMOTE_ENV+=("NCCL_SOCKET_IFNAME=${OOB_IF}" "GLOO_SOCKET_IFNAME=${OOB_IF}")
fi
if [[ -n "$NCCL_IB_HCA" ]]; then
  REMOTE_ENV+=("NCCL_IB_HCA=${NCCL_IB_HCA}")
fi

HOST_RUNTIME_REMOTE_PREFIX="$(host_runtime_remote_prefix "$ROOT_DIR")"

PIDS=()
for idx in "${!HOST_ARR[@]}"; do
  host="${HOST_ARR[$idx]}"
  log_path="${OUT_RAW_DIR}/${RUN_ID}_torchrun_connectivity_probe_node${idx}.log"
  cmd=(
    "$TORCHRUN_BIN"
    --nnodes "${#HOST_ARR[@]}"
    --nproc_per_node "$GPUS_PER_NODE"
    --node_rank "$idx"
    --rdzv_backend c10d
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}"
    --rdzv_id "${RUN_ID}_connectivity_probe"
    "$SCRIPT_PATH"
    --barrier-iters "$BARRIER_ITERS"
    --payload-bytes "$PAYLOAD_BYTES"
    --timeout-sec "$TIMEOUT_SEC"
    --output "$OUTPUT_JSON"
  )

  remote_env="${REMOTE_ENV[*]}"
  if [[ -n "$remote_env" ]]; then
    remote_env="env ${remote_env}"
  fi
  remote_cmd="cd ${ROOT_DIR} && ${HOST_RUNTIME_REMOTE_PREFIX}${remote_env} $(printf "%q " "${cmd[@]}")"
  echo "Launching ${host} -> ${log_path}"
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "bash -lc $(printf '%q' "$remote_cmd")" 2>&1 | tee "$log_path" &
  PIDS+=($!)
done

exit_code=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    exit_code=1
  fi
done

if [[ "$exit_code" -ne 0 ]]; then
  echo "ERROR: connectivity probe failed; inspect logs under ${OUT_RAW_DIR}" >&2
  exit "$exit_code"
fi

if [[ ! -f "$OUTPUT_JSON" ]]; then
  echo "ERROR: expected connectivity probe output missing: ${OUTPUT_JSON}" >&2
  exit 1
fi

echo "Wrote ${OUTPUT_JSON}"
