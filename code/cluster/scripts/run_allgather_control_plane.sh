#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_allgather_control_plane.sh --hosts <h1,h2,...> [options]

Compares control-plane collectives used for completion signaling:
  - all_gather_object (python objects)
  - all_gather tensor
  - all_reduce tensor

Outputs:
  results/structured/<run_id>_<label>.json

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --label <label>        Output label (default: allgather_control_plane)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo's root)
  --gpus-per-node <n>    GPUs per node (default: all visible GPUs)
  --iters <n>            Measured iterations per method (default: 2000)
  --warmup <n>           Warmup iterations per method (default: 200)
  --socket-ifname <if>   NCCL_SOCKET_IFNAME (default: --oob-if or $NCCL_SOCKET_IFNAME)
  --nccl-ib-hca <list>   NCCL_IB_HCA (default: $NCCL_IB_HCA)
  --oob-if <iface>       Optional bootstrap iface fallback for --socket-ifname
  --rdzv-port <port>     Rendezvous port (default: 29503)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_host_runtime_env.sh
source "${ROOT_DIR}/scripts/lib_host_runtime_env.sh"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
LABEL="allgather_control_plane"
HOSTS=""
SSH_USER="ubuntu"
REMOTE_ROOT=""
GPUS_PER_NODE=""
ITERS="2000"
WARMUP="200"
SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
IB_HCA="${NCCL_IB_HCA:-}"
SSH_KEY="${SSH_KEY:-}"
OOB_IF="${OOB_IF:-}"
RDZV_PORT="29503"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --socket-ifname) SOCKET_IFNAME="$2"; shift 2 ;;
    --nccl-ib-hca) IB_HCA="$2"; shift 2 ;;
    --oob-if) OOB_IF="$2"; shift 2 ;;
    --rdzv-port) RDZV_PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "$SOCKET_IFNAME" ]]; then
  SOCKET_IFNAME="$OOB_IF"
fi

if [[ -z "$REMOTE_ROOT" ]]; then
  REMOTE_ROOT="$ROOT_DIR"
fi
source_host_runtime_env_if_present "$ROOT_DIR"
resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"

if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi

if ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --gpus-per-node must be a positive integer (got: $GPUS_PER_NODE)" >&2
  exit 2
fi
if ! [[ "$ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --iters must be a positive integer (got: $ITERS)" >&2
  exit 2
fi
if ! [[ "$WARMUP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --warmup must be >= 0 (got: $WARMUP)" >&2
  exit 2
fi

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
if [[ "${#HOST_ARR[@]}" -lt 1 ]]; then
  echo "ERROR: --hosts must contain at least one host" >&2
  exit 2
fi
for i in "${!HOST_ARR[@]}"; do
  HOST_ARR[$i]="$(echo "${HOST_ARR[$i]}" | xargs)"
done

NNODES="${#HOST_ARR[@]}"
MASTER_ADDR="${HOST_ARR[0]}"
OUT_STRUCT="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}/${RUN_ID}_${LABEL}.json"
OUT_STRUCT_REL="${OUT_STRUCT#${ROOT_DIR}/}"
OUT_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}"
mkdir -p "${CLUSTER_STRUCTURED_DIR_EFFECTIVE}" "${OUT_RAW_DIR}"
REMOTE_OUT_STRUCT="$(cluster_structured_dir_for_root "${REMOTE_ROOT}" "${RUN_ID}")/${RUN_ID}_${LABEL}.json"

DEVICE_LIST="$(seq 0 $((GPUS_PER_NODE - 1)) | paste -sd, -)"

is_local_host() {
  local host="$1"
  local h_full
  h_full="$(hostname -f 2>/dev/null || hostname)"
  [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$(hostname)" || "$host" == "$(hostname -s)" || "$host" == "$h_full" ]]
}

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

echo "========================================"
echo "All-gather control-plane benchmark"
echo "  RUN_ID: ${RUN_ID}"
echo "  LABEL: ${LABEL}"
echo "  Hosts: ${HOSTS}"
echo "  NNODES: ${NNODES}"
echo "  GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "  Iters: ${ITERS} (warmup: ${WARMUP})"
echo "  Output: ${OUT_STRUCT_REL}"
if [[ -n "$SOCKET_IFNAME" ]]; then
  echo "  NCCL_SOCKET_IFNAME: ${SOCKET_IFNAME}"
fi
if [[ -n "$IB_HCA" ]]; then
  echo "  NCCL_IB_HCA: ${IB_HCA}"
fi
echo "========================================"

PIDS=()
fail=0
HOST_RUNTIME_REMOTE_PREFIX="$(host_runtime_remote_prefix "$REMOTE_ROOT")"
REMOTE_ARTIFACT_ENV="$(cluster_artifact_env_prefix_for_root "${REMOTE_ROOT}" "${RUN_ID}")"
for idx in "${!HOST_ARR[@]}"; do
  host="${HOST_ARR[$idx]}"
  log_path="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_node${idx}.log"

  launch_cmd=(
    "${REMOTE_ROOT}/scripts/run_with_gpu_clocks.sh"
    --devices "${DEVICE_LIST}"
    --
    "${REMOTE_ROOT}/env/venv/bin/torchrun"
    --nproc_per_node="${GPUS_PER_NODE}"
    --nnodes="${NNODES}"
    --node_rank="${idx}"
    --rdzv_backend=c10d
    --rdzv_endpoint="${MASTER_ADDR}:${RDZV_PORT}"
    --max_restarts=0
    "${REMOTE_ROOT}/scripts/allgather_control_plane_bench.py"
    --iters "${ITERS}"
    --warmup "${WARMUP}"
    --output "${REMOTE_OUT_STRUCT}"
  )

  launch_str="$(printf '%q ' "${launch_cmd[@]}")"
  env_prefix="cd $(printf '%q' "${REMOTE_ROOT}") && ${HOST_RUNTIME_REMOTE_PREFIX}${REMOTE_ARTIFACT_ENV} NCCL_DEBUG=WARN"
  if [[ -n "$SOCKET_IFNAME" ]]; then
    env_prefix+=" NCCL_SOCKET_IFNAME=$(printf '%q' "${SOCKET_IFNAME}") GLOO_SOCKET_IFNAME=$(printf '%q' "${SOCKET_IFNAME}")"
  fi
  if [[ -n "$IB_HCA" ]]; then
    env_prefix+=" NCCL_IB_HCA=$(printf '%q' "${IB_HCA}")"
  fi
  remote_cmd="${env_prefix} RUN_ID=$(printf '%q' "${RUN_ID}") LABEL=$(printf '%q' "${LABEL}_node${idx}") ${launch_str}"

  echo "Launching node_rank=${idx} on host=${host} -> ${log_path}"
  if is_local_host "$host"; then
    bash -lc "$remote_cmd" 2>&1 | tee "$log_path" &
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "bash -lc $(printf '%q' "$remote_cmd")" 2>&1 | tee "$log_path" &
  fi
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "ERROR: control-plane benchmark failed; check logs under ${OUT_RAW_DIR}." >&2
  exit 1
fi

if [[ ! -f "$OUT_STRUCT" ]]; then
  if ! is_local_host "${MASTER_ADDR}"; then
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${MASTER_ADDR}:${REMOTE_OUT_STRUCT}" "$OUT_STRUCT" >/dev/null
  fi
fi

if [[ ! -f "$OUT_STRUCT" ]]; then
  echo "ERROR: expected output not found: ${OUT_STRUCT}" >&2
  exit 1
fi

echo ""
echo "All-gather control-plane benchmark complete."
echo "Output: ${OUT_STRUCT_REL}"
