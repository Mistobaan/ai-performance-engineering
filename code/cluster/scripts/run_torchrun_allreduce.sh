#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_torchrun_allreduce.sh [options]

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --gpus-per-node <n>    GPUs per node (default: all visible GPUs)
  --cuda-visible-devices <list>  Optional CUDA_VISIBLE_DEVICES list for torchrun
                                 (example: 1,2,3 to exclude GPU0)
  --master-addr <ip>     Rendezvous address (default: first host)
  --master-port <port>   Rendezvous port (default: 29500)
  --oob-if <iface>       Socket interface for torchdist OOB (sets NCCL/GLOO *_SOCKET_IFNAME)
  --nccl-ib-hca <list>   Comma-separated IB HCA allowlist (sets NCCL_IB_HCA)
  --nccl-nvls-enable <0|1|2>  Export NCCL_NVLS_ENABLE to torchrun processes
                            (default: unset)
  --nccl-debug <level>   Set NCCL_DEBUG for torchrun processes (example: INFO)
  --nccl-debug-subsys <s>  Set NCCL_DEBUG_SUBSYS (example: INIT,NET)
  --warmup <n>           Warmup iterations (default: 5)
  --iters <n>            Measurement iterations (default: 20)
  --sizes <bytes,...>    Comma-separated message sizes in bytes
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
GPUS_PER_NODE=""
CUDA_VISIBLE_DEVICES_LIST=""
MASTER_ADDR=""
MASTER_PORT=29500
OOB_IF="${OOB_IF:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
NCCL_NVLS_ENABLE=""
NCCL_DEBUG_LEVEL=""
NCCL_DEBUG_SUBSYS=""
WARMUP=5
ITERS=20
SIZES=""
SSH_KEY="${SSH_KEY:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --hosts)
      HOSTS="$2"
      shift 2
      ;;
    --gpus-per-node)
      GPUS_PER_NODE="$2"
      shift 2
      ;;
    --cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_LIST="$2"
      shift 2
      ;;
    --master-addr)
      MASTER_ADDR="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --oob-if)
      OOB_IF="$2"
      shift 2
      ;;
    --nccl-ib-hca)
      NCCL_IB_HCA="$2"
      shift 2
      ;;
    --nccl-nvls-enable)
      NCCL_NVLS_ENABLE="$2"
      shift 2
      ;;
    --nccl-debug)
      NCCL_DEBUG_LEVEL="$2"
      shift 2
      ;;
    --nccl-debug-subsys)
      NCCL_DEBUG_SUBSYS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --sizes)
      SIZES="$2"
      shift 2
      ;;
    --ssh-key)
      SSH_KEY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
if [[ "${#HOST_ARR[@]}" -eq 0 ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

if [[ -n "$NCCL_NVLS_ENABLE" && "$NCCL_NVLS_ENABLE" != "0" && "$NCCL_NVLS_ENABLE" != "1" && "$NCCL_NVLS_ENABLE" != "2" ]]; then
  echo "ERROR: --nccl-nvls-enable must be 0, 1, or 2 (got: ${NCCL_NVLS_ENABLE})" >&2
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
OUT_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}"
OUT_STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_RAW_DIR" "$OUT_STRUCT_DIR"

SCRIPT_PATH="${ROOT_DIR}/scripts/torchrun_allreduce_sanity.py"
TORCHRUN_BIN="${ROOT_DIR}/env/venv/bin/torchrun"

if [[ ! -x "$TORCHRUN_BIN" ]]; then
  echo "Missing torchrun at $TORCHRUN_BIN" >&2
  exit 1
fi

OUTPUT_JSON="${OUT_STRUCT_DIR}/${RUN_ID}_torchrun_allreduce.json"

REMOTE_ENV=(
  "RUN_ID=${RUN_ID}"
  "WARMUP_ITERS=${WARMUP}"
  "MEASURE_ITERS=${ITERS}"
)
if [[ -n "$CUDA_VISIBLE_DEVICES_LIST" ]]; then
  REMOTE_ENV+=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST}")
fi
if [[ -n "$NCCL_DEBUG_LEVEL" ]]; then
  REMOTE_ENV+=("NCCL_DEBUG=${NCCL_DEBUG_LEVEL}")
fi
if [[ -n "$NCCL_DEBUG_SUBSYS" ]]; then
  REMOTE_ENV+=("NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}")
fi
if [[ -n "$OOB_IF" ]]; then
  REMOTE_ENV+=("NCCL_SOCKET_IFNAME=${OOB_IF}" "GLOO_SOCKET_IFNAME=${OOB_IF}")
fi
if [[ -n "$NCCL_IB_HCA" ]]; then
  REMOTE_ENV+=("NCCL_IB_HCA=${NCCL_IB_HCA}")
fi
if [[ -n "$NCCL_NVLS_ENABLE" ]]; then
  REMOTE_ENV+=("NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE}")
fi
if [[ -n "$SIZES" ]]; then
  REMOTE_ENV+=("SIZES_BYTES=${SIZES}")
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

HOST_RUNTIME_REMOTE_PREFIX="$(host_runtime_remote_prefix "$ROOT_DIR")"

for host in "${HOST_ARR[@]}"; do
  if ! ssh "${SSH_OPTS[@]}" "ubuntu@${host}" "sudo -n true >/dev/null"; then
    echo "ERROR: GPU clock locking is required, but passwordless sudo is not available on ${host}." >&2
    echo "Fix: configure sudoers for this user so `sudo -n true` succeeds, then re-run." >&2
    exit 3
  fi
done

echo "== Torchrun all-reduce sanity =="
echo "RUN_ID=$RUN_ID"
echo "HOSTS=$HOSTS"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
if [[ -n "$NCCL_DEBUG_LEVEL" ]]; then echo "NCCL_DEBUG=${NCCL_DEBUG_LEVEL}"; fi
if [[ -n "$NCCL_DEBUG_SUBSYS" ]]; then echo "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"; fi
if [[ -n "$NCCL_NVLS_ENABLE" ]]; then echo "NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE}"; fi

PIDS=()
for idx in "${!HOST_ARR[@]}"; do
  host="${HOST_ARR[$idx]}"
  log_path="${OUT_RAW_DIR}/${RUN_ID}_torchrun_node${idx}.log"
  cmd=(
    "$TORCHRUN_BIN"
    --nnodes "${#HOST_ARR[@]}"
    --nproc_per_node "$GPUS_PER_NODE"
    --node_rank "$idx"
    --rdzv_backend c10d
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}"
    --rdzv_id "${RUN_ID}"
    "$SCRIPT_PATH"
  )

  echo "Launching ${host} via SSH -> ${log_path}"
  remote_env="${REMOTE_ENV[*]}"
  if [[ -n "$remote_env" ]]; then
    remote_env="env ${remote_env}"
  fi
  if [[ "$idx" -eq 0 ]]; then
    remote_env="${remote_env} OUTPUT_JSON=${OUTPUT_JSON}"
  fi
  ssh "${SSH_OPTS[@]}" "ubuntu@${host}" "cd ${ROOT_DIR} && ${HOST_RUNTIME_REMOTE_PREFIX}${remote_env} ${cmd[*]}" 2>&1 | tee "$log_path" &
  PIDS+=($!)
done

EXIT_CODE=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    EXIT_CODE=1
  fi
done

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "torchrun all-reduce failed; check logs in ${OUT_RAW_DIR}" >&2
  exit "$EXIT_CODE"
fi

echo "Wrote ${OUTPUT_JSON}"
