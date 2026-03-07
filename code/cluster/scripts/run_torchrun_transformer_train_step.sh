#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_torchrun_transformer_train_step.sh [options]

Runs a small end-to-end training step benchmark (forward+backward+optimizer) via
torchrun across 1+ nodes. This is intended to complement nccl-tests by
measuring real training-step behavior (compute + comm overlap).

Outputs:
  results/structured/<run_id>_<label>_torchrun_train_step.json

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --label <label>        Label for output filenames (default: hosts concatenated)
  --gpus-per-node <n>    GPUs per node (default: all visible GPUs)
  --cuda-visible-devices <list>  Optional CUDA_VISIBLE_DEVICES list for torchrun
                                 (example: 1,2,3 to exclude GPU0)
  --master-addr <ip>     Rendezvous address (default: first host)
  --master-port <port>   Rendezvous port (default: 29510)
  --oob-if <iface>       Socket interface for torchdist OOB (sets NCCL/GLOO *_SOCKET_IFNAME)
  --nccl-ib-hca <list>   Comma-separated IB HCA allowlist (sets NCCL_IB_HCA)
  --nccl-nvls-enable <0|1|2>  Export NCCL_NVLS_ENABLE to torchrun processes
  --nccl-debug <level>   NCCL_DEBUG (example: INFO)
  --nccl-debug-subsys <s>  NCCL_DEBUG_SUBSYS (example: INIT,NET)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key for remote launch (default: $SSH_KEY)

Training config:
  --steps <n>            Measured steps (default: 30)
  --warmup-steps <n>     Warmup steps (default: 5)
  --batch-size <n>       Per-rank batch size (default: 2)
  --seq-len <n>          Sequence length (default: 2048)
  --hidden <n>           Hidden size (default: 4096)
  --layers <n>           Layers (default: 24)
  --heads <n>            Heads (default: 32)
  --mlp-ratio <n>        MLP ratio (default: 4)
  --precision <bf16|fp16> (default: bf16)
  --fsdp <0|1>           Enable FSDP FULL_SHARD (default: 1)
  --lr <float>           AdamW learning rate (default: 1e-4)
  -h, --help             Show help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_host_runtime_env.sh
source "${ROOT_DIR}/scripts/lib_host_runtime_env.sh"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="$(date +%Y-%m-%d)"
HOSTS=""
LABEL=""
GPUS_PER_NODE=""
CUDA_VISIBLE_DEVICES_LIST=""
MASTER_ADDR=""
MASTER_PORT=29510
OOB_IF="${OOB_IF:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
NCCL_NVLS_ENABLE=""
NCCL_DEBUG_LEVEL=""
NCCL_DEBUG_SUBSYS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"

STEPS=30
WARMUP_STEPS=5
BATCH_SIZE=2
SEQ_LEN=2048
HIDDEN=4096
LAYERS=24
HEADS=32
MLP_RATIO=4
PRECISION="bf16"
FSDP=1
LR="1e-4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_LIST="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --oob-if) OOB_IF="$2"; shift 2 ;;
    --nccl-ib-hca) NCCL_IB_HCA="$2"; shift 2 ;;
    --nccl-nvls-enable) NCCL_NVLS_ENABLE="$2"; shift 2 ;;
    --nccl-debug) NCCL_DEBUG_LEVEL="$2"; shift 2 ;;
    --nccl-debug-subsys) NCCL_DEBUG_SUBSYS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --hidden) HIDDEN="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --heads) HEADS="$2"; shift 2 ;;
    --mlp-ratio) MLP_RATIO="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --fsdp) FSDP="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
if [[ "${#HOST_ARR[@]}" -eq 0 ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "$LABEL" ]]; then
  LABEL="${HOSTS//,/}"
  LABEL="${LABEL//./_}"
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

SCRIPT_PATH="${ROOT_DIR}/scripts/torchrun_transformer_train_step.py"
TORCHRUN_BIN="${ROOT_DIR}/env/venv/bin/torchrun"

if [[ ! -x "$TORCHRUN_BIN" ]]; then
  echo "Missing torchrun at $TORCHRUN_BIN" >&2
  exit 1
fi

OUTPUT_JSON="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_torchrun_train_step.json"

REMOTE_ENV=()
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
REMOTE_ARTIFACT_ENV="$(cluster_artifact_env_prefix_for_root "$ROOT_DIR" "$RUN_ID")"

is_local_host() {
  local host="$1"
  local h short fqdn
  h="$(hostname)"
  short="$(hostname -s 2>/dev/null || true)"
  fqdn="$(hostname -f 2>/dev/null || true)"
  [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "::1" || "$host" == "$h" || "$host" == "$short" || "$host" == "$fqdn" ]]
}

for host in "${HOST_ARR[@]}"; do
  if is_local_host "$host"; then
    if ! sudo -n true >/dev/null 2>&1; then
      echo "ERROR: GPU clock locking is required, but passwordless sudo is not available on local host ${host}." >&2
      echo "Fix: configure sudoers for this user so `sudo -n true` succeeds, then re-run." >&2
      exit 3
    fi
  else
    if ! ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "sudo -n true >/dev/null"; then
      echo "ERROR: GPU clock locking is required, but passwordless sudo is not available on ${host}." >&2
      echo "Fix: configure sudoers for this user so `sudo -n true` succeeds, then re-run." >&2
      exit 3
    fi
  fi
done

echo "== Torchrun training step benchmark =="
echo "RUN_ID=$RUN_ID"
echo "LABEL=$LABEL"
echo "HOSTS=$HOSTS"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
if [[ -n "$CUDA_VISIBLE_DEVICES_LIST" ]]; then echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST}"; fi
if [[ -n "$OOB_IF" ]]; then echo "OOB_IF=${OOB_IF}"; fi
if [[ -n "$NCCL_IB_HCA" ]]; then echo "NCCL_IB_HCA=${NCCL_IB_HCA}"; fi
if [[ -n "$NCCL_NVLS_ENABLE" ]]; then echo "NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE}"; fi
echo "Training: steps=${STEPS} warmup=${WARMUP_STEPS} bs=${BATCH_SIZE} sl=${SEQ_LEN} hidden=${HIDDEN} layers=${LAYERS} heads=${HEADS} mlp_ratio=${MLP_RATIO} precision=${PRECISION} fsdp=${FSDP} lr=${LR}"
echo "OUT=${OUTPUT_JSON}"

PIDS=()
for idx in "${!HOST_ARR[@]}"; do
  host="${HOST_ARR[$idx]}"
  log_path="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_torchrun_train_step_node${idx}.log"
  cmd=(
    "$TORCHRUN_BIN"
    --nnodes "${#HOST_ARR[@]}"
    --nproc_per_node "$GPUS_PER_NODE"
    --node_rank "$idx"
    --rdzv_backend c10d
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}"
    --rdzv_id "${RUN_ID}_${LABEL}"
    "$SCRIPT_PATH"
    --run-id "$RUN_ID"
    --label "$LABEL"
    --output-json "$OUTPUT_JSON"
    --steps "$STEPS"
    --warmup-steps "$WARMUP_STEPS"
    --batch-size "$BATCH_SIZE"
    --seq-len "$SEQ_LEN"
    --hidden "$HIDDEN"
    --layers "$LAYERS"
    --heads "$HEADS"
    --mlp-ratio "$MLP_RATIO"
    --precision "$PRECISION"
    --fsdp "$FSDP"
    --lr "$LR"
  )

  if is_local_host "$host"; then
    echo "Launching ${host} locally -> ${log_path}"
  else
    echo "Launching ${host} via SSH -> ${log_path}"
  fi
  remote_env="${REMOTE_ENV[*]}"
  if [[ -n "$remote_env" ]]; then
    remote_env="env ${REMOTE_ARTIFACT_ENV} ${remote_env}"
  else
    remote_env="env ${REMOTE_ARTIFACT_ENV}"
  fi
  if is_local_host "$host"; then
    if [[ -n "$remote_env" ]]; then
      bash -lc "cd ${ROOT_DIR} && ${HOST_RUNTIME_REMOTE_PREFIX}${remote_env} ${cmd[*]}" 2>&1 | tee "$log_path" &
    else
      bash -lc "cd ${ROOT_DIR} && ${HOST_RUNTIME_REMOTE_PREFIX}${cmd[*]}" 2>&1 | tee "$log_path" &
    fi
    PIDS+=($!)
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "cd ${ROOT_DIR} && ${HOST_RUNTIME_REMOTE_PREFIX}${remote_env} ${cmd[*]}" 2>&1 | tee "$log_path" &
    PIDS+=($!)
  fi
done

EXIT_CODE=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    EXIT_CODE=1
  fi
done

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "torchrun train-step benchmark failed; check logs in ${OUT_RAW_DIR}" >&2
  exit "$EXIT_CODE"
fi

echo "Wrote ${OUTPUT_JSON}"
