#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run a 2-node vLLM online-serving benchmark (Ray + tensor parallel across nodes)
with strict GPU clock locking on both leader and worker nodes.

Usage:
  scripts/repro/run_vllm_serve_multinode_container.sh --hosts <leader,worker> [options]

Options:
  --run-id <id>                  RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2>                Required. Exactly two hosts (leader first)
  --labels <l1,l2>               Optional labels matching --hosts
  --ssh-user <user>              SSH user for worker host (default: ubuntu)
  --ssh-key <path>               SSH key path (default: $SSH_KEY)

  --model <hf_model_id>          Model id (default: openai/gpt-oss-120b)
  --tp <n>                       Tensor parallel degree (default: all visible GPUs across both nodes)
  --isl <n>                      Input sequence length (default: 1024)
  --osl <n>                      Output sequence length (default: 1024)
  --concurrency <n>              Max concurrency for vllm bench serve (default: 64)
  --num-prompts <n>              Number of prompts (default: concurrency * 10)
  --port <port>                  vLLM serving port on leader (default: 8888)
  --ray-port <port>              Ray head port on leader (default: 6379)
  --image <docker_image>         vLLM image (default: auto by architecture)
                                 Tag refs are auto-pinned to a single repo digest across leader/worker

  --socket-ifname <iface>        NCCL socket interface (optional)
  --gloo-socket-ifname <iface>   Gloo socket interface (default: socket-ifname)
  --nccl-ib-hca <list>           NCCL_IB_HCA allowlist (optional)
  --nccl-cross-nic <0|1>         NCCL_CROSS_NIC (default: 1)

  --ray-ready-timeout <sec>      Timeout waiting for Ray cluster (default: 300)
  --server-ready-timeout <sec>   Timeout waiting for vLLM health (default: 1200)
  --worker-startup-wait <sec>    Delay before leader launch after worker start (default: 10)

Artifacts:
  - runs/<run_id>/raw/<run_id>_<leader_label>_vllm_multinode_serve/
  - runs/<run_id>/structured/<run_id>_<leader_label>_vllm_multinode_serve.json
  - runs/<run_id>/structured/<run_id>_<leader_label>_vllm_multinode_serve.csv
  - runs/<run_id>/structured/<run_id>_<leader_label>_vllm_multinode_serve.jsonl
  - runs/<run_id>/structured/<run_id>_<leader_label>_vllm_multinode_leader_clock_lock.json
  - runs/<run_id>/structured/<run_id>_<worker_label>_vllm_multinode_worker_clock_lock.json
USAGE
}

trim() {
  local s="$1"
  s="${s#${s%%[![:space:]]*}}"
  s="${s%${s##*[![:space:]]}}"
  printf '%s' "$s"
}

sanitize() {
  local s="$1"
  s="${s//[^A-Za-z0-9_.-]/_}"
  printf '%s' "$s"
}

image_repo_no_tag() {
  local ref="$1"
  local tail=""
  ref="${ref%%@*}"
  tail="${ref##*/}"
  if [[ "$tail" == *:* ]]; then
    ref="${ref%:*}"
  fi
  printf '%s' "$ref"
}

resolve_pinned_image() {
  local requested="$1"
  local expected_repo=""
  local digests=""
  local line=""

  if [[ "$requested" == *@sha256:* ]]; then
    printf '%s\n' "$requested"
    return 0
  fi

  docker pull "$requested" >/dev/null 2>&1 || true

  digests="$(docker image inspect "$requested" --format '{{range .RepoDigests}}{{println .}}{{end}}' 2>/dev/null || true)"
  if [[ -z "$digests" ]]; then
    echo "ERROR: could not resolve a repo digest for image '$requested'." >&2
    echo "Use --image <repo@sha256:...> to enforce a fixed digest explicitly." >&2
    return 1
  fi

  expected_repo="$(image_repo_no_tag "$requested")"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    if [[ "$line" == "${expected_repo}@sha256:"* ]]; then
      printf '%s\n' "$line"
      return 0
    fi
  done <<<"$digests"

  printf '%s\n' "$(printf '%s\n' "$digests" | head -n 1)"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="$(date +%Y-%m-%d)"
HOSTS=""
LABELS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
TP="${TP:-}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONCURRENCY="${CONCURRENCY:-64}"
NUM_PROMPTS="${NUM_PROMPTS:-}"
PORT="${PORT:-8888}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_CLUSTER_SIZE=2

SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
GLOO_SOCKET_IFNAME=""
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"

RAY_READY_TIMEOUT="300"
SERVER_READY_TIMEOUT="1200"
WORKER_STARTUP_WAIT="10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;

    --model) MODEL="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --isl) ISL="$2"; shift 2 ;;
    --osl) OSL="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --ray-port) RAY_PORT="$2"; shift 2 ;;
    --image) CONTAINER_IMAGE="$2"; shift 2 ;;

    --socket-ifname) SOCKET_IFNAME="$2"; shift 2 ;;
    --gloo-socket-ifname) GLOO_SOCKET_IFNAME="$2"; shift 2 ;;
    --nccl-ib-hca) NCCL_IB_HCA="$2"; shift 2 ;;
    --nccl-cross-nic) NCCL_CROSS_NIC="$2"; shift 2 ;;

    --ray-ready-timeout) RAY_READY_TIMEOUT="$2"; shift 2 ;;
    --server-ready-timeout) SERVER_READY_TIMEOUT="$2"; shift 2 ;;
    --worker-startup-wait) WORKER_STARTUP_WAIT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required." >&2
  exit 2
fi

if ! [[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --concurrency must be >= 1." >&2
  exit 2
fi
if [[ -n "$NUM_PROMPTS" ]] && ! [[ "$NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --num-prompts must be >= 1." >&2
  exit 2
fi
for int_field in PORT RAY_PORT RAY_READY_TIMEOUT SERVER_READY_TIMEOUT WORKER_STARTUP_WAIT; do
  if ! [[ "${!int_field}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --${int_field,,} must be >= 1 (got: ${!int_field})" >&2
    exit 2
  fi
done
if ! [[ "$NCCL_CROSS_NIC" =~ ^[01]$ ]]; then
  echo "ERROR: --nccl-cross-nic must be 0 or 1." >&2
  exit 2
fi

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
if [[ "${#HOST_ARR[@]}" -ne 2 ]]; then
  echo "ERROR: --hosts must contain exactly 2 hosts (leader,worker)." >&2
  exit 2
fi

for i in "${!HOST_ARR[@]}"; do
  HOST_ARR[$i]="$(trim "${HOST_ARR[$i]}")"
done

LEADER_HOST="${HOST_ARR[0]}"
WORKER_HOST="${HOST_ARR[1]}"

LEADER_LABEL="$(sanitize "$LEADER_HOST")"
WORKER_LABEL="$(sanitize "$WORKER_HOST")"
if [[ -n "$LABELS" ]]; then
  IFS=',' read -r -a LABEL_ARR <<<"$LABELS"
  if [[ "${#LABEL_ARR[@]}" -ne 2 ]]; then
    echo "ERROR: --labels must contain exactly 2 labels when provided." >&2
    exit 2
  fi
  LEADER_LABEL="$(sanitize "$(trim "${LABEL_ARR[0]}")")"
  WORKER_LABEL="$(sanitize "$(trim "${LABEL_ARR[1]}")")"
fi

if [[ -z "$GLOO_SOCKET_IFNAME" ]]; then
  GLOO_SOCKET_IFNAME="$SOCKET_IFNAME"
fi

if [[ -z "$NUM_PROMPTS" ]]; then
  NUM_PROMPTS="$((CONCURRENCY * 10))"
fi

if [[ -z "$CONTAINER_IMAGE" ]]; then
  ARCH="$(uname -m)"
  if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    CONTAINER_IMAGE="vllm/vllm-openai:cu130-nightly-aarch64"
  else
    CONTAINER_IMAGE="vllm/vllm-openai:cu130-nightly"
  fi
fi

if [[ -z "$TP" ]]; then
  LOCAL_GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
  if ! [[ "$LOCAL_GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: unable to detect local GPU count for default TP." >&2
    exit 1
  fi
  TP="$((LOCAL_GPU_COUNT * RAY_CLUSTER_SIZE))"
fi
if ! [[ "$TP" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --tp must be >= 1." >&2
  exit 2
fi

MAX_MODEL_LEN="$((ISL + OSL + 256))"

INNER_SCRIPT="${ROOT_DIR}/scripts/repro/vllm_multinode_inner.sh"
if [[ ! -f "$INNER_SCRIPT" ]]; then
  echo "ERROR: missing inner script: $INNER_SCRIPT" >&2
  exit 1
fi

if [[ ! -S "/run/nvidia-persistenced/socket" ]]; then
  if command -v systemctl >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo systemctl start nvidia-persistenced >/dev/null 2>&1 || true
  fi
fi
if [[ ! -S "/run/nvidia-persistenced/socket" ]]; then
  echo "ERROR: /run/nvidia-persistenced/socket is missing on leader node." >&2
  echo "Fix: sudo systemctl start nvidia-persistenced" >&2
  exit 1
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}/${RUN_ID}_${LEADER_LABEL}_vllm_multinode_serve"
STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_RAW_DIR" "$STRUCT_DIR"

LEADER_LOG="${OUT_RAW_DIR}/leader.log"
WORKER_LOG="${OUT_RAW_DIR}/worker_${WORKER_LABEL}.log"
SERVER_LOG="${OUT_RAW_DIR}/server.log"
BENCH_LOG="${OUT_RAW_DIR}/bench.log"
RESULT_JSON_RAW="${OUT_RAW_DIR}/bench_result.json"

SUMMARY_JSON="${STRUCT_DIR}/${RUN_ID}_${LEADER_LABEL}_vllm_multinode_serve.json"
SUMMARY_CSV="${STRUCT_DIR}/${RUN_ID}_${LEADER_LABEL}_vllm_multinode_serve.csv"
SUMMARY_JSONL="${STRUCT_DIR}/${RUN_ID}_${LEADER_LABEL}_vllm_multinode_serve.jsonl"
RESULT_JSON_STRUCT="${STRUCT_DIR}/${RUN_ID}_${LEADER_LABEL}_vllm_multinode_serve_result.json"

LEADER_LOCK_META="${STRUCT_DIR}/${RUN_ID}_${LEADER_LABEL}_vllm_multinode_leader_clock_lock.json"
WORKER_LOCK_META="${STRUCT_DIR}/${RUN_ID}_${WORKER_LABEL}_vllm_multinode_worker_clock_lock.json"
REMOTE_WORKER_LOCK="/tmp/${RUN_ID}_${WORKER_LABEL}_vllm_multinode_worker_clock_lock.json"

WORKER_CONTAINER_NAME="vllm_multinode_worker_$(sanitize "${RUN_ID}_${WORKER_LABEL}")"
LEADER_CONTAINER_NAME="vllm_multinode_leader_$(sanitize "${RUN_ID}_${LEADER_LABEL}")"

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
TIKTOKEN_CACHE="${HOME}/.cache/tiktoken_rs"
VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$HOME/.cache/vllm}"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-$HOME/.cache/flashinfer}"
mkdir -p "$HF_CACHE_DIR" "$TIKTOKEN_CACHE" "$VLLM_CACHE_DIR" "$FLASHINFER_CACHE_DIR"

TIKTOKEN_VOCAB_FILE="${TIKTOKEN_CACHE}/fb374d419588a4632f3f557e76b4b70aebbca790"
if [[ ! -f "$TIKTOKEN_VOCAB_FILE" ]]; then
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "$TIKTOKEN_VOCAB_FILE" https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken || true
  else
    curl -fsSL https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken -o "$TIKTOKEN_VOCAB_FILE" || true
  fi
fi

SSH_BASE=(ssh -o BatchMode=yes)
if [[ -n "$SSH_KEY" ]]; then
  SSH_BASE+=( -i "$SSH_KEY" )
fi
WORKER_TARGET="${SSH_USER}@${WORKER_HOST}"

REQUESTED_IMAGE="$CONTAINER_IMAGE"
PINNED_IMAGE="$(resolve_pinned_image "$CONTAINER_IMAGE")"
if [[ -z "$PINNED_IMAGE" ]]; then
  echo "ERROR: unable to resolve pinned image digest from ${CONTAINER_IMAGE}" >&2
  exit 1
fi
PINNED_DIGEST="${PINNED_IMAGE##*@}"
if [[ "$PINNED_DIGEST" != sha256:* ]]; then
  echo "ERROR: resolved pinned image does not contain sha256 digest: ${PINNED_IMAGE}" >&2
  exit 1
fi

LEADER_IMAGE_ID="$(docker image inspect "$PINNED_IMAGE" --format '{{.Id}}' 2>/dev/null || true)"
LEADER_IMAGE_REPODIGEST="$(docker image inspect "$PINNED_IMAGE" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"
if [[ -z "$LEADER_IMAGE_REPODIGEST" || "$LEADER_IMAGE_REPODIGEST" != *"@${PINNED_DIGEST}" ]]; then
  echo "ERROR: leader image digest verification failed for ${PINNED_IMAGE}" >&2
  echo "leader repo digest: ${LEADER_IMAGE_REPODIGEST:-<missing>}" >&2
  exit 1
fi

if ! "${SSH_BASE[@]}" "$WORKER_TARGET" "docker pull '$PINNED_IMAGE'" >"$WORKER_LOG" 2>&1; then
  echo "ERROR: worker failed to pull pinned image ${PINNED_IMAGE}" >&2
  echo "worker pull log: ${WORKER_LOG}" >&2
  exit 1
fi
WORKER_IMAGE_ID="$("${SSH_BASE[@]}" "$WORKER_TARGET" "docker image inspect '$PINNED_IMAGE' --format '{{.Id}}'" 2>/dev/null || true)"
WORKER_IMAGE_REPODIGEST="$("${SSH_BASE[@]}" "$WORKER_TARGET" "docker image inspect '$PINNED_IMAGE' --format '{{index .RepoDigests 0}}'" 2>/dev/null || true)"
if [[ -z "$WORKER_IMAGE_REPODIGEST" || "$WORKER_IMAGE_REPODIGEST" != *"@${PINNED_DIGEST}" ]]; then
  echo "ERROR: worker image digest verification failed for ${PINNED_IMAGE}" >&2
  echo "worker repo digest: ${WORKER_IMAGE_REPODIGEST:-<missing>}" >&2
  exit 1
fi

WORKER_REMOTE_SCRIPT="$(mktemp)"
cat > "$WORKER_REMOTE_SCRIPT" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$1"
LOCK_META="$2"
IMAGE="$3"
MASTER_ADDR="$4"
RAY_PORT="$5"
SOCKET_IFNAME="$6"
GLOO_IFNAME="$7"
CROSS_NIC="$8"
IB_HCA="$9"
CONTAINER_NAME="${10}"

cd "$ROOT_DIR"

if [[ ! -S "/run/nvidia-persistenced/socket" ]]; then
  if command -v systemctl >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo systemctl start nvidia-persistenced >/dev/null 2>&1 || true
  fi
fi

cmd=(docker run --rm --name "$CONTAINER_NAME" --gpus all --privileged --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864)
if [[ -d /dev/infiniband ]]; then
  cmd+=( -v /dev/infiniband:/dev/infiniband )
fi
if [[ -e /dev/nvidia_imex ]]; then
  cmd+=( -v /dev/nvidia_imex:/dev/nvidia_imex )
fi
if [[ -n "$SOCKET_IFNAME" ]]; then
  cmd+=( -e "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME" )
fi
if [[ -n "$GLOO_IFNAME" ]]; then
  cmd+=( -e "GLOO_SOCKET_IFNAME=$GLOO_IFNAME" )
fi
if [[ -n "$IB_HCA" ]]; then
  cmd+=( -e "NCCL_IB_HCA=$IB_HCA" )
fi
cmd+=( -e "NCCL_CROSS_NIC=$CROSS_NIC" )
cmd+=( --entrypoint bash "$IMAGE" -lc "ray start --address=${MASTER_ADDR}:${RAY_PORT} --block" )

RUN_ID="${RUN_ID:-}" LABEL="${LABEL:-}" "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" --lock-meta-out "$LOCK_META" -- "${cmd[@]}"
WORKER
chmod +x "$WORKER_REMOTE_SCRIPT"

WORKER_SSH_PID=""
WORKER_STOP_REQUESTED=0

stop_remote_worker() {
  local force_kill="${1:-0}"
  if [[ -z "${WORKER_SSH_PID:-}" ]]; then
    return 0
  fi
  if ! kill -0 "$WORKER_SSH_PID" >/dev/null 2>&1; then
    return 0
  fi

  WORKER_STOP_REQUESTED=1
  "${SSH_BASE[@]}" "$WORKER_TARGET" "docker exec '$WORKER_CONTAINER_NAME' bash -lc 'ray stop --force >/dev/null 2>&1 || true'" >/dev/null 2>&1 || true
  "${SSH_BASE[@]}" "$WORKER_TARGET" "docker stop -t 30 '$WORKER_CONTAINER_NAME' >/dev/null 2>&1 || true" >/dev/null 2>&1 || true

  local waited=0
  while kill -0 "$WORKER_SSH_PID" >/dev/null 2>&1 && [[ "$waited" -lt 30 ]]; do
    sleep 1
    waited=$((waited + 1))
  done

  if [[ "$force_kill" -eq 1 ]] && kill -0 "$WORKER_SSH_PID" >/dev/null 2>&1; then
    kill "$WORKER_SSH_PID" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  local rc=$?
  set +e
  if [[ -n "${WORKER_SSH_PID:-}" ]]; then
    stop_remote_worker 1
    wait "$WORKER_SSH_PID" >/dev/null 2>&1 || true
  fi
  rm -f "$WORKER_REMOTE_SCRIPT"
  exit "$rc"
}
trap cleanup EXIT

echo "========================================"
echo "vLLM Multinode Serving Benchmark"
echo "========================================"
echo "RUN_ID: ${RUN_ID}"
echo "hosts:  leader=${LEADER_HOST} worker=${WORKER_HOST}"
echo "labels: leader=${LEADER_LABEL} worker=${WORKER_LABEL}"
echo "model=${MODEL} tp=${TP} isl=${ISL} osl=${OSL} concurrency=${CONCURRENCY} num_prompts=${NUM_PROMPTS}"
echo "socket_if=${SOCKET_IFNAME:-<unset>} gloo_if=${GLOO_SOCKET_IFNAME:-<unset>} nccl_ib_hca=${NCCL_IB_HCA:-<unset>}"
echo "image_requested=${REQUESTED_IMAGE}"
echo "image_pinned=${PINNED_IMAGE}"
echo "image_digest=${PINNED_DIGEST}"
echo "leader_image_id=${LEADER_IMAGE_ID:-<unknown>} worker_image_id=${WORKER_IMAGE_ID:-<unknown>}"
echo ""

set +e
RUN_ID="$RUN_ID" LABEL="$WORKER_LABEL" \
  "${SSH_BASE[@]}" "$WORKER_TARGET" \
  "bash -s -- '$ROOT_DIR' '$REMOTE_WORKER_LOCK' '$PINNED_IMAGE' '$LEADER_HOST' '$RAY_PORT' '$SOCKET_IFNAME' '$GLOO_SOCKET_IFNAME' '$NCCL_CROSS_NIC' '$NCCL_IB_HCA' '$WORKER_CONTAINER_NAME'" \
  < "$WORKER_REMOTE_SCRIPT" >"$WORKER_LOG" 2>&1 &
WORKER_SSH_PID=$!
set -e

sleep "$WORKER_STARTUP_WAIT"
if ! kill -0 "$WORKER_SSH_PID" >/dev/null 2>&1; then
  echo "WARNING: worker bootstrap process exited early; leader run will likely fail." >&2
fi

LEADER_DOCKER_ARGS=(
  docker run --rm --name "$LEADER_CONTAINER_NAME"
  --gpus all
  --privileged
  --ipc=host
  --network host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -e TIKTOKEN_RS_CACHE_DIR=/root/.cache/tiktoken_rs
  -e RAY_USAGE_STATS_ENABLED=0
)
if [[ -d /dev/infiniband ]]; then
  LEADER_DOCKER_ARGS+=( -v /dev/infiniband:/dev/infiniband )
fi
if [[ -e /dev/nvidia_imex ]]; then
  LEADER_DOCKER_ARGS+=( -v /dev/nvidia_imex:/dev/nvidia_imex )
fi
if [[ -n "$SOCKET_IFNAME" ]]; then
  LEADER_DOCKER_ARGS+=( -e "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME" )
fi
if [[ -n "$GLOO_SOCKET_IFNAME" ]]; then
  LEADER_DOCKER_ARGS+=( -e "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME" )
fi
if [[ -n "$NCCL_IB_HCA" ]]; then
  LEADER_DOCKER_ARGS+=( -e "NCCL_IB_HCA=$NCCL_IB_HCA" )
fi
LEADER_DOCKER_ARGS+=( -e "NCCL_CROSS_NIC=$NCCL_CROSS_NIC" )
if [[ -n "${HF_TOKEN:-}" ]]; then
  LEADER_DOCKER_ARGS+=( -e "HF_TOKEN=${HF_TOKEN}" -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" )
fi
LEADER_DOCKER_ARGS+=(
  -v "$INNER_SCRIPT":/vllm_multinode_inner.sh:ro
  -v "$OUT_RAW_DIR":/results
  -v "$HF_CACHE_DIR":/root/.cache/huggingface
  -v "$TIKTOKEN_CACHE":/root/.cache/tiktoken_rs
  -v "$VLLM_CACHE_DIR":/root/.cache/vllm
  -v "$FLASHINFER_CACHE_DIR":/root/.cache/flashinfer
  --entrypoint bash
  "$PINNED_IMAGE"
  /vllm_multinode_inner.sh
  "$MODEL" "$TP" "$ISL" "$OSL" "$MAX_MODEL_LEN" "$PORT" "$RAY_PORT" "$RAY_CLUSTER_SIZE"
  "$SERVER_READY_TIMEOUT" "$RAY_READY_TIMEOUT" "$CONCURRENCY" "$NUM_PROMPTS"
  /results "$(basename "$RESULT_JSON_RAW")" "/results/$(basename "$SERVER_LOG")" "/results/$(basename "$BENCH_LOG")"
)

set +e
RUN_ID="$RUN_ID" LABEL="$LEADER_LABEL" \
  "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
  --lock-meta-out "$LEADER_LOCK_META" \
  -- "${LEADER_DOCKER_ARGS[@]}" > >(tee "$LEADER_LOG") 2>&1
LEADER_RC=$?
set -e

stop_remote_worker 0

set +e
wait "$WORKER_SSH_PID"
WORKER_RC=$?
set -e
WORKER_SSH_PID=""

WORKER_RC_RAW="$WORKER_RC"
WORKER_RC_NORMALIZED=0
if [[ "$WORKER_STOP_REQUESTED" -eq 1 && "$WORKER_RC" -ne 0 && "$LEADER_RC" -eq 0 && -f "$RESULT_JSON_RAW" ]]; then
  echo "INFO: worker terminated during intentional teardown; normalizing worker return code ${WORKER_RC} -> 0."
  WORKER_RC=0
  WORKER_RC_NORMALIZED=1
fi

if "${SSH_BASE[@]}" "$WORKER_TARGET" "test -f '$REMOTE_WORKER_LOCK'" >/dev/null 2>&1; then
  "${SSH_BASE[@]}" "$WORKER_TARGET" "cat '$REMOTE_WORKER_LOCK'" > "$WORKER_LOCK_META" || true
  "${SSH_BASE[@]}" "$WORKER_TARGET" "rm -f '$REMOTE_WORKER_LOCK'" >/dev/null 2>&1 || true
fi

if [[ "$WORKER_RC_NORMALIZED" -eq 1 && -f "$WORKER_LOCK_META" ]]; then
  python3 - <<'PY' "$WORKER_LOCK_META" "$WORKER_RC_RAW"
import json
import sys
from pathlib import Path

lock_path = Path(sys.argv[1])
raw_rc = int(sys.argv[2])

try:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
except Exception:
    payload = {}

if not isinstance(payload, dict):
    payload = {}

payload["returncode_raw"] = raw_rc
payload["returncode"] = 0
payload["teardown_normalized"] = True
payload["teardown_reason"] = "intentional worker shutdown after successful leader completion"

lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

if [[ -f "$RESULT_JSON_RAW" ]]; then
  cp -f "$RESULT_JSON_RAW" "$RESULT_JSON_STRUCT"
fi

python3 - <<'PY' \
  "$RUN_ID" "$LEADER_HOST" "$WORKER_HOST" "$LEADER_LABEL" "$WORKER_LABEL" \
  "$MODEL" "$TP" "$ISL" "$OSL" "$CONCURRENCY" "$NUM_PROMPTS" \
  "$LEADER_RC" "$WORKER_RC" "$RESULT_JSON_RAW" "$RESULT_JSON_STRUCT" \
  "$LEADER_LOCK_META" "$WORKER_LOCK_META" "$LEADER_LOG" "$WORKER_LOG" "$SERVER_LOG" "$BENCH_LOG" \
  "$SUMMARY_JSON" "$SUMMARY_CSV" "$SUMMARY_JSONL" \
  "$REQUESTED_IMAGE" "$PINNED_IMAGE" "$PINNED_DIGEST" "$LEADER_IMAGE_ID" "$WORKER_IMAGE_ID" \
  "$LEADER_IMAGE_REPODIGEST" "$WORKER_IMAGE_REPODIGEST"
import csv
import json
import sys
from pathlib import Path

(
    run_id,
    leader_host,
    worker_host,
    leader_label,
    worker_label,
    model,
    tp,
    isl,
    osl,
    concurrency,
    num_prompts,
    leader_rc,
    worker_rc,
    raw_result_path,
    struct_result_path,
    leader_lock_path,
    worker_lock_path,
    leader_log,
    worker_log,
    server_log,
    bench_log,
    summary_json,
    summary_csv,
    summary_jsonl,
    requested_image,
    pinned_image,
    pinned_digest,
    leader_image_id,
    worker_image_id,
    leader_image_repodigest,
    worker_image_repodigest,
) = sys.argv[1:]


def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def detect_failure_reason(log_path: str):
    p = Path(log_path)
    if not p.exists():
        return None
    try:
        lines = [line.strip() for line in p.read_text(encoding="utf-8", errors="replace").splitlines()]
        for marker in ("ModuleNotFoundError:", "RuntimeError:", "ERROR:"):
            for text in lines:
                if text and marker in text:
                    return text
    except Exception:
        return None
    return None


def lock_summary(payload):
    out = {
        "path": None,
        "returncode": None,
        "all_devices_locked": False,
        "device_count": 0,
        "application_sm_mhz": [],
        "application_mem_mhz": [],
        "current_sm_mhz": [],
        "current_mem_mhz": [],
    }
    if not isinstance(payload, dict):
        return out
    out["returncode"] = payload.get("returncode")
    locks = payload.get("locks") or []
    out["device_count"] = len(locks)
    locked = []
    app_sm = []
    app_mem = []
    cur_sm = []
    cur_mem = []
    for item in locks:
        lock = (item or {}).get("lock") or {}
        clocks = (item or {}).get("clocks") or {}
        locked.append(bool(lock.get("locked")))
        if "applications_sm_mhz" in clocks:
            app_sm.append(clocks.get("applications_sm_mhz"))
        if "applications_mem_mhz" in clocks:
            app_mem.append(clocks.get("applications_mem_mhz"))
        if "current_sm_mhz" in clocks:
            cur_sm.append(clocks.get("current_sm_mhz"))
        if "current_mem_mhz" in clocks:
            cur_mem.append(clocks.get("current_mem_mhz"))
    out["all_devices_locked"] = len(locked) > 0 and all(locked)
    out["application_sm_mhz"] = sorted({int(v) for v in app_sm if v is not None})
    out["application_mem_mhz"] = sorted({int(v) for v in app_mem if v is not None})
    out["current_sm_mhz"] = sorted({int(v) for v in cur_sm if v is not None})
    out["current_mem_mhz"] = sorted({int(v) for v in cur_mem if v is not None})
    return out


result = load_json(raw_result_path)
leader_lock_payload = load_json(leader_lock_path)
worker_lock_payload = load_json(worker_lock_path)

leader_lock = lock_summary(leader_lock_payload)
leader_lock["path"] = leader_lock_path
worker_lock = lock_summary(worker_lock_payload)
worker_lock["path"] = worker_lock_path

status = "ok"
if int(leader_rc) != 0 or int(worker_rc) != 0:
    status = "failed"
if not result:
    status = "failed"
if not leader_lock["all_devices_locked"] or not worker_lock["all_devices_locked"]:
    status = "failed"
failure_reason = detect_failure_reason(leader_log) if status != "ok" else None

metrics = {
    "request_throughput": None,
    "output_throughput": None,
    "total_token_throughput": None,
    "mean_ttft_ms": None,
    "median_ttft_ms": None,
    "p99_ttft_ms": None,
    "mean_tpot_ms": None,
    "median_tpot_ms": None,
    "p99_tpot_ms": None,
    "completed": None,
    "failed": None,
}
if isinstance(result, dict):
    for k in metrics:
        metrics[k] = result.get(k)

payload = {
    "run_id": run_id,
    "status": status,
    "hosts": {"leader": leader_host, "worker": worker_host},
    "labels": {"leader": leader_label, "worker": worker_label},
    "model": model,
    "tp": int(tp),
    "isl": int(isl),
    "osl": int(osl),
    "concurrency": int(concurrency),
    "num_prompts": int(num_prompts),
    "ray_cluster_size": 2,
    "return_codes": {"leader": int(leader_rc), "worker": int(worker_rc)},
    "failure_reason": failure_reason,
    "image": {
        "requested": requested_image,
        "pinned": pinned_image,
        "digest": pinned_digest,
        "leader_image_id": leader_image_id,
        "worker_image_id": worker_image_id,
        "leader_repodigest": leader_image_repodigest,
        "worker_repodigest": worker_image_repodigest,
    },
    "clock_lock": {"leader": leader_lock, "worker": worker_lock},
    "metrics": metrics,
    "artifacts": {
        "raw_result_json": raw_result_path,
        "structured_result_json": struct_result_path,
        "leader_log": leader_log,
        "worker_log": worker_log,
        "server_log": server_log,
        "bench_log": bench_log,
    },
}

Path(summary_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

csv_fields = [
    "model",
    "tp",
    "isl",
    "osl",
    "concurrency",
    "num_prompts",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "gpu_util_mean_pct",
    "gpu_util_p95_pct",
    "mem_used_mean_mb",
    "mem_used_max_mb",
    "completed",
    "failed",
    "status",
    "leader_label",
    "worker_label",
    "leader_app_sm_mhz",
    "leader_app_mem_mhz",
    "worker_app_sm_mhz",
    "worker_app_mem_mhz",
    "failure_reason",
]

row = {
    "model": model,
    "tp": int(tp),
    "isl": int(isl),
    "osl": int(osl),
    "concurrency": int(concurrency),
    "num_prompts": int(num_prompts),
    "request_throughput": metrics["request_throughput"],
    "output_throughput": metrics["output_throughput"],
    "total_token_throughput": metrics["total_token_throughput"],
    "mean_ttft_ms": metrics["mean_ttft_ms"],
    "median_ttft_ms": metrics["median_ttft_ms"],
    "p99_ttft_ms": metrics["p99_ttft_ms"],
    "mean_tpot_ms": metrics["mean_tpot_ms"],
    "median_tpot_ms": metrics["median_tpot_ms"],
    "p99_tpot_ms": metrics["p99_tpot_ms"],
    "gpu_util_mean_pct": "",
    "gpu_util_p95_pct": "",
    "mem_used_mean_mb": "",
    "mem_used_max_mb": "",
    "completed": metrics["completed"],
    "failed": metrics["failed"],
    "status": status,
    "leader_label": leader_label,
    "worker_label": worker_label,
    "leader_app_sm_mhz": "|".join(map(str, leader_lock["application_sm_mhz"])),
    "leader_app_mem_mhz": "|".join(map(str, leader_lock["application_mem_mhz"])),
    "worker_app_sm_mhz": "|".join(map(str, worker_lock["application_sm_mhz"])),
    "worker_app_mem_mhz": "|".join(map(str, worker_lock["application_mem_mhz"])),
    "failure_reason": failure_reason or "",
}

csv_path = Path(summary_csv)
write_header = not csv_path.exists()
with csv_path.open("a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    if write_header:
        writer.writeheader()
    writer.writerow(row)

with Path(summary_jsonl).open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")

print(summary_json)
print(summary_csv)
print(summary_jsonl)
PY

FINAL_RC=0
if [[ "$LEADER_RC" -ne 0 || "$WORKER_RC" -ne 0 ]]; then
  FINAL_RC=1
fi
if [[ "$FINAL_RC" -eq 0 && ! -f "$RESULT_JSON_RAW" ]]; then
  FINAL_RC=1
fi
if [[ ! -s "$LEADER_LOCK_META" || ! -s "$WORKER_LOCK_META" ]]; then
  FINAL_RC=1
fi

if [[ "$FINAL_RC" -ne 0 ]]; then
  echo "ERROR: multinode vLLM benchmark did not complete cleanly." >&2
  echo "  leader log:  $LEADER_LOG" >&2
  echo "  worker log:  $WORKER_LOG" >&2
  echo "  summary:     $SUMMARY_JSON" >&2
  exit "$FINAL_RC"
fi

echo "Wrote ${SUMMARY_JSON}"
echo "Wrote ${SUMMARY_CSV}"
echo "Wrote ${SUMMARY_JSONL}"
