#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_cluster_health_suite_repeats.sh [options]

Runs the cluster health suite repeatedly to quantify variance.

Modes:
  base      run the base suite only
  extended  run the base suite with --extended only
  both      run base then extended for each repetition (default)

Options:
  --hosts <h1,h2>        Comma-separated host list (required)
  --repeats <n>          Number of repetitions (default: 3)
  --mode <base|extended|both>
  --prefix <id>          Run id prefix (default: YYYY-MM-DD_HHMMSS_cluster_health_suite_variance)
  --ssh-key <path>       SSH key for remote launch (default: $SSH_KEY)
  --oob-if <iface>       OOB interface (passed through)
  --nccl-ib-hca <list>   NCCL IB HCA allowlist (passed through)
  --gpus-per-node <n>    GPUs per node (passed through)
  -- <args...>           Extra args passed through to run_cluster_health_suite.sh
                         (example: --skip-iperf3 --skip-ib --skip-torchdist)

Examples:
  scripts/run_cluster_health_suite_repeats.sh --hosts node1,node2
  scripts/run_cluster_health_suite_repeats.sh --hosts node1,node2 --repeats 3 --mode both
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${ROOT_DIR}/scripts/run_cluster_health_suite.sh"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"

HOSTS=""
REPEATS=3
MODE="both"
PREFIX="$(date +%Y-%m-%d_%H%M%S)_cluster_health_suite_variance"
SSH_KEY="${SSH_KEY:-}"
OOB_IF="${OOB_IF:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
GPUS_PER_NODE=""
SUITE_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hosts) HOSTS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --oob-if) OOB_IF="$2"; shift 2 ;;
    --nccl-ib-hca) NCCL_IB_HCA="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --) shift; SUITE_EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

case "$MODE" in
  base|extended|both) ;;
  *) echo "ERROR: invalid --mode: $MODE" >&2; usage >&2; exit 2 ;;
esac

resolve_cluster_artifact_dirs "$ROOT_DIR" "$PREFIX"
OUT_STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
OUT_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}"
mkdir -p "$OUT_RAW_DIR" "$OUT_STRUCT_DIR"
LOG="${OUT_RAW_DIR}/${PREFIX}_repeats.log"

suite_args=(--hosts "$HOSTS")
if [[ -n "$SSH_KEY" ]]; then suite_args+=(--ssh-key "$SSH_KEY"); fi
if [[ -n "$OOB_IF" ]]; then suite_args+=(--oob-if "$OOB_IF"); fi
if [[ -n "$NCCL_IB_HCA" ]]; then suite_args+=(--nccl-ib-hca "$NCCL_IB_HCA"); fi
if [[ -n "$GPUS_PER_NODE" ]]; then suite_args+=(--gpus-per-node "$GPUS_PER_NODE"); fi

ts() { date -Is; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "PREFIX=${PREFIX} MODE=${MODE} REPEATS=${REPEATS} HOSTS=${HOSTS}"
log "suite_args=${suite_args[*]}"
if [[ "${#SUITE_EXTRA_ARGS[@]}" -gt 0 ]]; then
  log "suite_extra_args=${SUITE_EXTRA_ARGS[*]}"
fi

run_one() {
  local run_id="$1"
  shift
  log "== START run_id=${run_id} =="
  set +e
  "$SUITE" --run-id "$run_id" "${suite_args[@]}" "${SUITE_EXTRA_ARGS[@]}" "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  log "== END run_id=${run_id} rc=${rc} =="
  return 0
}

for ((i=1; i<=REPEATS; i++)); do
  if [[ "$MODE" == "base" || "$MODE" == "both" ]]; then
    run_one "${PREFIX}_r${i}_base"
  fi
  if [[ "$MODE" == "extended" || "$MODE" == "both" ]]; then
    run_one "${PREFIX}_r${i}_extended" --extended
  fi
done

log "Done. Repeat log: ${LOG}"
log "Summaries:"
ls -1 "${OUT_STRUCT_DIR}/${PREFIX}"*_cluster_health_suite_summary.json 2>/dev/null | tee -a "$LOG" || true
