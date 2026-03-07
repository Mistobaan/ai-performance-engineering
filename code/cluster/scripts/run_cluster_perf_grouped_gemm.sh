#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Run grouped GEMM benchmark (DeepGEMM FP8xFP4 path + torch baselines)
and write a structured log + summary + plot under runs/<run_id>/.

This validates the DeepGEMM grouped-GEMM FP8xFP4 path on GB200/SM100.

Usage:
  scripts/run_cluster_perf_grouped_gemm.sh \
    --run-id 2026-02-08_deepgemm_grouped_gemm \
    --label node1

Options:
  --run-id <id>       RUN_ID prefix for outputs (default: YYYY-MM-DD_grouped_gemm).
  --label <label>     Label used in output filenames (default: hostname).
  --preset <name>     Preset passed to grouped_gemm_bench.py (default: all).
  --warmup <n>        Warmup iterations (default: 5).
  --iters <n>         Benchmark iterations (default: 30).
  --runtime <mode>    host|container (default: host).
  --stack-profile <name>
                      Stack profile: old_container|orig_parity_container|new_container|host_only
                      (default: runtime-specific from configs/cluster_perf_stack_profiles.json).
  --image <image>     Container image for runtime=container
                      (default: stack-profile image_ref or $CONTAINER_IMAGE).
  --require-deepgemm  Fail if grouped GEMM summary reports DeepGEMM unsupported
                      or no DeepGEMM datapoints (default: off).
  --allow-deepgemm-unsupported
                      Disable --require-deepgemm after it was set earlier.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh" ]]; then
  echo "ERROR: missing stack profile helper: ${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh" >&2
  exit 1
fi
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
# shellcheck source=scripts/cluster_perf_stack_profiles.sh
source "${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh"

RUN_ID="$(date +%Y-%m-%d)_grouped_gemm"
LABEL="$(hostname)"
PRESET="all"
WARMUP="5"
ITERS="30"
RUNTIME="host"
STACK_PROFILE=""
IMAGE="${CONTAINER_IMAGE:-}"
REQUIRE_DEEPGEMM=0

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --label) LABEL="${2:-}"; shift 2 ;;
    --preset) PRESET="${2:-}"; shift 2 ;;
    --warmup) WARMUP="${2:-}"; shift 2 ;;
    --iters) ITERS="${2:-}"; shift 2 ;;
    --runtime) RUNTIME="${2:-}"; shift 2 ;;
    --stack-profile) STACK_PROFILE="${2:-}"; shift 2 ;;
    --image) IMAGE="${2:-}"; shift 2 ;;
    --require-deepgemm) REQUIRE_DEEPGEMM=1; shift ;;
    --allow-deepgemm-unsupported) REQUIRE_DEEPGEMM=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$RUNTIME" != "host" && "$RUNTIME" != "container" ]]; then
  echo "ERROR: --runtime must be host or container (got: ${RUNTIME})" >&2
  exit 1
fi

if [[ -z "$STACK_PROFILE" ]]; then
  STACK_PROFILE="$(cluster_perf_default_profile_for_runtime "$ROOT_DIR" "$RUNTIME")"
fi
if ! cluster_perf_profile_exists "$ROOT_DIR" "$STACK_PROFILE"; then
  echo "ERROR: unknown --stack-profile: ${STACK_PROFILE}" >&2
  exit 1
fi
if ! cluster_perf_profile_runtime_allowed "$ROOT_DIR" "$STACK_PROFILE" "$RUNTIME"; then
  echo "ERROR: --stack-profile ${STACK_PROFILE} does not allow runtime=${RUNTIME}" >&2
  exit 1
fi

if [[ "$RUNTIME" == "container" && -z "$IMAGE" ]]; then
  IMAGE="$(cluster_perf_profile_image_ref "$ROOT_DIR" "$STACK_PROFILE")"
fi
if [[ "$RUNTIME" == "container" && -z "$IMAGE" ]]; then
  echo "ERROR: no container image resolved for runtime=container/profile=${STACK_PROFILE}" >&2
  exit 1
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
OUT_FIG_DIR="${CLUSTER_FIGURES_DIR_EFFECTIVE}"
mkdir -p "${OUT_STRUCT_DIR}" "${OUT_FIG_DIR}"

OUT_LOG="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_cluster_perf_grouped_gemm.txt"
OUT_JSON="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_cluster_perf_grouped_gemm_summary.json"
OUT_PNG="${OUT_FIG_DIR}/${RUN_ID}_${LABEL}_cluster_perf_grouped_gemm_tflops.png"
LOCK_META="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_cluster_perf_grouped_gemm_clock_lock.json"
PREFLIGHT_STACK_META="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_cluster_perf_grouped_gemm_preflight_stack.json"
PREFLIGHT_CLOCK_META="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_cluster_perf_grouped_gemm_preflight_clock_lock.json"
BENCH_SCRIPT_REL="scripts/benchmarks/grouped_gemm_bench.py"
BENCH_SCRIPT="${ROOT_DIR}/${BENCH_SCRIPT_REL}"
MATH_ALLOW_TF32="$(cluster_perf_profile_math_allow_tf32 "$ROOT_DIR" "$STACK_PROFILE")"
MATH_FP32_MATMUL_PRECISION="$(cluster_perf_profile_math_precision "$ROOT_DIR" "$STACK_PROFILE")"
HOST_CUDA_HOME=""

resolve_host_cuda_home() {
  local resolved="${CUDA_HOME:-}"
  if [[ -z "$resolved" ]]; then
    if [[ -d /usr/local/cuda ]]; then
      resolved="/usr/local/cuda"
    elif command -v nvcc >/dev/null 2>&1; then
      local nvcc_path
      nvcc_path="$(command -v nvcc)"
      resolved="$(cd "$(dirname "$nvcc_path")/.." && pwd -P)"
    fi
  fi
  if [[ -z "$resolved" || ! -f "${resolved}/include/cuda_runtime.h" ]]; then
    echo "ERROR: CUDA toolkit headers not found for host runtime (resolved CUDA_HOME=${resolved:-<empty>})." >&2
    exit 1
  fi
  printf '%s\n' "$resolved"
}

echo "== Cluster Perf Grouped GEMM =="
echo "RUN_ID=${RUN_ID}"
echo "LABEL=${LABEL}"
echo "RUNTIME=${RUNTIME}"
echo "STACK_PROFILE=${STACK_PROFILE}"
if [[ "$RUNTIME" == "container" ]]; then
  echo "IMAGE=${IMAGE}"
fi
echo "MATH_POLICY=allow_tf32:${MATH_ALLOW_TF32},float32_matmul_precision:${MATH_FP32_MATMUL_PRECISION}"
echo "PRESET=${PRESET}"
echo "WARMUP=${WARMUP}"
echo "ITERS=${ITERS}"
echo "BENCH_SCRIPT=${BENCH_SCRIPT_REL}"
echo

if [[ ! -f "$BENCH_SCRIPT" ]]; then
  echo "ERROR: grouped benchmark script not found: ${BENCH_SCRIPT}" >&2
  exit 1
fi
if [[ ! -x "${ROOT_DIR}/env/venv/bin/python" ]]; then
  echo "ERROR: venv python not found: ${ROOT_DIR}/env/venv/bin/python" >&2
  exit 1
fi
if [[ "$RUNTIME" == "host" ]]; then
  if [[ "$STACK_PROFILE" == "host_only" ]]; then
    HOST_RUNTIME_ENV_FILE="${ROOT_DIR}/env/venv/orig_parity_runtime_env.sh"
    if [[ ! -f "$HOST_RUNTIME_ENV_FILE" ]]; then
      echo "ERROR: missing host runtime env file for host_only parity: ${HOST_RUNTIME_ENV_FILE}" >&2
      echo "Run scripts/bootstrap_cluster_nodes.sh (or run_fp4_checks_all_nodes.sh with bootstrap enabled) first." >&2
      exit 1
    fi
    # shellcheck disable=SC1090
    source "$HOST_RUNTIME_ENV_FILE"
  fi
  HOST_CUDA_HOME="$(resolve_host_cuda_home)"
fi

preflight_args=(
  --runtime "${RUNTIME}"
  --stack-profile "${STACK_PROFILE}"
  --profiles-json "${ROOT_DIR}/configs/cluster_perf_stack_profiles.json"
  --out-json "${PREFLIGHT_STACK_META}"
  --host-python "${ROOT_DIR}/env/venv/bin/python"
)
if [[ "$RUNTIME" == "container" ]]; then
  preflight_args+=(--image "${IMAGE}")
fi
"${ROOT_DIR}/env/venv/bin/python" "${ROOT_DIR}/scripts/preflight_cluster_perf_runtime.py" "${preflight_args[@]}"

# Fail fast before long benchmark runs if clocks cannot be locked.
"${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
  --lock-meta-out "${PREFLIGHT_CLOCK_META}" \
  -- bash -lc "set -euo pipefail; nvidia-smi -L >/dev/null"

if [[ "$RUNTIME" == "host" ]]; then
  echo "HOST_CUDA_HOME=${HOST_CUDA_HOME}"
  "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --lock-meta-out "$LOCK_META" \
    -- bash -lc "set -euo pipefail; CUDA_HOME=\"${HOST_CUDA_HOME}\" CUDACXX=\"${HOST_CUDA_HOME}/bin/nvcc\" CLUSTER_PERF_ALLOW_TF32=\"${MATH_ALLOW_TF32}\" CLUSTER_PERF_FLOAT32_MATMUL_PRECISION=\"${MATH_FP32_MATMUL_PRECISION}\" \"${ROOT_DIR}/env/venv/bin/python\" -u \"${BENCH_SCRIPT}\" --preset \"${PRESET}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" 2>&1 | tee \"${OUT_LOG}\""
else
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found for --runtime container." >&2
    exit 1
  fi
  "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --lock-meta-out "$LOCK_META" \
    -- bash -lc "set -euo pipefail; docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e CLUSTER_PERF_ALLOW_TF32=\"${MATH_ALLOW_TF32}\" -e CLUSTER_PERF_FLOAT32_MATMUL_PRECISION=\"${MATH_FP32_MATMUL_PRECISION}\" -v \"${ROOT_DIR}:/workspace\" -w /workspace \"${IMAGE}\" python -u \"${BENCH_SCRIPT_REL}\" --preset \"${PRESET}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" 2>&1 | tee \"${OUT_LOG}\""
fi

"${ROOT_DIR}/env/venv/bin/python" \
  "${ROOT_DIR}/analysis/summarize_grouped_gemm_torch_fp16_vs_fp8.py" \
  --in-log "$OUT_LOG" \
  --out-json "$OUT_JSON"

if [[ "$REQUIRE_DEEPGEMM" -eq 1 ]]; then
  "${ROOT_DIR}/env/venv/bin/python" - "$OUT_JSON" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)

counts = payload.get("counts") or {}
deepgemm = payload.get("deepgemm") or {}
unsupported_reason = str(deepgemm.get("unsupported_reason") or "").strip()
datapoints_ok = int(counts.get("deepgemm_datapoints_ok") or 0)

if unsupported_reason:
    raise SystemExit(f"DeepGEMM grouped GEMM unsupported: {unsupported_reason}")
if datapoints_ok < 1:
    raise SystemExit("DeepGEMM grouped GEMM produced no valid datapoints")

print(f"DeepGEMM grouped GEMM validation passed: deepgemm_datapoints_ok={datapoints_ok}")
PY
fi

"${ROOT_DIR}/env/venv/bin/python" \
  "${ROOT_DIR}/analysis/plot_grouped_gemm_torch_fp16_vs_fp8.py" \
  --summary-json "$OUT_JSON" \
  --title "Grouped GEMM: Torch FP16/FP8 vs DeepGEMM FP8xFP4" \
  --out "$OUT_PNG"

echo
echo "Outputs:"
echo "  - $OUT_LOG"
echo "  - $OUT_JSON"
echo "  - $OUT_PNG"
echo "  - $LOCK_META"
echo "  - $PREFLIGHT_STACK_META"
echo "  - $PREFLIGHT_CLOCK_META"
