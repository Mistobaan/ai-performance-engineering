#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_fp4_checks_all_nodes.sh --hosts <h1,h2,...> [options]

Runs FP4 checks:
  1) Cluster Perf grouped GEMM benchmark (DeepGEMM path) per host
  2) DeepGEMM FP8xFP4 smoke/perf probe in paired rounds per host
  3) Cross-host smoke skew guard on median TFLOPS
  4) Balanced FP4 attestation (semantic source checks + provenance capture
     + cross-host consistency report)

Outputs are written under:
  results/raw/
  results/structured/
  docs/figures/

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo root)
  --runtime <mode>       host|container (default: host)
  --stack-profile <name> Stack profile: old_container|orig_parity_container|new_container|host_only
                         (default: runtime-specific from configs/cluster_perf_stack_profiles.json)
  --image <image>        Container image for runtime=container
                         (default: stack-profile image_ref or $CONTAINER_IMAGE)
  --preset <name>        Grouped-GEMM preset (default: auto; GB-family hosts use all)
  --warmup <n>           Grouped-GEMM warmup (default: 5)
  --iters <n>            Grouped-GEMM measured iterations (default: 30)
  --skip-smoke           Skip the FP4 smoke/perf probe step
  --smoke-m <int>        Smoke shape M (default: 4096)
  --smoke-n <int>        Smoke shape N (default: 4096)
  --smoke-k <int>        Smoke shape K (default: 4096)
  --smoke-warmup <n>     Smoke warmup (default: 10)
  --smoke-iters <n>      Smoke measured iterations (default: 30)
  --smoke-rounds <n>     Paired smoke rounds per host for skew guard (default: 3)
  --smoke-skew-threshold-pct <pct>
                        Fail when max pairwise median smoke gap exceeds this percent (default: 5)

Bootstrap (recommended for reproducibility; default: enabled):
  --bootstrap-nodes                Run per-node bootstrap before FP4 checks
  --skip-bootstrap-nodes           Skip bootstrap
  --bootstrap-install-system-packages   Install missing system deps (default: on)
  --bootstrap-skip-system-packages      Skip system package installation
  --bootstrap-sync-code            Sync scripts/analysis/env requirements to remotes (default: on)
  --bootstrap-skip-sync-code       Skip code sync
  --bootstrap-install-python-deps  Ensure env/venv + python deps (default: on)
  --bootstrap-skip-python-deps     Skip python dependency install
  --bootstrap-host-parity-image <ref>  Source image for host-only parity install
                                       (default: cfregly/cluster_perf_orig_parity:latest)
  --bootstrap-torch-index-url <url>  Legacy fallback torch index (default: https://pypi.ngc.nvidia.com)
  --bootstrap-torch-version <ver>    Expected torch version after bootstrap parity install
                                     (default: 2.10.0a0+a36e1d39eb.nv26.01.42222806)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh" ]]; then
  echo "ERROR: missing stack profile helper: ${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh" >&2
  exit 1
fi
# shellcheck source=scripts/cluster_perf_stack_profiles.sh
source "${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh"

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
HOSTS=""
LABELS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"
REMOTE_ROOT="${REMOTE_ROOT:-$ROOT_DIR}"
RUNTIME="host"
STACK_PROFILE=""
IMAGE="${CONTAINER_IMAGE:-}"
PRESET="auto"
WARMUP="5"
ITERS="30"
SKIP_SMOKE=0
SMOKE_M="4096"
SMOKE_N="4096"
SMOKE_K="4096"
SMOKE_WARMUP="10"
SMOKE_ITERS="30"
SMOKE_ROUNDS="3"
SMOKE_SKEW_THRESHOLD_PCT="5"
BOOTSTRAP_NODES=1
BOOTSTRAP_INSTALL_SYSTEM_PACKAGES=1
BOOTSTRAP_SYNC_CODE=1
BOOTSTRAP_INSTALL_PYTHON_DEPS=1
BOOTSTRAP_HOST_PARITY_IMAGE="${BOOTSTRAP_HOST_PARITY_IMAGE:-cfregly/cluster_perf_orig_parity:latest}"
BOOTSTRAP_TORCH_INDEX_URL="https://pypi.ngc.nvidia.com"
BOOTSTRAP_TORCH_VERSION="2.10.0a0+a36e1d39eb.nv26.01.42222806"
ATTESTATION_MODE="balanced"
ATTESTATION_PROFILE="gb200_grouped_gemm_balanced_v1"
ATTESTATION_TARGET_REL="scripts/benchmarks/grouped_gemm_bench.py"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --hosts) HOSTS="${2:-}"; shift 2 ;;
    --labels) LABELS="${2:-}"; shift 2 ;;
    --ssh-user) SSH_USER="${2:-}"; shift 2 ;;
    --ssh-key) SSH_KEY="${2:-}"; shift 2 ;;
    --remote-root) REMOTE_ROOT="${2:-}"; shift 2 ;;
    --runtime) RUNTIME="${2:-}"; shift 2 ;;
    --stack-profile) STACK_PROFILE="${2:-}"; shift 2 ;;
    --image) IMAGE="${2:-}"; shift 2 ;;
    --preset) PRESET="${2:-}"; shift 2 ;;
    --warmup) WARMUP="${2:-}"; shift 2 ;;
    --iters) ITERS="${2:-}"; shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --smoke-m) SMOKE_M="${2:-}"; shift 2 ;;
    --smoke-n) SMOKE_N="${2:-}"; shift 2 ;;
    --smoke-k) SMOKE_K="${2:-}"; shift 2 ;;
    --smoke-warmup) SMOKE_WARMUP="${2:-}"; shift 2 ;;
    --smoke-iters) SMOKE_ITERS="${2:-}"; shift 2 ;;
    --smoke-rounds) SMOKE_ROUNDS="${2:-}"; shift 2 ;;
    --smoke-skew-threshold-pct) SMOKE_SKEW_THRESHOLD_PCT="${2:-}"; shift 2 ;;
    --bootstrap-nodes) BOOTSTRAP_NODES=1; shift ;;
    --skip-bootstrap-nodes) BOOTSTRAP_NODES=0; shift ;;
    --bootstrap-install-system-packages) BOOTSTRAP_INSTALL_SYSTEM_PACKAGES=1; shift ;;
    --bootstrap-skip-system-packages) BOOTSTRAP_INSTALL_SYSTEM_PACKAGES=0; shift ;;
    --bootstrap-sync-code) BOOTSTRAP_SYNC_CODE=1; shift ;;
    --bootstrap-skip-sync-code) BOOTSTRAP_SYNC_CODE=0; shift ;;
    --bootstrap-install-python-deps) BOOTSTRAP_INSTALL_PYTHON_DEPS=1; shift ;;
    --bootstrap-skip-python-deps) BOOTSTRAP_INSTALL_PYTHON_DEPS=0; shift ;;
    --bootstrap-host-parity-image) BOOTSTRAP_HOST_PARITY_IMAGE="${2:-}"; shift 2 ;;
    --bootstrap-torch-index-url) BOOTSTRAP_TORCH_INDEX_URL="${2:-}"; shift 2 ;;
    --bootstrap-torch-version) BOOTSTRAP_TORCH_VERSION="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$SMOKE_ROUNDS" =~ ^[0-9]+$ ]] || [[ "$SMOKE_ROUNDS" -lt 1 ]]; then
  echo "ERROR: --smoke-rounds must be an integer >= 1 (got: ${SMOKE_ROUNDS})" >&2
  exit 2
fi
if ! [[ "$SMOKE_SKEW_THRESHOLD_PCT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: --smoke-skew-threshold-pct must be a non-negative number (got: ${SMOKE_SKEW_THRESHOLD_PCT})" >&2
  exit 2
fi

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi
if [[ "$RUNTIME" != "host" && "$RUNTIME" != "container" ]]; then
  echo "ERROR: --runtime must be host or container (got: ${RUNTIME})" >&2
  usage >&2
  exit 2
fi
if [[ -z "$STACK_PROFILE" ]]; then
  STACK_PROFILE="$(cluster_perf_default_profile_for_runtime "$ROOT_DIR" "$RUNTIME")"
fi
if ! cluster_perf_profile_exists "$ROOT_DIR" "$STACK_PROFILE"; then
  echo "ERROR: unknown --stack-profile: ${STACK_PROFILE}" >&2
  exit 2
fi
if ! cluster_perf_profile_runtime_allowed "$ROOT_DIR" "$STACK_PROFILE" "$RUNTIME"; then
  echo "ERROR: --stack-profile ${STACK_PROFILE} does not allow runtime=${RUNTIME}" >&2
  exit 2
fi
if [[ "$RUNTIME" == "container" && -z "$IMAGE" ]]; then
  IMAGE="$(cluster_perf_profile_image_ref "$ROOT_DIR" "$STACK_PROFILE")"
fi
if [[ "$RUNTIME" == "container" && -z "$IMAGE" ]]; then
  echo "ERROR: no container image resolved for runtime=container/profile=${STACK_PROFILE}" >&2
  usage >&2
  exit 2
fi

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
IFS=',' read -r -a LABEL_ARR <<<"$LABELS"
if [[ -n "$LABELS" && "${#LABEL_ARR[@]}" -ne "${#HOST_ARR[@]}" ]]; then
  echo "ERROR: --labels count must match --hosts count" >&2
  exit 2
fi

if [[ "$BOOTSTRAP_NODES" -eq 1 ]]; then
  bootstrap_args=(
    --run-id "${RUN_ID}"
    --hosts "${HOSTS}"
    --ssh-user "${SSH_USER}"
    --remote-root "${REMOTE_ROOT}"
    --host-parity-image "${BOOTSTRAP_HOST_PARITY_IMAGE}"
    --torch-index-url "${BOOTSTRAP_TORCH_INDEX_URL}"
    --torch-version "${BOOTSTRAP_TORCH_VERSION}"
  )
  if [[ -n "$LABELS" ]]; then
    bootstrap_args+=(--labels "${LABELS}")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    bootstrap_args+=(--ssh-key "${SSH_KEY}")
  fi
  if [[ "$BOOTSTRAP_INSTALL_SYSTEM_PACKAGES" -eq 0 ]]; then
    bootstrap_args+=(--skip-system-packages)
  fi
  if [[ "$BOOTSTRAP_SYNC_CODE" -eq 0 ]]; then
    bootstrap_args+=(--skip-sync-code)
  fi
  if [[ "$BOOTSTRAP_INSTALL_PYTHON_DEPS" -eq 0 ]]; then
    bootstrap_args+=(--skip-python-deps)
  fi

  echo "Running bootstrap across hosts..."
  "${ROOT_DIR}/scripts/bootstrap_cluster_nodes.sh" "${bootstrap_args[@]}"
fi

sanitize_label() {
  local raw="$1"
  raw="${raw//./_}"
  raw="${raw//:/_}"
  echo "$raw"
}

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=5
  -o ConnectionAttempts=3
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
  -o IdentitiesOnly=yes
  -o IdentityAgent=none
)
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

run_remote() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "$@"
}

run_host_cmd() {
  local host="$1"
  local cmd="$2"
  if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
    bash -lc "$cmd"
  else
    run_remote "$host" "bash -lc $(printf '%q' "$cmd")"
  fi
}

trim_ws() {
  local s="${1:-}"
  s="${s//$'\r'/}"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

copy_attestation_target_snapshot() {
  local host="$1"
  local label="$2"
  local local_target_abs="${ROOT_DIR}/${ATTESTATION_TARGET_REL}"
  local remote_target_abs="${REMOTE_ROOT}/${ATTESTATION_TARGET_REL}"
  local rel_path="results/raw/${RUN_ID}_${label}_grouped_gemm_bench.snapshot.py"
  local abs_path="${ROOT_DIR}/${rel_path}"

  mkdir -p "$(dirname "$abs_path")"
  if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
    cp "$local_target_abs" "$abs_path"
  else
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${remote_target_abs}" "$abs_path" >/dev/null
  fi

  echo "$rel_path"
}

build_semantic_attestation() {
  local snapshot_abs="$1"
  local profile="$2"

  python3 - "$snapshot_abs" "$profile" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

snapshot_path = Path(sys.argv[1])
profile = sys.argv[2]
text = snapshot_path.read_text(encoding="utf-8")

required_patterns = {
    "global_unsupported_reason": r"DEEPGEMM_UNSUPPORTED_REASON\s*:\s*Optional\[str\]\s*=\s*None",
    "ue8m0_arch_switch": r"use_ue8m0\s*=\s*arch_major\s*>=\s*10",
    "disable_cast_switch": r"disable_ue8m0_cast\s*=\s*not\s*use_ue8m0",
    "per_token_use_ue8m0": r"per_token_cast_to_fp8\(\s*a_bf16\s*,\s*use_ue8m0\s*=\s*use_ue8m0\s*\)",
    "per_block_use_ue8m0": r"per_block_cast_to_fp8\(\s*b_bf16\[i\]\s*,\s*use_ue8m0\s*=\s*use_ue8m0\s*\)",
    "kernel_fallback_getattr": r"getattr\(\s*deep_gemm\s*,\s*\"m_grouped_fp8_gemm_nt_contiguous\"",
    "kernel_disable_cast_var": r"disable_ue8m0_cast\s*=\s*disable_ue8m0_cast",
    "unsupported_print": r"DeepGEMM unsupported:\s*\{e\}",
}

forbidden_patterns = {
    # Legacy callsite form; require this exact code-like shape to avoid comment-only false positives.
    "legacy_disable_true_literal": r"disable_ue8m0_cast\s*=\s*True\s*,",
}

def normalize_snippet(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())

required_matches = {}
missing_required = []
for marker, pattern in required_patterns.items():
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        required_matches[marker] = None
        missing_required.append(marker)
    else:
        required_matches[marker] = normalize_snippet(match.group(0))

forbidden_hits = {}
for marker, pattern in forbidden_patterns.items():
    matches = [normalize_snippet(m.group(0)) for m in re.finditer(pattern, text, flags=re.MULTILINE)]
    forbidden_hits[marker] = matches

status = "pass"
if missing_required:
    status = "fail"
if any(forbidden_hits.values()):
    status = "fail"

semantic_signature_input = {
    "profile": profile,
    "required_matches": required_matches,
    "forbidden_hit_markers": sorted(k for k, v in forbidden_hits.items() if v),
}
semantic_signature = hashlib.sha256(
    json.dumps(semantic_signature_input, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

payload = {
    "profile": profile,
    "status": status,
    "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    "semantic_signature": semantic_signature,
    "required_markers": required_matches,
    "missing_required_markers": missing_required,
    "forbidden_hits": forbidden_hits,
}

print(json.dumps(payload, sort_keys=True))
if status != "pass":
    raise SystemExit(1)
PY
}

write_platform_meta() {
  local out_path="$1"
  local host="$2"
  local label="$3"
  local requested_preset="$4"
  local selected_preset="$5"
  local gb_sku="$6"
  local runtime="$7"
  local stack_profile="$8"
  local image="$9"
  local gpu_names_b64="${10}"
  local semantic_json_b64="${11}"
  local attestation_target_rel="${12}"
  local attestation_target_snapshot_rel="${13}"
  local repo_git_commit="${14}"
  local repo_git_dirty="${15}"
  local image_id="${16}"
  local image_repo_digests_b64="${17}"
  local driver_version="${18}"
  local cuda_version="${19}"
  local torch_version="${20}"
  local deep_gemm_version="${21}"
  local grouped_summary_rel="${22}"
  local grouped_log_rel="${23}"
  local grouped_clock_rel="${24}"

  python3 - "$out_path" "$host" "$label" "$requested_preset" "$selected_preset" "$gb_sku" "$runtime" "$stack_profile" "$image" "$gpu_names_b64" "$semantic_json_b64" "$attestation_target_rel" "$attestation_target_snapshot_rel" "$repo_git_commit" "$repo_git_dirty" "$image_id" "$image_repo_digests_b64" "$driver_version" "$cuda_version" "$torch_version" "$deep_gemm_version" "$grouped_summary_rel" "$grouped_log_rel" "$grouped_clock_rel" <<'PY'
import base64
import json
import sys
import time
from pathlib import Path

(
    out_path,
    host,
    label,
    requested_preset,
    selected_preset,
    gb_sku,
    runtime,
    stack_profile,
    image,
    gpu_names_b64,
    semantic_json_b64,
    attestation_target_rel,
    attestation_target_snapshot_rel,
    repo_git_commit,
    repo_git_dirty,
    image_id,
    image_repo_digests_b64,
    driver_version,
    cuda_version,
    torch_version,
    deep_gemm_version,
    grouped_summary_rel,
    grouped_log_rel,
    grouped_clock_rel,
) = sys.argv[1:]
gpu_names = [line.strip() for line in base64.b64decode(gpu_names_b64).decode("utf-8").splitlines() if line.strip()]
semantic = json.loads(base64.b64decode(semantic_json_b64).decode("utf-8"))

repo_digests: list[str] = []
if image_repo_digests_b64:
    try:
        decoded = base64.b64decode(image_repo_digests_b64).decode("utf-8")
        parsed = json.loads(decoded)
        if isinstance(parsed, list):
            repo_digests = [str(x) for x in parsed]
    except Exception:
        repo_digests = []

payload = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "host": host,
    "label": label,
    "gpu_names": gpu_names,
    "gb_family_detected": bool(gb_sku),
    "gb_sku": gb_sku or None,
    "fp4": {
        "requested_preset": requested_preset,
        "selected_preset": selected_preset,
        "runtime": runtime,
        "stack_profile": stack_profile,
        "image": image if runtime == "container" else None,
        "attestation": {
            "mode": "balanced",
            "target_relative_path": attestation_target_rel,
            "snapshot_relative_path": attestation_target_snapshot_rel,
            "semantic": semantic,
        },
        "provenance": {
            "repo": {
                "path": ".",
                "git_commit": repo_git_commit or None,
                "git_dirty": None if repo_git_dirty == "unknown" else (repo_git_dirty == "true"),
            },
            "container": {
                "image_ref": image if runtime == "container" else None,
                "image_id": image_id or None,
                "repo_digests": repo_digests,
            },
            "python": {
                "torch_version": torch_version or None,
                "deep_gemm_version": deep_gemm_version or None,
            },
            "runtime": {
                "driver_version": driver_version or None,
                "cuda_version": cuda_version or None,
            },
            "note": "metadata capture only; no state mutation or locking",
        },
        "artifacts": {
            "grouped_summary": grouped_summary_rel,
            "grouped_log": grouped_log_rel,
            "grouped_clock_lock": grouped_clock_rel,
        },
    },
}

out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

write_attestation_consistency() {
  local root_dir="$1"
  local run_id="$2"
  local labels_csv="$3"
    local out_path="$4"

  python3 - "$root_dir" "$run_id" "$labels_csv" "$out_path" <<'PY'
import json
import sys
import time
from pathlib import Path

root_dir, run_id, labels_csv, out_path = sys.argv[1:]
labels = [x.strip() for x in labels_csv.split(",") if x.strip()]
structured = Path(root_dir) / "results" / "structured"

entries = []
missing_platform_files = []
for label in labels:
    path = structured / f"{run_id}_{label}_cluster_perf_fp4_platform.json"
    if not path.exists():
        missing_platform_files.append(str(path))
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    fp4 = payload.get("fp4") or {}
    att = fp4.get("attestation") or {}
    sem = att.get("semantic") or {}
    prov = fp4.get("provenance") or {}
    container = prov.get("container") or {}
    py = prov.get("python") or {}
    entries.append(
        {
            "label": label,
            "platform_file": str(path),
            "runtime": fp4.get("runtime"),
            "semantic_status": sem.get("status"),
            "semantic_signature": sem.get("semantic_signature"),
            "source_sha256": sem.get("source_sha256"),
            "image_id": container.get("image_id"),
            "repo_digests": container.get("repo_digests") or [],
            "torch_version": py.get("torch_version"),
            "deep_gemm_version": py.get("deep_gemm_version"),
        }
    )

status = "pass"
reasons = []
warnings = []

if missing_platform_files:
    status = "fail"
    reasons.append("missing_platform_files")

semantic_status_failures = [e["label"] for e in entries if e.get("semantic_status") != "pass"]
if semantic_status_failures:
    status = "fail"
    reasons.append("semantic_attestation_failed")

semantic_signatures = {e["semantic_signature"] for e in entries if e.get("semantic_signature")}
if len(semantic_signatures) > 1:
    status = "fail"
    reasons.append("semantic_signature_mismatch")

runtimes = {e.get("runtime") for e in entries if e.get("runtime")}
if len(runtimes) > 1:
    status = "fail"
    reasons.append("runtime_mode_mismatch")

runtime_mode = next(iter(runtimes)) if runtimes else None
image_ids = {e["image_id"] for e in entries if e.get("image_id")}
if runtime_mode == "container":
    if len(image_ids) > 1:
        status = "fail"
        reasons.append("image_id_mismatch")
    if len(image_ids) == 0:
        status = "fail"
        reasons.append("missing_image_id_provenance")

source_hashes = {e["source_sha256"] for e in entries if e.get("source_sha256")}
if len(source_hashes) > 1:
    warnings.append("source_sha256_differs_across_hosts")

deep_gemm_versions = {e["deep_gemm_version"] for e in entries if e.get("deep_gemm_version")}
if len(deep_gemm_versions) > 1:
    warnings.append("deep_gemm_version_differs_across_hosts")

payload = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "run_id": run_id,
    "mode": "balanced",
    "runtime_mode": runtime_mode,
    "labels": labels,
    "status": status,
    "reasons": reasons,
    "warnings": warnings,
    "missing_platform_files": missing_platform_files,
    "consistency": {
        "runtimes": sorted(x for x in runtimes if x),
        "semantic_signatures": sorted(x for x in semantic_signatures if x),
        "image_ids": sorted(x for x in image_ids if x),
        "source_sha256": sorted(x for x in source_hashes if x),
        "deep_gemm_versions": sorted(x for x in deep_gemm_versions if x),
    },
    "hosts": entries,
}

out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

print(f"attestation_consistency_json={out_path}")
print(f"status={status}")
if reasons:
    print("reasons=" + ",".join(reasons))
if warnings:
    print("warnings=" + ",".join(warnings))

if status != "pass":
    raise SystemExit(1)
PY
}

fetch_remote_artifact() {
  local host="$1"
  local rel_path="$2"
  local dst_dir
  dst_dir="${ROOT_DIR}/$(dirname "$rel_path")"
  mkdir -p "$dst_dir"
  if ! scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${REMOTE_ROOT}/${rel_path}" "${dst_dir}/" >/dev/null 2>&1; then
    echo "ERROR: failed to fetch ${rel_path} from ${host}" >&2
    return 1
  fi
  if [[ ! -f "${ROOT_DIR}/${rel_path}" ]]; then
    echo "ERROR: fetched artifact missing locally after copy: ${rel_path}" >&2
    return 1
  fi
}

verify_local_artifact() {
  local rel_path="$1"
  if [[ ! -f "${ROOT_DIR}/${rel_path}" ]]; then
    echo "ERROR: expected artifact not found: ${rel_path}" >&2
    return 1
  fi
}

write_smoke_skew_guard() {
  local root_dir="$1"
  local out_path="$2"
  local run_id="$3"
  local threshold_pct="$4"
  local smoke_rounds="$5"
  local labels_csv="$6"

  python3 - "$root_dir" "$out_path" "$run_id" "$threshold_pct" "$smoke_rounds" "$labels_csv" <<'PY'
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

root_dir, out_path, run_id, threshold_pct_raw, smoke_rounds_raw, labels_csv = sys.argv[1:]
threshold_pct = float(threshold_pct_raw)
smoke_rounds = int(smoke_rounds_raw)
labels = [x.strip() for x in labels_csv.split(",") if x.strip()]

root = Path(root_dir)
structured = root / "results" / "structured"

def load_tflops(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["results"]["deepgemm_fp8_fp4"]["avg_tflops"])

per_host = {}
for label in labels:
    rounds = []
    for round_id in range(1, smoke_rounds + 1):
        smoke_json = structured / f"{run_id}_r{round_id}_{label}_cluster_perf_fp4_smoke.json"
        rounds.append(
            {
                "round": round_id,
                "smoke_json": str(smoke_json),
                "deepgemm_avg_tflops": load_tflops(smoke_json),
            }
        )
    values = [entry["deepgemm_avg_tflops"] for entry in rounds]
    per_host[label] = {
        "rounds": rounds,
        "deepgemm_avg_tflops": {
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        },
    }

pairwise = []
max_gap_pct = 0.0
max_gap_pair = None
for a, b in itertools.combinations(labels, 2):
    med_a = per_host[a]["deepgemm_avg_tflops"]["median"]
    med_b = per_host[b]["deepgemm_avg_tflops"]["median"]
    denom = max(med_a, med_b)
    gap_pct = 0.0 if denom == 0 else (abs(med_a - med_b) / denom) * 100.0
    row = {
        "pair": [a, b],
        "median_tflops": {a: med_a, b: med_b},
        "median_gap_pct": gap_pct,
    }
    pairwise.append(row)
    if gap_pct > max_gap_pct:
        max_gap_pct = gap_pct
        max_gap_pair = [a, b]

status = "pass"
reason = "max_pairwise_median_gap_within_threshold"
if len(labels) > 1 and max_gap_pct > threshold_pct:
    status = "fail"
    reason = "max_pairwise_median_gap_exceeds_threshold"
elif len(labels) <= 1:
    reason = "single_host_no_pairwise_comparison"

payload = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "run_id": run_id,
    "smoke_rounds": smoke_rounds,
    "smoke_skew_threshold_pct": threshold_pct,
    "labels": labels,
    "status": status,
    "reason": reason,
    "max_pairwise_median_gap_pct": max_gap_pct,
    "max_pairwise_median_gap_pair": max_gap_pair,
    "pairwise_median_gaps": pairwise,
    "per_host": per_host,
}

out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

print(f"status={status}")
print(f"max_pairwise_median_gap_pct={max_gap_pct:.6f}")
if max_gap_pair:
    print(f"max_pairwise_pair={','.join(max_gap_pair)}")
print(f"guard_json={out_path}")

if status == "fail":
    raise SystemExit(1)
PY
}

trim_csv() {
  local csv="$1"
  local out=()
  local part
  IFS=',' read -r -a _parts <<<"$csv"
  for part in "${_parts[@]}"; do
    part="$(echo "$part" | xargs)"
    if [[ -n "$part" ]]; then
      out+=("$part")
    fi
  done
  local joined=""
  local item
  for item in "${out[@]}"; do
    if [[ -n "$joined" ]]; then
      joined+=","
    fi
    joined+="$item"
  done
  printf '%s' "$joined"
}

fetch_and_verify_if_remote() {
  local host="$1"
  local rel_path="$2"
  if [[ "$host" != "localhost" && "$host" != "$(hostname)" ]]; then
    fetch_remote_artifact "$host" "$rel_path"
  fi
  verify_local_artifact "$rel_path"
}

TRIMMED_LABELS="$LABELS"
if [[ -n "$LABELS" ]]; then
  TRIMMED_LABELS="$(trim_csv "$LABELS")"
fi

echo "FP4 smoke guard config: rounds=${SMOKE_ROUNDS} max_pairwise_median_gap_pct=${SMOKE_SKEW_THRESHOLD_PCT}"
echo "FP4 attestation mode: ${ATTESTATION_MODE} (${ATTESTATION_PROFILE})"

declare -a ATTESTATION_LABELS=()

for idx in "${!HOST_ARR[@]}"; do
  host="$(echo "${HOST_ARR[$idx]}" | xargs)"
  [[ -n "$host" ]] || continue

  label=""
  if [[ -n "$LABELS" ]]; then
    label="$(echo "${LABEL_ARR[$idx]}" | xargs)"
  fi
  if [[ -z "$label" ]]; then
    label="$(sanitize_label "$host")"
  fi

  echo "========================================"
  echo "FP4 checks: host=${host} label=${label}"
  echo "RUN_ID=${RUN_ID}"
  echo "RUNTIME=${RUNTIME}"
  echo "STACK_PROFILE=${STACK_PROFILE}"
  if [[ "$RUNTIME" == "container" ]]; then
    echo "IMAGE=${IMAGE}"
  fi
  echo "========================================"

  gpu_names="$(run_host_cmd "$host" "nvidia-smi --query-gpu=name --format=csv,noheader")"
  gpu_names="${gpu_names//$'\r'/}"
  if [[ -z "$gpu_names" ]]; then
    echo "ERROR: unable to detect GPU names on host ${host}" >&2
    exit 2
  fi
  gb_sku="$(printf '%s\n' "$gpu_names" | grep -Eoi 'GB[0-9]{3}' | head -n 1 | tr '[:lower:]' '[:upper:]' || true)"
  host_preset="$PRESET"
  if [[ "$PRESET" == "auto" ]]; then
    host_preset="all"
  fi

  if [[ -z "$gb_sku" ]]; then
    echo "ERROR: FP4 checks require GB-family GPUs (GB200/GB300/...). Host ${host} reported:" >&2
    printf '%s\n' "$gpu_names" | sed 's/^/  - /' >&2
    exit 2
  fi

  echo "Detected GPUs:"
  printf '%s\n' "$gpu_names" | sed 's/^/  - /'
  echo "Detected GB family SKU: ${gb_sku}"
  echo "FP4 grouped preset: ${host_preset} (requested: ${PRESET})"

if [[ "$RUNTIME" == "host" ]]; then
    ensure_deep_gemm_cmd="
set -euo pipefail
cd $(printf '%q' "${REMOTE_ROOT}")
if [[ ! -x ./env/venv/bin/python ]]; then
  echo 'ERROR: missing env/venv/bin/python; run bootstrap with --bootstrap-install-python-deps.' >&2
  exit 2
fi
if [[ $(printf '%q' "${STACK_PROFILE}") == host_only ]]; then
  if [[ ! -f ./env/venv/orig_parity_runtime_env.sh ]]; then
    echo 'ERROR: missing env/venv/orig_parity_runtime_env.sh for host_only parity.' >&2
    echo 'Run bootstrap with --bootstrap-host-parity-image to install host orig-parity binaries.' >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source ./env/venv/orig_parity_runtime_env.sh
fi
if ! ./env/venv/bin/python -c 'import deep_gemm' >/dev/null 2>&1; then
  echo 'ERROR: deep_gemm is not importable in env/venv on this host.' >&2
  echo 'Run bootstrap with --bootstrap-install-python-deps (host_only uses orig-parity image provenance).' >&2
  exit 2
fi
"
    run_host_cmd "$host" "$ensure_deep_gemm_cmd"
  fi

  attestation_target_abs="${REMOTE_ROOT}/${ATTESTATION_TARGET_REL}"
  if ! run_host_cmd "$host" "test -f $(printf '%q' "$attestation_target_abs")"; then
    echo "ERROR: required attestation target missing on ${host}: ${attestation_target_abs}" >&2
    exit 2
  fi

  attestation_snapshot_rel="$(copy_attestation_target_snapshot "$host" "$label")"
  attestation_snapshot_abs="${ROOT_DIR}/${attestation_snapshot_rel}"
  verify_local_artifact "${attestation_snapshot_rel}"

  if ! semantic_json="$(build_semantic_attestation "$attestation_snapshot_abs" "$ATTESTATION_PROFILE")"; then
    echo "ERROR: semantic attestation failed for host=${host} label=${label} target=${attestation_target_abs}" >&2
    exit 1
  fi
  semantic_signature="$(python3 -c 'import json,sys; print((json.loads(sys.stdin.read()) or {}).get("semantic_signature",""))' <<<"$semantic_json")"
  semantic_source_sha="$(python3 -c 'import json,sys; print((json.loads(sys.stdin.read()) or {}).get("source_sha256",""))' <<<"$semantic_json")"
  echo "Semantic attestation passed: signature=${semantic_signature} source_sha256=${semantic_source_sha}"

  grouped_log_rel="results/structured/${RUN_ID}_${label}_cluster_perf_grouped_gemm.txt"
  grouped_summary_rel="results/structured/${RUN_ID}_${label}_cluster_perf_grouped_gemm_summary.json"
  grouped_clock_rel="results/structured/${RUN_ID}_${label}_cluster_perf_grouped_gemm_clock_lock.json"
  grouped_plot_rel="docs/figures/${RUN_ID}_${label}_cluster_perf_grouped_gemm_tflops.png"
  grouped_args=(
    scripts/run_cluster_perf_grouped_gemm.sh
    --runtime "${RUNTIME}"
    --stack-profile "${STACK_PROFILE}"
    --run-id "${RUN_ID}"
    --label "${label}"
    --preset "${host_preset}"
    --warmup "${WARMUP}"
    --iters "${ITERS}"
    --require-deepgemm
  )
  if [[ "$RUNTIME" == "container" ]]; then
    grouped_args+=(--image "${IMAGE}")
  fi

  grouped_str="$(printf '%q ' "${grouped_args[@]}")"
  remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && ${grouped_str}"
  run_host_cmd "$host" "$remote_cmd"

  fetch_and_verify_if_remote "$host" "$grouped_log_rel"
  fetch_and_verify_if_remote "$host" "$grouped_summary_rel"
  fetch_and_verify_if_remote "$host" "$grouped_clock_rel"
  fetch_and_verify_if_remote "$host" "$grouped_plot_rel"

  repo_git_commit="$(trim_ws "$(run_host_cmd "$host" "git -C $(printf '%q' "$REMOTE_ROOT") rev-parse HEAD 2>/dev/null || true" | head -n 1)")"
  repo_git_dirty="unknown"
  if [[ -n "$repo_git_commit" ]]; then
    repo_git_dirty_line="$(trim_ws "$(run_host_cmd "$host" "git -C $(printf '%q' "$REMOTE_ROOT") status --porcelain --untracked-files=no 2>/dev/null | head -n 1" | head -n 1)")"
    if [[ -n "$repo_git_dirty_line" ]]; then
      repo_git_dirty="true"
    else
      repo_git_dirty="false"
    fi
  fi

  image_id=""
  image_repo_digests_json="[]"
  if [[ "$RUNTIME" == "container" ]]; then
    image_id="$(trim_ws "$(run_host_cmd "$host" "docker image inspect --format '{{.Id}}' $(printf '%q' "$IMAGE") 2>/dev/null || true" | head -n 1)")"
    if [[ -z "$image_id" ]]; then
      echo "ERROR: unable to capture container image ID for ${IMAGE} on host ${host}" >&2
      exit 1
    fi
    image_repo_digests_json="$(trim_ws "$(run_host_cmd "$host" "docker image inspect --format '{{json .RepoDigests}}' $(printf '%q' "$IMAGE") 2>/dev/null || true" | head -n 1)")"
    if [[ -z "$image_repo_digests_json" ]]; then
      image_repo_digests_json="[]"
    fi
  fi

  driver_version="$(trim_ws "$(run_host_cmd "$host" "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1" | head -n 1)")"
  if [[ -z "$driver_version" ]]; then
    driver_version="$(trim_ws "$(run_host_cmd "$host" "nvidia-smi 2>/dev/null | sed -n 's/.*Driver Version: \\([0-9.]*\\).*/\\1/p' | head -n 1" | head -n 1)")"
  fi
  cuda_version="$(trim_ws "$(run_host_cmd "$host" "nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \\([0-9.]*\\).*/\\1/p' | head -n 1" | head -n 1)")"
  torch_version="$(trim_ws "$(run_host_cmd "$host" "cd $(printf '%q' "$REMOTE_ROOT") && ./env/venv/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null || true" | head -n 1)")"
  deep_gemm_version="$(trim_ws "$(run_host_cmd "$host" "cd $(printf '%q' "$REMOTE_ROOT") && ./env/venv/bin/python -c 'import importlib.metadata as m; print(m.version(\"deep_gemm\"))' 2>/dev/null || true" | head -n 1)")"

  platform_meta_rel="results/structured/${RUN_ID}_${label}_cluster_perf_fp4_platform.json"
  platform_meta_abs="${ROOT_DIR}/${platform_meta_rel}"
  gpu_names_b64="$(printf '%s' "$gpu_names" | base64 | tr -d '\n')"
  semantic_json_b64="$(printf '%s' "$semantic_json" | base64 | tr -d '\n')"
  image_repo_digests_b64="$(printf '%s' "$image_repo_digests_json" | base64 | tr -d '\n')"
  write_platform_meta \
    "$platform_meta_abs" \
    "$host" \
    "$label" \
    "$PRESET" \
    "$host_preset" \
    "$gb_sku" \
    "$RUNTIME" \
    "$STACK_PROFILE" \
    "$IMAGE" \
    "$gpu_names_b64" \
    "$semantic_json_b64" \
    "$ATTESTATION_TARGET_REL" \
    "$attestation_snapshot_rel" \
    "$repo_git_commit" \
    "$repo_git_dirty" \
    "$image_id" \
    "$image_repo_digests_b64" \
    "$driver_version" \
    "$cuda_version" \
    "$torch_version" \
    "$deep_gemm_version" \
    "$grouped_summary_rel" \
    "$grouped_log_rel" \
    "$grouped_clock_rel"
  echo "Platform meta: ${platform_meta_rel}"
  verify_local_artifact "${platform_meta_rel}"
  ATTESTATION_LABELS+=("$label")
done

if [[ "${#ATTESTATION_LABELS[@]}" -gt 0 ]]; then
  attestation_labels_csv="$(IFS=,; echo "${ATTESTATION_LABELS[*]}")"
  attestation_consistency_rel="results/structured/${RUN_ID}_fp4_attestation_consistency.json"
  echo "Evaluating FP4 attestation consistency across hosts..."
  if ! write_attestation_consistency "${ROOT_DIR}" "${RUN_ID}" "${attestation_labels_csv}" "${ROOT_DIR}/${attestation_consistency_rel}"; then
    echo "ERROR: FP4 attestation consistency failed. See ${attestation_consistency_rel}" >&2
    exit 1
  fi
  echo "FP4 attestation consistency passed: ${attestation_consistency_rel}"
fi

if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  for round in $(seq 1 "$SMOKE_ROUNDS"); do
    echo "----------------------------------------"
    echo "FP4 paired smoke round ${round}/${SMOKE_ROUNDS}"
    for idx in "${!HOST_ARR[@]}"; do
      host="$(echo "${HOST_ARR[$idx]}" | xargs)"
      [[ -n "$host" ]] || continue

      label=""
      if [[ -n "$LABELS" ]]; then
        label="$(echo "${LABEL_ARR[$idx]}" | xargs)"
      fi
      if [[ -z "$label" ]]; then
        label="$(sanitize_label "$host")"
      fi

      round_run_id="${RUN_ID}_r${round}"
      smoke_args=(
        scripts/run_cluster_perf_fp4_smoke.sh
        --run-id "${round_run_id}"
        --label "${label}"
        --runtime "${RUNTIME}"
        --stack-profile "${STACK_PROFILE}"
        --m "${SMOKE_M}"
        --n "${SMOKE_N}"
        --k "${SMOKE_K}"
        --warmup "${SMOKE_WARMUP}"
        --iters "${SMOKE_ITERS}"
      )
      if [[ "$RUNTIME" == "container" ]]; then
        smoke_args+=(--image "${IMAGE}")
      fi
      smoke_str="$(printf '%q ' "${smoke_args[@]}")"
      remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && ${smoke_str}"
      run_host_cmd "$host" "$remote_cmd"

      fetch_and_verify_if_remote "$host" "results/raw/${round_run_id}_${label}_cluster_perf_fp4_smoke.log"
      fetch_and_verify_if_remote "$host" "results/structured/${round_run_id}_${label}_cluster_perf_fp4_smoke.json"
      fetch_and_verify_if_remote "$host" "results/structured/${round_run_id}_${label}_cluster_perf_fp4_smoke_clock_lock.json"
    done
  done

  guard_rel="results/structured/${RUN_ID}_fp4_smoke_skew_guard.json"
  echo "Evaluating FP4 smoke skew guard (rounds=${SMOKE_ROUNDS}, threshold_pct=${SMOKE_SKEW_THRESHOLD_PCT})..."
  if [[ -n "$TRIMMED_LABELS" ]]; then
    guard_labels="$TRIMMED_LABELS"
  else
    guard_labels=""
    for idx in "${!HOST_ARR[@]}"; do
      host="$(echo "${HOST_ARR[$idx]}" | xargs)"
      [[ -n "$host" ]] || continue
      label="$(sanitize_label "$host")"
      if [[ -n "$guard_labels" ]]; then
        guard_labels+=","
      fi
      guard_labels+="$label"
    done
  fi

  if ! write_smoke_skew_guard "${ROOT_DIR}" "${ROOT_DIR}/${guard_rel}" "${RUN_ID}" "${SMOKE_SKEW_THRESHOLD_PCT}" "${SMOKE_ROUNDS}" "${guard_labels}"; then
    echo "ERROR: FP4 smoke skew guard failed. See ${guard_rel}" >&2
    exit 1
  fi
  echo "FP4 smoke skew guard passed: ${guard_rel}"
fi

echo ""
echo "FP4 checks complete."
