#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/collect_container_runtime_all_nodes.sh --hosts <h1,h2,...> [options]

Writes per-node container runtime evidence to:
  results/structured/${RUN_ID}_${label}_container_runtime.txt

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required; hostnames or IPs)
  --labels <l1,l2,...>   Optional comma-separated labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
EOF
}

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
HOSTS=""
LABELS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_DIR"

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

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
IFS=',' read -r -a LABEL_ARR <<<"$LABELS"

if [[ -n "$LABELS" && "${#LABEL_ARR[@]}" -ne "${#HOST_ARR[@]}" ]]; then
  echo "ERROR: --labels count must match --hosts count" >&2
  exit 2
fi

sanitize_label() {
  local raw="$1"
  raw="${raw//./_}"
  raw="${raw//:/_}"
  echo "$raw"
}

is_local_host() {
  local host="$1"
  local hn hn_s
  hn="$(hostname 2>/dev/null || true)"
  hn_s="$(hostname -s 2>/dev/null || true)"
  if [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "::1" ]]; then
    return 0
  fi
  if [[ -n "$hn" && "$host" == "$hn" ]]; then
    return 0
  fi
  if [[ -n "$hn_s" && "$host" == "$hn_s" ]]; then
    return 0
  fi
  return 1
}

collect_cmd='
set -euo pipefail
echo "== container runtime evidence =="
date -Iseconds
id || true
groups || true
ls -l /var/run/docker.sock || true
docker --version || true
docker info 2>/dev/null | grep -E -i "^( Runtimes:| Default Runtime:)" || true
docker info --format "DefaultRuntime={{.DefaultRuntime}}" 2>/dev/null || true
runc --version 2>/dev/null || true
crun --version 2>/dev/null || true
nvidia-ctk --version 2>/dev/null || true
nvidia-container-cli --version 2>/dev/null || true
nvidia-container-runtime --version 2>/dev/null || true
nvidia-container-toolkit --version 2>/dev/null || true
docker ps --format "{{.ID}} {{.Image}}" 2>/dev/null | head || true
echo "containerd: $(systemctl is-active containerd 2>/dev/null || true)"
grep -n "disabled_plugins" /etc/containerd/config.toml 2>/dev/null || true
dpkg -l | grep -E "(docker|containerd|nvidia-container)" | head -n 120 || true

echo "== cve checks =="
# CVE-2025-23266 ("NVIDIAScape") is fixed in NVIDIA Container Toolkit >= 1.17.8.
# NVIDIA notes it does not affect systems where crun is used as the low-level runtime.
extract_semver() {
  echo "$1" | grep -oE "[0-9]+\\.[0-9]+\\.[0-9]+" | head -n 1 || true
}
semver_ge() {
  local a b a1 a2 a3 b1 b2 b3
  a="$(extract_semver "$1")"
  b="$(extract_semver "$2")"
  [[ -n "$a" && -n "$b" ]] || return 2
  IFS=. read -r a1 a2 a3 <<<"$a"
  IFS=. read -r b1 b2 b3 <<<"$b"
  if (( a1 != b1 )); then (( a1 > b1 )); return $?; fi
  if (( a2 != b2 )); then (( a2 > b2 )); return $?; fi
  (( a3 >= b3 )); return $?
}
ver_ge() {
  local have="$1"
  local need="$2"
  if command -v dpkg >/dev/null 2>&1; then
    dpkg --compare-versions "$have" ge "$need"
    return $?
  fi
  semver_ge "$have" "$need"
  return $?
}

nct_pkg_ver=""
if command -v dpkg >/dev/null 2>&1; then
  nct_pkg_ver="$(dpkg -s nvidia-container-toolkit 2>/dev/null | sed -n "s/^Version: //p" | head -n 1 || true)"
elif command -v rpm >/dev/null 2>&1; then
  nct_pkg_ver="$(rpm -q --qf "%{VERSION}-%{RELEASE}\n" nvidia-container-toolkit 2>/dev/null || true)"
fi

default_runtime="$(docker info --format "{{.DefaultRuntime}}" 2>/dev/null || true)"
runtime_kind="unknown"
if [[ "$default_runtime" == *crun* ]]; then
  runtime_kind="crun"
elif [[ "$default_runtime" == *runc* ]]; then
  runtime_kind="runc"
fi

cve_23266_status="unknown"
cve_23266_reason="missing nvidia-container-toolkit version"
if [[ -n "$nct_pkg_ver" ]]; then
  if [[ "$runtime_kind" == "crun" ]]; then
    cve_23266_status="pass"
    cve_23266_reason="not affected when low-level runtime is crun (per NVIDIA)"
  else
    if ver_ge "$nct_pkg_ver" "1.17.8"; then
      cve_23266_status="pass"
      cve_23266_reason="nvidia-container-toolkit >= 1.17.8"
    else
      rc=$?
      if [[ "$rc" -eq 2 ]]; then
        cve_23266_status="unknown"
        cve_23266_reason="could not compare versions (have=${nct_pkg_ver})"
      else
        cve_23266_status="fail"
        cve_23266_reason="nvidia-container-toolkit < 1.17.8"
      fi
    fi
  fi
fi

echo "cve_2025_23266_status=${cve_23266_status}"
echo "cve_2025_23266_reason=${cve_23266_reason}"
echo "cve_2025_23266_nvidia_container_toolkit_version=${nct_pkg_ver:-<unknown>}"
echo "cve_2025_23266_docker_default_runtime=${default_runtime:-<unknown>}"

# If this is a K8s cluster using NVIDIA GPU Operator, also capture/operator version for completeness.
gpu_operator_ns="missing"
gpu_operator_image=""
gpu_operator_ver=""
if command -v kubectl >/dev/null 2>&1; then
  if kubectl --request-timeout=5s get ns gpu-operator >/dev/null 2>&1; then
    gpu_operator_ns="present"
    gpu_operator_image="$(kubectl --request-timeout=5s -n gpu-operator get deployment gpu-operator -o jsonpath="{.spec.template.spec.containers[0].image}" 2>/dev/null || true)"
    if [[ -n "$gpu_operator_image" ]]; then
      tag="${gpu_operator_image##*:}"
      gpu_operator_ver="${tag#v}"
    fi
  fi
fi
echo "gpu_operator_namespace=${gpu_operator_ns}"
echo "gpu_operator_image=${gpu_operator_image:-<unknown>}"
echo "gpu_operator_version=${gpu_operator_ver:-<unknown>}"

# GPU Operator fix version for CVE-2025-23266 is >= 25.3.2 (if GPU Operator is in use).
gpu_operator_cve_23266_status="n/a"
gpu_operator_cve_23266_reason="gpu-operator namespace missing"
if [[ "$gpu_operator_ns" == "present" ]]; then
  if [[ -z "$gpu_operator_ver" ]]; then
    gpu_operator_cve_23266_status="unknown"
    gpu_operator_cve_23266_reason="could not detect gpu-operator version"
  else
    if semver_ge "$gpu_operator_ver" "25.3.2"; then
      gpu_operator_cve_23266_status="pass"
      gpu_operator_cve_23266_reason="gpu-operator >= 25.3.2"
    else
      rc=$?
      if [[ "$rc" -eq 2 ]]; then
        gpu_operator_cve_23266_status="unknown"
        gpu_operator_cve_23266_reason="could not compare versions (have=${gpu_operator_ver})"
      else
        gpu_operator_cve_23266_status="fail"
        gpu_operator_cve_23266_reason="gpu-operator < 25.3.2"
      fi
    fi
  fi
fi
echo "gpu_operator_cve_2025_23266_status=${gpu_operator_cve_23266_status}"
echo "gpu_operator_cve_2025_23266_reason=${gpu_operator_cve_23266_reason}"

# CVE-2025-23267 is fixed in NVIDIA Container Toolkit >= 1.17.7.
cve_23267_status="unknown"
cve_23267_reason="missing nvidia-container-toolkit version"
if [[ -n "$nct_pkg_ver" ]]; then
  if ver_ge "$nct_pkg_ver" "1.17.7"; then
    cve_23267_status="pass"
    cve_23267_reason="nvidia-container-toolkit >= 1.17.7"
  else
    rc=$?
    if [[ "$rc" -eq 2 ]]; then
      cve_23267_status="unknown"
      cve_23267_reason="could not compare versions (have=${nct_pkg_ver})"
    else
      cve_23267_status="fail"
      cve_23267_reason="nvidia-container-toolkit < 1.17.7"
    fi
  fi
fi

echo "cve_2025_23267_status=${cve_23267_status}"
echo "cve_2025_23267_reason=${cve_23267_reason}"
'

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

  out_file="${OUT_DIR}/${RUN_ID}_${label}_container_runtime.txt"
  echo "=== ${label} (${host}) ==="

  if is_local_host "$host"; then
    bash -lc "$collect_cmd" >"$out_file"
    echo "Saved ${out_file}"
    continue
  fi

  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "bash -lc '$collect_cmd'" >"$out_file" || {
    echo "ERROR: collection failed for ${host}" >&2
    continue
  }
  echo "Saved ${out_file}"
done
