#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/bootstrap_cluster_nodes.sh --hosts <h1,h2,...> [options]

Bootstraps nodes so cluster checks are reproducible:
  1) Syncs harness code (scripts/, analysis/, configs/, env/requirements.txt)
  2) Optionally installs missing system packages required by checks
  3) Ensures env/venv exists and installs Python deps (torch + plotting deps)
  4) Optionally syncs Cluster Perf standalone compute suite for FP4 checks
  5) Writes per-node bootstrap status JSON in results/structured/

Options:
  --run-id <id>              RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>        Comma-separated host list (required)
  --labels <l1,l2,...>       Optional labels (must match host count)
  --ssh-user <user>          SSH user (default: ubuntu)
  --ssh-key <path>           SSH key path (default: $SSH_KEY)
  --remote-root <path>       Repo root on remote nodes (default: this repo root)

  --sync-code                Sync scripts/analysis/env requirements to remote nodes (default: on)
  --skip-sync-code           Skip code sync
  --install-system-packages  Install missing system packages for checks (default: on)
  --skip-system-packages     Skip system package installation
  --install-python-deps      Ensure env/venv + Python deps (default: on)
  --skip-python-deps         Skip Python dependency install

  --sync-suite-dir <dir>     Optional local Cluster Perf suite path to sync to remote.
                             Accepted forms:
                               - suite root containing standalone/compute/
                               - standalone/ directory
                               - standalone/compute/ directory
                               - parent directory containing a single suite root
  --host-parity-image <ref>  Source image for host-only orig-parity binaries
                             (default: cfregly/cluster_perf_orig_parity:latest)
  --torch-index-url <url>    Legacy fallback torch index (default: https://pypi.ngc.nvidia.com)
  --torch-version <ver>      Expected torch version after host parity install
                             (default: 2.10.0a0+a36e1d39eb.nv26.01.42222806)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
HOSTS=""
LABELS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"
REMOTE_ROOT="${REMOTE_ROOT:-$ROOT_DIR}"

SYNC_CODE=1
INSTALL_SYSTEM_PACKAGES=1
INSTALL_PYTHON_DEPS=1

SYNC_SUITE_DIR=""
TORCH_INDEX_URL="https://pypi.ngc.nvidia.com"
TORCH_VERSION="2.10.0a0+a36e1d39eb.nv26.01.42222806"
TORCH_CUDA_VERSION="13.1"
TORCH_CUDNN_VERSION="91701"
TORCH_NCCL_VERSION="2.29.2"
DEEP_GEMM_VERSION="2.3.0+0f5f266"
HOST_PARITY_IMAGE="${HOST_PARITY_IMAGE:-cfregly/cluster_perf_orig_parity:latest}"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --hosts) HOSTS="${2:-}"; shift 2 ;;
    --labels) LABELS="${2:-}"; shift 2 ;;
    --ssh-user) SSH_USER="${2:-}"; shift 2 ;;
    --ssh-key) SSH_KEY="${2:-}"; shift 2 ;;
    --remote-root) REMOTE_ROOT="${2:-}"; shift 2 ;;
    --sync-code) SYNC_CODE=1; shift ;;
    --skip-sync-code) SYNC_CODE=0; shift ;;
    --install-system-packages) INSTALL_SYSTEM_PACKAGES=1; shift ;;
    --skip-system-packages) INSTALL_SYSTEM_PACKAGES=0; shift ;;
    --install-python-deps) INSTALL_PYTHON_DEPS=1; shift ;;
    --skip-python-deps) INSTALL_PYTHON_DEPS=0; shift ;;
    --sync-suite-dir) SYNC_SUITE_DIR="${2:-}"; shift 2 ;;
    --host-parity-image) HOST_PARITY_IMAGE="${2:-}"; shift 2 ;;
    --torch-index-url) TORCH_INDEX_URL="${2:-}"; shift 2 ;;
    --torch-version) TORCH_VERSION="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

if [[ "$SYNC_CODE" -eq 1 && ! -x "$(command -v rsync || true)" ]]; then
  echo "ERROR: rsync is required locally for --sync-code." >&2
  exit 2
fi

resolve_suite_dir() {
  local raw="${1:-}"
  local cand=""
  if [[ -z "$raw" ]]; then
    return 1
  fi
  if [[ -d "${raw}/standalone/compute" ]]; then
    (cd "$raw" && pwd -P)
    return 0
  fi
  if [[ -d "$raw" ]]; then
    if [[ "$(basename "$raw")" == "compute" && "$(basename "$(dirname "$raw")")" == "standalone" ]]; then
      (cd "$(dirname "$(dirname "$raw")")" && pwd -P)
      return 0
    fi
    if [[ "$(basename "$raw")" == "standalone" && -d "${raw}/compute" ]]; then
      (cd "$(dirname "$raw")" && pwd -P)
      return 0
    fi
    for cand in "${raw}"/* "${raw}"/*/*; do
      [[ -d "$cand" ]] || continue
      if [[ -d "${cand}/standalone/compute" ]]; then
        (cd "$cand" && pwd -P)
        return 0
      fi
    done
  fi
  local parent
  parent="$(dirname "$raw")"
  if [[ -d "$parent" ]]; then
    local -a sibling_matches=()
    for cand in "${parent}"/*; do
      [[ -d "$cand" ]] || continue
      if [[ -d "${cand}/standalone/compute" && "$(basename "$cand")" == "$(basename "$raw")" ]]; then
        (cd "$cand" && pwd -P)
        return 0
      fi
      if [[ -d "${cand}/standalone/compute" ]]; then
        sibling_matches+=("$cand")
      fi
    done
    if [[ "${#sibling_matches[@]}" -eq 1 ]]; then
      (cd "${sibling_matches[0]}" && pwd -P)
      return 0
    fi
  fi
  return 1
}

if [[ -n "$SYNC_SUITE_DIR" ]]; then
  resolved_suite_dir="$(resolve_suite_dir "$SYNC_SUITE_DIR" || true)"
  if [[ -z "$resolved_suite_dir" ]]; then
    echo "ERROR: --sync-suite-dir must resolve to a suite root containing standalone/compute." >&2
    echo "Provided: ${SYNC_SUITE_DIR}" >&2
    exit 2
  fi
  if [[ "$resolved_suite_dir" != "$SYNC_SUITE_DIR" ]]; then
    echo "Resolved --sync-suite-dir: ${SYNC_SUITE_DIR} -> ${resolved_suite_dir}"
  fi
  SYNC_SUITE_DIR="$resolved_suite_dir"
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

build_rsync_ssh() {
  local parts=("ssh")
  local opt
  for opt in "${SSH_OPTS[@]}"; do
    parts+=("$opt")
  done
  printf '%q ' "${parts[@]}"
}
RSYNC_SSH_CMD="$(build_rsync_ssh)"

run_remote() {
  local host="$1"
  shift
  # Avoid consuming stdin of callers (e.g., here-doc loops) during SSH command execution.
  # Note: do NOT add -n to SSH_OPTS globally because rsync and docker streaming use SSH stdin.
  ssh -n "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "$@"
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

is_local_host() {
  local host="$1"
  local hn hn_s
  hn="$(hostname 2>/dev/null || true)"
  hn_s="$(hostname -s 2>/dev/null || true)"
  [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "::1" ]] && return 0
  [[ -n "$hn" && "$host" == "$hn" ]] && return 0
  [[ -n "$hn_s" && "$host" == "$hn_s" ]] && return 0
  return 1
}

ensure_parity_image_on_host() {
  local host="$1"
  local q_parity_image
  printf -v q_parity_image '%q' "$HOST_PARITY_IMAGE"

  if run_host_cmd "$host" "docker image inspect ${q_parity_image} >/dev/null 2>&1"; then
    return 0
  fi

  if ! is_local_host "$host" && command -v docker >/dev/null 2>&1 && docker image inspect "$HOST_PARITY_IMAGE" >/dev/null 2>&1; then
    echo "INFO: host ${host} missing parity image; streaming local cache (${HOST_PARITY_IMAGE})"
    if docker save "$HOST_PARITY_IMAGE" | ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "docker load >/dev/null"; then
      if run_host_cmd "$host" "docker image inspect ${q_parity_image} >/dev/null 2>&1"; then
        return 0
      fi
      echo "WARNING: streamed parity image to ${host}, but ${HOST_PARITY_IMAGE} was not resolved afterward; falling back to pull." >&2
    else
      echo "WARNING: failed to stream local parity image to ${host}; falling back to pull." >&2
    fi
  fi

  echo "INFO: pulling parity image on host ${host}: ${HOST_PARITY_IMAGE}"
  run_host_cmd "$host" "docker pull ${q_parity_image} >/dev/null"
  run_host_cmd "$host" "docker image inspect ${q_parity_image} >/dev/null 2>&1"
}

mkdir -p "${ROOT_DIR}/results/raw" "${ROOT_DIR}/results/structured"
LOG_DIR="${ROOT_DIR}/results/raw/${RUN_ID}_bootstrap_nodes"
mkdir -p "$LOG_DIR"

check_cmd_path() {
  local host="$1"
  local cmd="$2"
  local out
  out="$(run_host_cmd "$host" "command -v ${cmd} || true" | tr -d '\r' | head -n1 | xargs || true)"
  echo "$out"
}

sync_code_to_host() {
  local host="$1"
  if is_local_host "$host"; then
    return 0
  fi

  run_host_cmd "$host" "mkdir -p $(printf '%q' "${REMOTE_ROOT}/scripts") $(printf '%q' "${REMOTE_ROOT}/analysis") $(printf '%q' "${REMOTE_ROOT}/configs") $(printf '%q' "${REMOTE_ROOT}/env")"

  rsync -az --exclude '__pycache__' -e "$RSYNC_SSH_CMD" \
    "${ROOT_DIR}/scripts/" "${SSH_USER}@${host}:${REMOTE_ROOT}/scripts/"
  rsync -az --exclude '__pycache__' -e "$RSYNC_SSH_CMD" \
    "${ROOT_DIR}/analysis/" "${SSH_USER}@${host}:${REMOTE_ROOT}/analysis/"
  rsync -az --exclude '__pycache__' -e "$RSYNC_SSH_CMD" \
    "${ROOT_DIR}/configs/" "${SSH_USER}@${host}:${REMOTE_ROOT}/configs/"
  rsync -az -e "$RSYNC_SSH_CMD" \
    "${ROOT_DIR}/env/requirements.txt" "${SSH_USER}@${host}:${REMOTE_ROOT}/env/requirements.txt"
}

sync_suite_to_host() {
  local host="$1"
  if [[ -z "$SYNC_SUITE_DIR" ]]; then
    return 0
  fi
  if is_local_host "$host"; then
    return 0
  fi
  run_host_cmd "$host" "mkdir -p $(printf '%q' "${SYNC_SUITE_DIR}/standalone/compute")"
  rsync -az --exclude '__pycache__' -e "$RSYNC_SSH_CMD" \
    "${SYNC_SUITE_DIR}/standalone/compute/" \
    "${SSH_USER}@${host}:${SYNC_SUITE_DIR}/standalone/compute/"
}

install_system_packages_on_host() {
  local host="$1"
  local -a missing_pkgs=()

  # Command -> package mapping for current harness checks.
  local cmd pkg
  while IFS=':' read -r cmd pkg; do
    [[ -n "$cmd" ]] || continue
    if [[ -z "$(check_cmd_path "$host" "$cmd")" ]]; then
      missing_pkgs+=("$pkg")
    fi
  done <<'EOF'
python3:python3
pip3:python3-pip
git:git
rsync:rsync
numactl:numactl
mpirun:openmpi-bin
iperf3:iperf3
ethtool:ethtool
rdma:rdma-core
ib_write_bw:perftest
ibstat:infiniband-diags
fio:fio
jq:jq
wget:wget
curl:curl
bc:bc
ping:iputils-ping
ip:iproute2
docker:docker.io
whois:whois
speedtest-cli:speedtest-cli
EOF

  # python -m venv capability.
  if [[ -z "$(run_host_cmd "$host" "python3 -m venv --help >/dev/null 2>&1 && echo ok || true" | tr -d '\r' | xargs || true)" ]]; then
    missing_pkgs+=("python3-venv")
  fi

  # nvcc is required by several benchmark build paths.
  if [[ -z "$(check_cmd_path "$host" "nvcc")" ]]; then
    missing_pkgs+=("cuda-toolkit-13-0")
  fi
  # Nsight tools are required by profiling workflows in this repo.
  if [[ -z "$(check_cmd_path "$host" "ncu")" ]]; then
    missing_pkgs+=("cuda-nsight-compute-13-0")
  fi
  if [[ -z "$(check_cmd_path "$host" "nsys")" ]]; then
    missing_pkgs+=("cuda-nsight-systems-13-0")
  fi

  if [[ "${#missing_pkgs[@]}" -eq 0 ]]; then
    echo ""
    return 0
  fi

  local unique_pkgs
  unique_pkgs="$(printf '%s\n' "${missing_pkgs[@]}" | awk 'NF && !seen[$0]++' | tr '\n' ' ')"
  unique_pkgs="$(echo "$unique_pkgs" | xargs || true)"
  if [[ -z "$unique_pkgs" ]]; then
    echo ""
    return 0
  fi

  run_host_cmd "$host" "sudo DEBIAN_FRONTEND=noninteractive apt-get update >/dev/null && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ${unique_pkgs} >/dev/null"
  # Some CUDA apt packages install Nsight Compute under /usr/local/cuda-*/bin
  # without adding it to PATH. Expose `ncu` on PATH for harness tooling.
  run_host_cmd "$host" "if ! command -v ncu >/dev/null 2>&1; then for p in /usr/local/cuda/bin/ncu /usr/local/cuda-13.0/bin/ncu /usr/local/cuda-13.1/bin/ncu; do if [[ -x \"\$p\" ]]; then sudo ln -sf \"\$p\" /usr/local/bin/ncu; break; fi; done; fi"
  echo "$unique_pkgs"
}

install_python_deps_on_host() {
  local host="$1"
  local venv_dir="${REMOTE_ROOT}/env/venv"
  local venv_py="${venv_dir}/bin/python"
  local installer_script="${REMOTE_ROOT}/scripts/install_host_orig_parity_stack.sh"
  local req_file="${REMOTE_ROOT}/env/requirements.txt"

  ensure_parity_image_on_host "$host"

  local q_venv_dir q_venv_py q_installer q_req q_parity_image
  local q_torch_version q_torch_cuda q_torch_cudnn q_torch_nccl q_deep_gemm_version
  printf -v q_venv_dir '%q' "$venv_dir"
  printf -v q_venv_py '%q' "$venv_py"
  printf -v q_installer '%q' "$installer_script"
  printf -v q_req '%q' "$req_file"
  printf -v q_parity_image '%q' "$HOST_PARITY_IMAGE"
  printf -v q_torch_version '%q' "$TORCH_VERSION"
  printf -v q_torch_cuda '%q' "$TORCH_CUDA_VERSION"
  printf -v q_torch_cudnn '%q' "$TORCH_CUDNN_VERSION"
  printf -v q_torch_nccl '%q' "$TORCH_NCCL_VERSION"
  printf -v q_deep_gemm_version '%q' "$DEEP_GEMM_VERSION"

  local cmd="
set -euo pipefail
mkdir -p $(printf '%q' "${REMOTE_ROOT}/env")
if [[ ! -x ${q_installer} ]]; then
  echo 'ERROR: missing host parity installer script: ${installer_script}' >&2
  exit 2
fi
if [[ ! -x ${q_venv_py} ]]; then
  python3 -m venv ${q_venv_dir}
fi
${q_installer} \
  --venv-dir ${q_venv_dir} \
  --requirements-file ${q_req} \
  --parity-image ${q_parity_image} \
  --expected-torch-version ${q_torch_version} \
  --expected-cuda-version ${q_torch_cuda} \
  --expected-cudnn-version ${q_torch_cudnn} \
  --expected-nccl-version ${q_torch_nccl} \
  --expected-deep-gemm-version ${q_deep_gemm_version}
"
  run_host_cmd "$host" "$cmd"
}

write_status_json() {
  local out_path="$1"
  local host="$2"
  local label="$3"
  local installed_pkgs="$4"

  local python3_path pip3_path docker_path nvidia_smi_path nvcc_path mpirun_path ib_write_bw_path ibstat_path iperf3_path numactl_path
  local fio_path jq_path wget_path curl_path ncu_path nsys_path dcgmi_path
  local nvbandwidth_path
  local venv_py torch_version torch_cuda_available matplotlib_version numpy_version deep_gemm_version

  python3_path="$(check_cmd_path "$host" "python3")"
  pip3_path="$(check_cmd_path "$host" "pip3")"
  docker_path="$(check_cmd_path "$host" "docker")"
  nvidia_smi_path="$(check_cmd_path "$host" "nvidia-smi")"
  nvcc_path="$(check_cmd_path "$host" "nvcc")"
  mpirun_path="$(check_cmd_path "$host" "mpirun")"
  ib_write_bw_path="$(check_cmd_path "$host" "ib_write_bw")"
  ibstat_path="$(check_cmd_path "$host" "ibstat")"
  iperf3_path="$(check_cmd_path "$host" "iperf3")"
  numactl_path="$(check_cmd_path "$host" "numactl")"
  fio_path="$(check_cmd_path "$host" "fio")"
  jq_path="$(check_cmd_path "$host" "jq")"
  wget_path="$(check_cmd_path "$host" "wget")"
  curl_path="$(check_cmd_path "$host" "curl")"
  ncu_path="$(check_cmd_path "$host" "ncu")"
  nsys_path="$(check_cmd_path "$host" "nsys")"
  dcgmi_path="$(check_cmd_path "$host" "dcgmi")"
  nvbandwidth_path="$(check_cmd_path "$host" "nvbandwidth")"

  venv_py="${REMOTE_ROOT}/env/venv/bin/python"
  torch_version="$(run_host_cmd "$host" "${venv_py} -c \"import torch; print(torch.__version__)\" 2>/dev/null || true" | tr -d '\r' | head -n1 | xargs || true)"
  torch_cuda_available="$(run_host_cmd "$host" "${venv_py} -c \"import torch; print(int(torch.cuda.is_available()))\" 2>/dev/null || true" | tr -d '\r' | head -n1 | xargs || true)"
  matplotlib_version="$(run_host_cmd "$host" "${venv_py} -c \"import matplotlib; print(matplotlib.__version__)\" 2>/dev/null || true" | tr -d '\r' | head -n1 | xargs || true)"
  numpy_version="$(run_host_cmd "$host" "${venv_py} -c \"import numpy; print(numpy.__version__)\" 2>/dev/null || true" | tr -d '\r' | head -n1 | xargs || true)"
  deep_gemm_version="$(run_host_cmd "$host" "${venv_py} -c \"import importlib.metadata as m; print(m.version('deep_gemm'))\" 2>/dev/null || true" | tr -d '\r' | head -n1 | xargs || true)"

  python3 - <<'PY' "$out_path" "$host" "$label" "$SYNC_CODE" "$INSTALL_SYSTEM_PACKAGES" "$INSTALL_PYTHON_DEPS" "$installed_pkgs" "$REMOTE_ROOT" "$python3_path" "$pip3_path" "$docker_path" "$nvidia_smi_path" "$nvcc_path" "$mpirun_path" "$ib_write_bw_path" "$ibstat_path" "$iperf3_path" "$numactl_path" "$fio_path" "$jq_path" "$wget_path" "$curl_path" "$ncu_path" "$nsys_path" "$dcgmi_path" "$nvbandwidth_path" "$venv_py" "$torch_version" "$torch_cuda_available" "$matplotlib_version" "$numpy_version" "$deep_gemm_version"
import json
import sys
import time
from pathlib import Path

(
    out_path,
    host,
    label,
    sync_code,
    install_system_packages,
    install_python_deps,
    installed_pkgs,
    remote_root,
    python3_path,
    pip3_path,
    docker_path,
    nvidia_smi_path,
    nvcc_path,
    mpirun_path,
    ib_write_bw_path,
    ibstat_path,
    iperf3_path,
    numactl_path,
    fio_path,
    jq_path,
    wget_path,
    curl_path,
    ncu_path,
    nsys_path,
    dcgmi_path,
    nvbandwidth_path,
    venv_py,
    torch_version,
    torch_cuda_available,
    matplotlib_version,
    numpy_version,
    deep_gemm_version,
) = sys.argv[1:]

payload = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "host": host,
    "label": label,
    "remote_root": remote_root,
    "bootstrap": {
        "sync_code": bool(int(sync_code)),
        "install_system_packages": bool(int(install_system_packages)),
        "install_python_deps": bool(int(install_python_deps)),
        "installed_system_packages": [p for p in installed_pkgs.split() if p],
    },
    "commands": {
        "python3": python3_path or None,
        "pip3": pip3_path or None,
        "docker": docker_path or None,
        "nvidia-smi": nvidia_smi_path or None,
        "nvcc": nvcc_path or None,
        "mpirun": mpirun_path or None,
        "ib_write_bw": ib_write_bw_path or None,
        "ibstat": ibstat_path or None,
        "iperf3": iperf3_path or None,
        "numactl": numactl_path or None,
        "fio": fio_path or None,
        "jq": jq_path or None,
        "wget": wget_path or None,
        "curl": curl_path or None,
        "ncu": ncu_path or None,
        "nsys": nsys_path or None,
        "dcgmi": dcgmi_path or None,
        "nvbandwidth": nvbandwidth_path or None,
    },
    "python_env": {
        "venv_python": venv_py,
        "torch_version": torch_version or None,
        "torch_cuda_available": bool(int(torch_cuda_available)) if torch_cuda_available in {"0", "1"} else None,
        "matplotlib_version": matplotlib_version or None,
        "numpy_version": numpy_version or None,
        "deep_gemm_version": deep_gemm_version or None,
    },
}

out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

echo "========================================"
echo "Bootstrap Cluster Nodes"
echo "========================================"
echo "RUN_ID=${RUN_ID}"
echo "HOSTS=${HOSTS}"
echo "REMOTE_ROOT=${REMOTE_ROOT}"
echo "SYNC_CODE=${SYNC_CODE}"
echo "INSTALL_SYSTEM_PACKAGES=${INSTALL_SYSTEM_PACKAGES}"
echo "INSTALL_PYTHON_DEPS=${INSTALL_PYTHON_DEPS}"
echo "HOST_PARITY_IMAGE=${HOST_PARITY_IMAGE}"
if [[ -n "$SYNC_SUITE_DIR" ]]; then
  echo "SYNC_SUITE_DIR=${SYNC_SUITE_DIR}"
fi
echo ""

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

  host_log="${LOG_DIR}/${label}.log"
  {
    echo "========================================"
    echo "Bootstrap host=${host} label=${label}"
    echo "========================================"

    if [[ "$SYNC_CODE" -eq 1 ]]; then
      echo "-- sync code"
      sync_code_to_host "$host"
    fi

    if [[ -n "$SYNC_SUITE_DIR" ]]; then
      echo "-- sync suite dir"
      sync_suite_to_host "$host"
    fi

    installed_pkgs=""
    if [[ "$INSTALL_SYSTEM_PACKAGES" -eq 1 ]]; then
      echo "-- install missing system packages"
      installed_pkgs="$(install_system_packages_on_host "$host")"
      if [[ -n "$installed_pkgs" ]]; then
        echo "installed: ${installed_pkgs}"
      else
        echo "installed: <none>"
      fi
    fi

    if [[ "$INSTALL_PYTHON_DEPS" -eq 1 ]]; then
      echo "-- install python deps"
      install_python_deps_on_host "$host"
    fi

    out_json="${ROOT_DIR}/results/structured/${RUN_ID}_${label}_bootstrap_status.json"
    write_status_json "$out_json" "$host" "$label" "$installed_pkgs"
    echo "status_json: ${out_json}"
  } | tee "$host_log"
done

echo ""
echo "Bootstrap complete."
