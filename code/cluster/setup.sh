#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: setup.sh [options]

Options:
  --run-id <id>            RUN_ID prefix (default: YYYY-MM-DD)
  --label <label>          Node label for discovery metadata (default: node1)
  --venv <path>            Python venv path (default: cluster/env/venv)
  --nccl-tests-dir <path>  nccl-tests checkout path (default: cluster/tools/nccl-tests)
  --torch-index-url <url>  PyTorch wheel index URL (default: https://pypi.ngc.nvidia.com)
  --torch-version <ver>    Torch version to install (default: 2.10.0a0+a36e1d39eb.nv26.01.42222806)
  --skip-discovery         Skip discovery metadata capture
  --skip-apt               Skip apt package installs
  --skip-python            Skip Python venv + torch install
  --skip-nccl-tests         Skip nccl-tests clone/build
  --install-vllm           Install vLLM inside the venv (default: off)
  -h, --help               Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$(date +%Y-%m-%d)"
LABEL="node1"
VENV_DIR="${ROOT_DIR}/env/venv"
VENV_PY="${VENV_DIR}/bin/python"
NCCL_TESTS_DIR="${ROOT_DIR}/tools/nccl-tests"
TORCH_INDEX_URL="https://pypi.ngc.nvidia.com"
TORCH_VERSION="2.10.0a0+a36e1d39eb.nv26.01.42222806"
TRANSFORMERS_VERSION="4.56.0"
DATASETS_VERSION="2.21.0"
SENTENCEPIECE_VERSION="0.2.0"
TOKENIZERS_VERSION="0.22.2"
HUGGINGFACE_HUB_VERSION="0.34.6"
FSSPEC_VERSION="2024.6.1"
SKIP_DISCOVERY=0
SKIP_APT=0
SKIP_PYTHON=0
SKIP_NCCL_TESTS=0
INSTALL_VLLM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --nccl-tests-dir)
      NCCL_TESTS_DIR="$2"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --skip-discovery)
      SKIP_DISCOVERY=1
      shift
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --skip-python)
      SKIP_PYTHON=1
      shift
      ;;
    --skip-nccl-tests)
      SKIP_NCCL_TESTS=1
      shift
      ;;
    --install-vllm)
      INSTALL_VLLM=1
      shift
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

resolve_torch_lib_dir() {
  local py_bin="$1"
  if [[ ! -x "$py_bin" ]]; then
    return 0
  fi

  "$py_bin" - <<'PY'
import os
import site
import sys
from pathlib import Path

roots = []
py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
roots.append(Path(sys.prefix) / "lib" / py_version / "site-packages")

venv = os.environ.get("VIRTUAL_ENV")
if venv:
    roots.append(Path(venv) / "lib" / py_version / "site-packages")

try:
    roots.extend(Path(p) for p in site.getsitepackages())
except Exception:
    pass

try:
    roots.append(Path(site.getusersitepackages()))
except Exception:
    pass

seen = set()
for root in roots:
    candidate = root / "torch" / "lib"
    key = str(candidate)
    if key in seen:
        continue
    seen.add(key)
    if candidate.is_dir():
        print(candidate)
        raise SystemExit(0)

print("")
PY
}

write_runtime_env_file() {
  local runtime_env_file="$1"
  local torch_lib_dir="$2"
  mkdir -p "$(dirname "$runtime_env_file")"

  if [[ -n "$torch_lib_dir" ]]; then
    cat >"${runtime_env_file}" <<EOF
#!/usr/bin/env bash
export AISP_CUDNN_RUNTIME_POLICY="\${AISP_CUDNN_RUNTIME_POLICY:-auto}"
if [[ -d "${torch_lib_dir}" ]]; then
  export LD_LIBRARY_PATH="${torch_lib_dir}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
fi
EOF
  else
    cat >"${runtime_env_file}" <<'EOF'
#!/usr/bin/env bash
export AISP_CUDNN_RUNTIME_POLICY="${AISP_CUDNN_RUNTIME_POLICY:-auto}"
EOF
  fi

  chmod 0755 "${runtime_env_file}"
}

OUT_STRUCT_DIR="${ROOT_DIR}/results/structured"
OUT_RAW_DIR="${ROOT_DIR}/results/raw/setup"
DISCOVERY_OUT="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_meta.json"
SETUP_OUT="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_setup.json"
SETUP_LOG="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_setup.log"

mkdir -p "$OUT_STRUCT_DIR" "$OUT_RAW_DIR"

if [[ -e "$DISCOVERY_OUT" && "$SKIP_DISCOVERY" -eq 0 ]]; then
  echo "Discovery output already exists: $DISCOVERY_OUT" >&2
  echo "Pick a different --run-id or use --skip-discovery." >&2
  exit 1
fi

if [[ -e "$SETUP_OUT" ]]; then
  echo "Setup output already exists: $SETUP_OUT" >&2
  echo "Pick a different --run-id before re-running." >&2
  exit 1
fi

exec > >(tee -a "$SETUP_LOG") 2>&1

echo "== Cluster setup starting ($(date -Iseconds)) =="
echo "ROOT_DIR=$ROOT_DIR"
echo "RUN_ID=$RUN_ID"
echo "LABEL=$LABEL"
echo "VENV_DIR=$VENV_DIR"
echo "NCCL_TESTS_DIR=$NCCL_TESTS_DIR"
echo "TORCH_INDEX_URL=$TORCH_INDEX_URL"
echo "TORCH_VERSION=$TORCH_VERSION"
echo "TRANSFORMERS_VERSION=$TRANSFORMERS_VERSION"
echo "DATASETS_VERSION=$DATASETS_VERSION"
echo "SENTENCEPIECE_VERSION=$SENTENCEPIECE_VERSION"
echo "TOKENIZERS_VERSION=$TOKENIZERS_VERSION"
echo "HUGGINGFACE_HUB_VERSION=$HUGGINGFACE_HUB_VERSION"
echo "FSSPEC_VERSION=$FSSPEC_VERSION"

if [[ "$SKIP_DISCOVERY" -eq 0 ]]; then
  echo "== Discovery =="
  "${ROOT_DIR}/scripts/collect_system_info.sh" --output "$DISCOVERY_OUT" --label "$LABEL"
fi

APT_PACKAGES=(
  build-essential
  cmake
  ninja-build
  git
  pkg-config
  python3-venv
  python3-pip
  python3-dev
  numactl
  hwloc
  pciutils
  ethtool
  iproute2
  iputils-ping
  iperf3
  rdma-core
  infiniband-diags
  ibverbs-utils
  libibverbs-dev
  libnuma-dev
  libopenmpi-dev
  openmpi-bin
  libssl-dev
  libffi-dev
  patchelf
  curl
  wget
  libnccl2
  libnccl-dev
  cuda-toolkit-13-0
)

if [[ "$SKIP_APT" -eq 0 ]]; then
  echo "== Apt install =="
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
fi

CUDA_HOME="/usr/local/cuda"
if [[ ! -d "$CUDA_HOME" ]]; then
  echo "CUDA_HOME missing at $CUDA_HOME; nvcc install may have failed." >&2
  exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export AISP_CUDNN_RUNTIME_POLICY="${AISP_CUDNN_RUNTIME_POLICY:-auto}"

MPI_HOME=""
if command -v mpicc >/dev/null 2>&1; then
  MPI_INCDIRS="$(mpicc --showme:incdirs 2>/dev/null | awk '{print $1}')"
  if [[ -n "$MPI_INCDIRS" ]]; then
    MPI_HOME="$(dirname "$MPI_INCDIRS")"
  fi
fi
if [[ -z "$MPI_HOME" && -d /usr/lib/aarch64-linux-gnu/openmpi ]]; then
  MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
fi

if [[ "$SKIP_PYTHON" -eq 0 ]]; then
  echo "== Python venv + torch =="
  python3 -m venv "$VENV_DIR"
  VENV_PY="${VENV_DIR}/bin/python"
  "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
  "$VENV_DIR/bin/pip" install --index-url "$TORCH_INDEX_URL" "torch==${TORCH_VERSION}"
  echo "== Installing benchmark runtime dependencies =="
  "$VENV_DIR/bin/pip" install --no-cache-dir --upgrade \
    --extra-index-url "https://pypi.org/simple" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "datasets==${DATASETS_VERSION}" \
    "sentencepiece==${SENTENCEPIECE_VERSION}" \
    "tokenizers==${TOKENIZERS_VERSION}" \
    "huggingface-hub==${HUGGINGFACE_HUB_VERSION}" \
    "fsspec==${FSSPEC_VERSION}"

  if [[ "$INSTALL_VLLM" -eq 1 ]]; then
    echo "== vLLM install =="
    "$VENV_DIR/bin/pip" install vllm
    echo "== Restore CUDA-enabled torch (vLLM pins CPU torch on PyPI) =="
    "$VENV_DIR/bin/pip" install --index-url "$TORCH_INDEX_URL" --upgrade --force-reinstall --no-deps "torch==${TORCH_VERSION}"
  fi

  TORCH_LIB_DIR="$(resolve_torch_lib_dir "$VENV_PY")"
  if [[ -n "$TORCH_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${LD_LIBRARY_PATH:-}"
    echo "Using torch runtime libraries from: $TORCH_LIB_DIR"
  else
    echo "WARNING: torch/lib was not found under ${VENV_DIR}; setup sanity checks may use system runtime libs."
  fi

  echo "== Python sanity =="
  "$VENV_DIR/bin/python" - <<'PY'
import os
import torch
import transformers
import datasets
import tokenizers
import huggingface_hub
import sentencepiece
import fsspec
print("torch", torch.__version__)
print("torch.cuda.is_available", torch.cuda.is_available())
print("torch.version.cuda", torch.version.cuda)
print("torch.version.git", torch.version.git_version)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("tokenizers", tokenizers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("sentencepiece", sentencepiece.__version__)
print("fsspec", fsspec.__version__)
print("AISP_CUDNN_RUNTIME_POLICY", os.environ.get("AISP_CUDNN_RUNTIME_POLICY", ""))
if torch.cuda.is_available():
    print("torch.backends.cudnn.version", torch.backends.cudnn.version())
    nccl_v = torch.cuda.nccl.version()
    if nccl_v:
        print("torch.cuda.nccl.version", ".".join(str(x) for x in nccl_v))
    else:
        print("torch.cuda.nccl.version", "")
PY

  if [[ "$INSTALL_VLLM" -eq 1 ]]; then
    "$VENV_DIR/bin/python" - <<'PY'
import vllm
print("vllm", vllm.__version__)
PY
  fi
  echo "== pip check =="
  "$VENV_DIR/bin/pip" check || true
fi

TORCH_LIB_DIR="${TORCH_LIB_DIR:-$(resolve_torch_lib_dir "$VENV_PY")}"
RUNTIME_ENV_FILE="${VENV_DIR}/orig_parity_runtime_env.sh"
write_runtime_env_file "$RUNTIME_ENV_FILE" "$TORCH_LIB_DIR"
if [[ -n "$TORCH_LIB_DIR" ]]; then
  echo "Wrote runtime env helper: ${RUNTIME_ENV_FILE} (torch/lib=${TORCH_LIB_DIR})"
else
  echo "Wrote runtime env helper: ${RUNTIME_ENV_FILE} (torch/lib unresolved)"
fi

if [[ "$SKIP_NCCL_TESTS" -eq 0 ]]; then
  echo "== nccl-tests =="
  mkdir -p "$(dirname "$NCCL_TESTS_DIR")"
  if [[ ! -d "$NCCL_TESTS_DIR/.git" ]]; then
    git clone https://github.com/NVIDIA/nccl-tests.git "$NCCL_TESTS_DIR"
  fi
  if [[ -z "$MPI_HOME" ]]; then
    echo "MPI_HOME not detected; set --skip-nccl-tests or install OpenMPI dev headers." >&2
    exit 1
  fi
  make -C "$NCCL_TESTS_DIR" MPI=1 MPI_HOME="$MPI_HOME" NCCL_HOME=/usr CUDA_HOME="$CUDA_HOME"
fi

echo "== Writing setup metadata =="
python3 - <<'PY' "$SETUP_OUT" "$RUN_ID" "$LABEL" "$VENV_DIR" "$NCCL_TESTS_DIR"
import json
import os
import subprocess
import sys
import time

out_path, run_id, label, venv_dir, nccl_tests_dir = sys.argv[1:6]

def run(cmd: str):
    proc = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "cmd": cmd,
        "rc": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }

def run_lines(cmd: str):
    proc = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return [], proc.stderr.strip()
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()], ""

apt_packages = [
    "build-essential","cmake","ninja-build","git","pkg-config","python3-venv","python3-pip","python3-dev",
    "numactl","hwloc","pciutils","ethtool","iproute2","iputils-ping","iperf3","rdma-core","infiniband-diags",
    "ibverbs-utils","libibverbs-dev","libnuma-dev","libopenmpi-dev","openmpi-bin","libssl-dev","libffi-dev",
    "patchelf","curl","wget","libnccl2","libnccl-dev","cuda-toolkit-13-0",
]

versions = {}
lines, err = run_lines(
    "dpkg-query -W -f='${Package} ${Version}\\n' " + " ".join(apt_packages)
)
if err:
    versions["dpkg_query_error"] = err
else:
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            versions[parts[0]] = parts[1]

torch_info = {}
if os.path.exists(os.path.join(venv_dir, "bin", "python")):
    torch_info = run(
        f"{os.path.join(venv_dir, 'bin', 'python')} - <<'PY'\n"
        "import json\n"
        "try:\n"
        "    import torch\n"
        "    info = {\n"
        "        'version': torch.__version__,\n"
        "        'cuda_available': torch.cuda.is_available(),\n"
        "        'cuda_version': torch.version.cuda,\n"
        "        'git_version': torch.version.git_version,\n"
        "    }\n"
        "except Exception as exc:\n"
        "    info = {'error': str(exc)}\n"
        "print(json.dumps(info))\n"
        "PY"
    )

pip_freeze = []
pip_freeze_err = ""
if os.path.exists(os.path.join(venv_dir, "bin", "pip")):
    pip_freeze, pip_freeze_err = run_lines(f"{os.path.join(venv_dir, 'bin', 'pip')} freeze")

nccl_commit = ""
if os.path.exists(os.path.join(nccl_tests_dir, ".git")):
    nccl_commit = run(f"git -C {nccl_tests_dir} rev-parse HEAD")

payload = {
    "run_id": run_id,
    "label": label,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "apt_versions": versions,
    "torch_info_cmd": torch_info,
    "pip_freeze": pip_freeze,
    "pip_freeze_error": pip_freeze_err,
    "pip_check": run(f"{os.path.join(venv_dir, 'bin', 'pip')} check") if os.path.exists(os.path.join(venv_dir, 'bin', 'pip')) else {},
    "nvcc": run("nvcc --version"),
    "mpirun": run("mpirun --version"),
    "nvidia_smi": run("nvidia-smi"),
    "nccl_tests_commit": nccl_commit,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)

print(f"Wrote {out_path}")
PY

echo "== Setup complete ($(date -Iseconds)) =="
