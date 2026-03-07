#!/usr/bin/env bash
set -euo pipefail

OUTPUT=""
LABEL=""
VENV_PY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --venv-py)
      VENV_PY="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  echo "Usage: $0 --output <path> [--label <node_label>] [--venv-py <path>]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_host_runtime_env.sh
source "${ROOT_DIR}/scripts/lib_host_runtime_env.sh"
source_host_runtime_env_if_present "$ROOT_DIR"
if [[ -z "$VENV_PY" ]]; then
  VENV_PY="${ROOT_DIR}/env/venv/bin/python"
fi

python3 - <<'PY' "$OUTPUT" "$LABEL" "$VENV_PY"
import json
import os
import subprocess
import sys
import time

out_path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else ""
venv_py = sys.argv[3] if len(sys.argv) > 3 else ""

def run(cmd: str):
    proc = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "cmd": cmd,
        "rc": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }

results = {
    "label": label,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "commands": {},
}

# System tools.
results["commands"]["date"] = run("date -Iseconds")
results["commands"]["hostname"] = run("hostname")
results["commands"]["uname"] = run("uname -a")
results["commands"]["os_release"] = run("cat /etc/os-release")

results["commands"]["python3_version"] = run("python3 --version")
results["commands"]["mpirun_version"] = run("mpirun --version")
results["commands"]["ompi_info_version"] = run("ompi_info --version")

results["commands"]["nvcc_version"] = run("nvcc --version")
results["commands"]["nvidia_smi"] = run("nvidia-smi")

# System NCCL packages (if installed).
results["commands"]["dpkg_nccl"] = run(
    "dpkg-query -W -f='${Package} ${Version}\\n' libnccl2 libnccl-dev 2>/dev/null || true"
)

# venv torch + NCCL info (if venv exists).
if venv_py and os.path.exists(venv_py) and os.access(venv_py, os.X_OK):
    results["commands"]["venv_python_version"] = run(f"{venv_py} --version")
    results["commands"]["venv_pip_show_torch"] = run(
        f"{os.path.dirname(venv_py)}/pip show torch"
    )
    results["commands"]["venv_torch_info"] = run(
        f"{venv_py} - <<'EOF'\n"
        "import json\n"
        "import torch\n"
        "payload = {\n"
        "  'torch': torch.__version__,\n"
        "  'cuda': torch.version.cuda,\n"
        "  'git': getattr(torch.version, 'git_version', None),\n"
        "  'nccl': None,\n"
        "}\n"
        "try:\n"
        "  payload['nccl'] = torch.cuda.nccl.version()\n"
        "except Exception as exc:\n"
        "  payload['nccl'] = f'<error: {exc}>'\n"
        "print(json.dumps(payload, sort_keys=True))\n"
        "EOF"
    )

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, sort_keys=True)

print(f"Wrote {out_path}")
PY
