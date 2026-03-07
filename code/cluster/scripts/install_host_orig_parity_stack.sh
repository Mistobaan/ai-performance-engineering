#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/install_host_orig_parity_stack.sh --venv-dir <path> [options]

Install host runtime stack from the orig-parity container image so host-only
FP4 checks use the same Torch/DeepGEMM binary provenance as orig_parity.

Options:
  --venv-dir <path>              Python venv directory (required)
  --requirements-file <path>     Optional requirements file to install first
  --parity-image <ref>           Source image (default: cluster_perf_orig_parity:latest)
  --expected-torch-version <v>   Expected torch.__version__
  --expected-cuda-version <v>    Expected torch.version.cuda
  --expected-cudnn-version <v>   Expected torch.backends.cudnn.version()
  --expected-nccl-version <v>    Expected torch.cuda.nccl.version()
  --expected-deep-gemm-version <v>  Expected deep_gemm package version
EOF
}

VENV_DIR=""
REQUIREMENTS_FILE=""
PARITY_IMAGE="${PARITY_IMAGE:-cluster_perf_orig_parity:latest}"
EXPECTED_TORCH_VERSION="2.10.0a0+a36e1d39eb.nv26.01.42222806"
EXPECTED_CUDA_VERSION="13.1"
EXPECTED_CUDNN_VERSION="91701"
EXPECTED_NCCL_VERSION="2.29.2"
EXPECTED_DEEP_GEMM_VERSION="2.3.0+0f5f266"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --venv-dir) VENV_DIR="${2:-}"; shift 2 ;;
    --requirements-file) REQUIREMENTS_FILE="${2:-}"; shift 2 ;;
    --parity-image) PARITY_IMAGE="${2:-}"; shift 2 ;;
    --expected-torch-version) EXPECTED_TORCH_VERSION="${2:-}"; shift 2 ;;
    --expected-cuda-version) EXPECTED_CUDA_VERSION="${2:-}"; shift 2 ;;
    --expected-cudnn-version) EXPECTED_CUDNN_VERSION="${2:-}"; shift 2 ;;
    --expected-nccl-version) EXPECTED_NCCL_VERSION="${2:-}"; shift 2 ;;
    --expected-deep-gemm-version) EXPECTED_DEEP_GEMM_VERSION="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$VENV_DIR" ]]; then
  echo "ERROR: --venv-dir is required." >&2
  usage >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is required to install host orig-parity stack." >&2
  exit 2
fi

mkdir -p "$VENV_DIR"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

"$VENV_PIP" install --upgrade pip "setuptools<81" wheel
if [[ -n "$REQUIREMENTS_FILE" && -f "$REQUIREMENTS_FILE" ]]; then
  "$VENV_PIP" install -r "$REQUIREMENTS_FILE"
fi

if ! docker image inspect "$PARITY_IMAGE" >/dev/null 2>&1; then
  echo "INFO: pulling parity image: ${PARITY_IMAGE}" >&2
  docker pull "$PARITY_IMAGE" >/dev/null
fi

META_JSON="$(docker run -i --rm --entrypoint python "$PARITY_IMAGE" - <<'PY'
import glob
import json
import os
import pathlib
import importlib.metadata as im
import torch

site_dir = pathlib.Path(torch.__file__).resolve().parent.parent
paths = []
for rel in ("torch", "functorch", "triton", "torchgen", "deep_gemm"):
    if (site_dir / rel).exists():
        paths.append(rel)

for pkg in ("torch", "functorch", "triton", "deep_gemm"):
    try:
        version = im.version(pkg)
    except Exception:
        continue
    dist_name = f"{pkg}-{version}.dist-info"
    if (site_dir / dist_name).exists():
        paths.append(dist_name)

lib_patterns = (
    "/usr/local/cuda/lib64/libcudnn*.so*",
    "/usr/local/cuda/targets/*/lib/libcudnn*.so*",
    "/usr/lib/*/libcudnn*.so*",
    "/usr/lib/*/*/libcudnn*.so*",
    "/usr/lib/*/libnvpl_*.so*",
    "/usr/lib/*/*/libnvpl_*.so*",
    "/usr/local/lib/libnvpl_*.so*",
    "/usr/lib/*/libucc.so*",
    "/usr/lib/*/*/libucc.so*",
    "/usr/local/lib/libucc.so*",
    "/opt/**/lib/libucc.so*",
    "/usr/local/lib/libuc*.so*",
    "/usr/lib/*/libuc*.so*",
    "/usr/lib/*/*/libuc*.so*",
    "/opt/**/lib/libuc*.so*",
    "/usr/local/lib/libmkl*.so*",
    "/usr/local/lib/libiomp*.so*",
    "/usr/lib/*/libmkl*.so*",
    "/usr/lib/*/*/libmkl*.so*",
    "/usr/lib/*/libiomp*.so*",
    "/usr/lib/*/*/libiomp*.so*",
    "/opt/**/lib/libmkl*.so*",
    "/opt/**/lib/libiomp*.so*",
)
runtime_libs = set()
for pattern in lib_patterns:
    for path in glob.glob(pattern, recursive=True):
        if not os.path.exists(path):
            continue
        real_path = os.path.realpath(path)
        if os.path.isfile(real_path):
            runtime_libs.add(real_path)

payload = {
    "site": str(site_dir),
    "paths": sorted(set(paths)),
    "runtime_libs": sorted(runtime_libs),
}
print(json.dumps(payload, sort_keys=True))
PY
)"

CONTAINER_SITE="$(python3 - <<'PY' "$META_JSON"
import json
import sys

print((json.loads(sys.argv[1]) or {}).get("site", ""))
PY
)"

mapfile -t COPY_PATHS < <(python3 - <<'PY' "$META_JSON"
import json
import sys

for entry in (json.loads(sys.argv[1]) or {}).get("paths", []):
    print(str(entry))
PY
)

mapfile -t RUNTIME_LIBS < <(python3 - <<'PY' "$META_JSON"
import json
import sys

for entry in (json.loads(sys.argv[1]) or {}).get("runtime_libs", []):
    print(str(entry))
PY
)

if [[ -z "$CONTAINER_SITE" ]]; then
  echo "ERROR: failed to determine container site-packages path from ${PARITY_IMAGE}." >&2
  exit 2
fi
if [[ "${#COPY_PATHS[@]}" -eq 0 ]]; then
  echo "ERROR: no package paths discovered in parity image ${PARITY_IMAGE}." >&2
  exit 2
fi

HOST_SITE="$("$VENV_PY" - <<'PY'
import site

candidates = []
for base in site.getsitepackages():
    if "site-packages" in base:
        candidates.append(base)
if not candidates:
    candidates = site.getsitepackages()
if not candidates:
    raise SystemExit("unable to resolve host site-packages path")
print(candidates[0])
PY
)"

if [[ -z "$HOST_SITE" ]]; then
  echo "ERROR: failed to determine host site-packages for ${VENV_DIR}." >&2
  exit 2
fi
mkdir -p "$HOST_SITE"

TMP_DIR="$(mktemp -d)"
CID=""
cleanup() {
  if [[ -n "$CID" ]]; then
    docker rm -f "$CID" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

CID="$(docker create "$PARITY_IMAGE")"

for pattern in \
  torch \
  torch-*.dist-info \
  functorch \
  functorch-*.dist-info \
  triton \
  triton-*.dist-info \
  torchgen \
  deep_gemm \
  deep_gemm-*.dist-info; do
  for existing in "$HOST_SITE"/$pattern; do
    [[ -e "$existing" ]] || continue
    rm -rf "$existing"
  done
done

for rel_path in "${COPY_PATHS[@]}"; do
  docker cp "${CID}:${CONTAINER_SITE}/${rel_path}" "${TMP_DIR}/"
done
cp -a "${TMP_DIR}/." "${HOST_SITE}/"
rm -rf "${TMP_DIR:?}/"*

TORCH_LIB_DIR="${HOST_SITE}/torch/lib"
if [[ ! -d "$TORCH_LIB_DIR" ]]; then
  echo "ERROR: expected torch library dir missing after parity copy: ${TORCH_LIB_DIR}" >&2
  exit 2
fi

for pattern in "libcudnn*.so*" "libnvpl_*.so*" "libucc.so*" "libuc*.so*" "libmkl*.so*" "libiomp*.so*"; do
  for existing in "${TORCH_LIB_DIR}/"${pattern}; do
    [[ -e "$existing" ]] || continue
    rm -f "$existing"
  done
done

for lib_abs in "${RUNTIME_LIBS[@]}"; do
  docker cp "${CID}:${lib_abs}" "${TORCH_LIB_DIR}/"
done

python3 - <<'PY' "$TORCH_LIB_DIR"
import os
import pathlib
import re
import sys

lib_dir = pathlib.Path(sys.argv[1])
prefix_re = re.compile(r"^(lib(?:cudnn[^.]*|ucc|uc[spmt][^.]*|nvpl_[^.]+|mkl[^.]*|iomp[^.]*))\.so\.(\d+)(?:\..+)?$")
for path in sorted(lib_dir.glob("lib*.so.*")):
    if path.is_symlink():
        continue
    m = prefix_re.match(path.name)
    if not m:
        continue
    prefix = m.group(1)
    major = m.group(2)
    major_link = lib_dir / f"{prefix}.so.{major}"
    bare_link = lib_dir / f"{prefix}.so"
    bare_target = path.name
    if path.name != major_link.name:
        if major_link.exists() or major_link.is_symlink():
            major_link.unlink()
        os.symlink(path.name, major_link)
        bare_target = major_link.name
    if bare_link.exists() or bare_link.is_symlink():
        bare_link.unlink()
    os.symlink(bare_target, bare_link)
PY

TMP_NVBW="${TMP_DIR}/nvbandwidth"
docker cp "${CID}:/usr/local/bin/nvbandwidth" "${TMP_NVBW}"
chmod 0755 "${TMP_NVBW}"
if [[ -w /usr/local/bin ]]; then
  install -m 0755 "${TMP_NVBW}" /usr/local/bin/nvbandwidth
else
  sudo install -m 0755 "${TMP_NVBW}" /usr/local/bin/nvbandwidth
fi

RUNTIME_ENV_FILE="${VENV_DIR}/orig_parity_runtime_env.sh"
cat >"${RUNTIME_ENV_FILE}" <<EOF
#!/usr/bin/env bash
if [[ -d "${TORCH_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${TORCH_LIB_DIR}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
fi
EOF
chmod 0755 "${RUNTIME_ENV_FILE}"

VERIFY_JSON="$(LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}" "$VENV_PY" - <<'PY'
import importlib.metadata as im
import json
import torch

payload = {
    "torch_version": str(torch.__version__),
    "cuda_version": str(torch.version.cuda),
    "cudnn_version": str(torch.backends.cudnn.version()),
    "nccl_version": ".".join(str(x) for x in (torch.cuda.nccl.version() or ())),
    "deep_gemm_version": str(im.version("deep_gemm")),
}
print(json.dumps(payload, sort_keys=True))
PY
)"

python3 - <<'PY' \
  "$VERIFY_JSON" \
  "$EXPECTED_TORCH_VERSION" \
  "$EXPECTED_CUDA_VERSION" \
  "$EXPECTED_CUDNN_VERSION" \
  "$EXPECTED_NCCL_VERSION" \
  "$EXPECTED_DEEP_GEMM_VERSION"
import json
import sys

payload = json.loads(sys.argv[1])
expected = {
    "torch_version": sys.argv[2],
    "cuda_version": sys.argv[3],
    "cudnn_version": sys.argv[4],
    "nccl_version": sys.argv[5],
    "deep_gemm_version": sys.argv[6],
}

errors = []
for key, want in expected.items():
    got = str(payload.get(key, ""))
    if want and not got.startswith(want):
        errors.append(f"{key}: expected prefix {want!r}, got {got!r}")

if errors:
    raise SystemExit("stack verification failed: " + "; ".join(errors))

print("host-orig-parity verification passed:", json.dumps(payload, sort_keys=True))
PY

echo "Installed host orig-parity stack from ${PARITY_IMAGE}"
echo "  venv: ${VENV_DIR}"
echo "  torch_lib_dir: ${TORCH_LIB_DIR}"
echo "  runtime_env: ${RUNTIME_ENV_FILE}"
echo "  nvbandwidth: /usr/local/bin/nvbandwidth"
