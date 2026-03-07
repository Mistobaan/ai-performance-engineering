#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Build local open container image for FP4 + nvbandwidth workflows.

Usage:
  scripts/repro/build_cluster_perf_image.sh [options]

Options:
  --profile <name>       Build profile: open|orig_parity (default: open)
  --tag <image:tag>      Output image tag (default depends on --profile)
  --dockerfile <path>    Dockerfile path (default depends on --profile)
  --pull                 Pull newer base layers before build
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="open"
TAG=""
DOCKERFILE=""
TAG_SET=0
DOCKERFILE_SET=0
PULL=0

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --dockerfile) DOCKERFILE="${2:-}"; DOCKERFILE_SET=1; shift 2 ;;
    --tag) TAG="${2:-}"; TAG_SET=1; shift 2 ;;
    --pull) PULL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${1}" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$PROFILE" != "open" && "$PROFILE" != "orig_parity" ]]; then
  echo "ERROR: unsupported --profile: ${PROFILE} (expected open|orig_parity)" >&2
  exit 2
fi
if [[ "$PROFILE" == "open" ]]; then
  if [[ "$TAG_SET" -eq 0 ]]; then
    TAG="cluster_perf:latest"
  fi
  if [[ "$DOCKERFILE_SET" -eq 0 ]]; then
    DOCKERFILE="${ROOT_DIR}/docker/cluster_perf.Dockerfile"
  fi
else
  if [[ "$TAG_SET" -eq 0 ]]; then
    TAG="cluster_perf_orig_parity:latest"
  fi
  if [[ "$DOCKERFILE_SET" -eq 0 ]]; then
    DOCKERFILE="${ROOT_DIR}/docker/cluster_perf_orig_parity.Dockerfile"
  fi
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found." >&2
  exit 1
fi
if [[ ! -f "$DOCKERFILE" ]]; then
  echo "ERROR: Dockerfile not found: ${DOCKERFILE}" >&2
  exit 1
fi

BUILD_ARGS=(docker build -t "$TAG" -f "$DOCKERFILE")
if [[ "$PULL" -eq 1 ]]; then
  BUILD_ARGS+=(--pull)
fi
BUILD_ARGS+=("$ROOT_DIR")

echo "Building ${TAG}"
echo "Profile: ${PROFILE}"
echo "Dockerfile: ${DOCKERFILE}"
"${BUILD_ARGS[@]}"

echo "Built image: ${TAG}"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${TAG}" 2>/dev/null || true)"
IMAGE_DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "${TAG}" 2>/dev/null || true)"
if [[ -n "${IMAGE_ID}" ]]; then
  echo "Image ID: ${IMAGE_ID}"
fi
if [[ -n "${IMAGE_DIGEST}" ]]; then
  echo "Pinned ref: ${IMAGE_DIGEST}"
fi
