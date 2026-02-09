# Cluster Perf Stack Profiles

This repository now uses explicit stack profiles for FP4/grouped-GEMM paths. The profile catalog lives in:

- `configs/cluster_perf_stack_profiles.json`

Canonical profile:

- `new_container` (digest-pinned `cfregly/cluster_perf` image)

## Profile Matrix

| Profile | Runtime | Image ref | Torch | CUDA | cuDNN | NCCL | DeepGEMM | Math policy |
|---|---|---|---|---|---|---|---|---|
| `old_container` | container | `ghcr.io/jordannanos/cmax-compute@sha256:d9ed224b30be7efff651d40801032c1f0117cab4012f143059af2c494a3e353e` | `2.10.0a0+a36e1d39eb.nv26.01.42222806` | `13.1` | `91701` | `2.29.2` | `2.3.0+0f5f266` | `allow_tf32=1`, `float32_matmul_precision=high` |
| `orig_parity_container` | container | explicit via `--image` (immutable repo@sha256 or local `sha256:<image_id>`) | `2.10.0a0+a36e1d39eb.nv26.01.42222806` | `13.1` | `91701` | `2.29.2` | `2.3.0+0f5f266` | `allow_tf32=1`, `float32_matmul_precision=high` |
| `new_container` | container | `cfregly/cluster_perf@sha256:f9b2f503384d1780206dda1435cc2fb4eebe43bb15ff4b040a3601356af63a42` | `2.9.1+cu130` | `13.0` | `91300` | `2.27.7` | `2.3.0+477618c` | `allow_tf32=1`, `float32_matmul_precision=high` |
| `host_only` | host | sourced from `cfregly/cluster_perf_orig_parity:latest` | `2.10.0a0+a36e1d39eb.nv26.01.42222806` | `13.1` | `91701` | `2.29.2` | `2.3.0+0f5f266` | `allow_tf32=1`, `float32_matmul_precision=high` |

## What Is Enforced

- `nvidia_peermem` must be loaded (strict preflight).
- Stack versions must match the selected profile.
- Host bootstrap installs host-only Torch/DeepGEMM/runtime libs from the orig-parity image and verifies expected versions.
- Container runs must use immutable image refs (repo `@sha256:...` or local `sha256:<image_id>`).
- Clock lock preflight must pass before benchmark execution.
- Benchmark math policy is explicitly set by environment (`CLUSTER_PERF_ALLOW_TF32`, `CLUSTER_PERF_FLOAT32_MATMUL_PRECISION`) rather than container defaults.
- Preflight validates effective applied math policy (`allow_tf32`, `cudnn_allow_tf32`, `float32_matmul_precision`) matches profile targets.
- For local image prototyping, preflight accepts immutable `sha256:<image_id>` in addition to repo digest refs.

Preflight artifacts are written alongside existing outputs as additional files:

- Grouped GEMM: `results/structured/<RUN_ID>_<LABEL>_cluster_perf_grouped_gemm_preflight_stack.json`
- Grouped GEMM: `results/structured/<RUN_ID>_<LABEL>_cluster_perf_grouped_gemm_preflight_clock_lock.json`
- FP4 smoke: `results/structured/<RUN_ID>_<LABEL>_cluster_perf_fp4_smoke_preflight_stack.json`
- FP4 smoke: `results/structured/<RUN_ID>_<LABEL>_cluster_perf_fp4_smoke_preflight_clock_lock.json`

Existing report artifact names are unchanged.

## Example Commands

Canonical decoupled container path:

```bash
scripts/run_cluster_perf_grouped_gemm.sh \
  --runtime container \
  --stack-profile new_container \
  --run-id <run_id> \
  --label <label>
```

Legacy comparison path:

```bash
scripts/run_cluster_perf_grouped_gemm.sh \
  --runtime container \
  --stack-profile old_container \
  --run-id <run_id> \
  --label <label>
```

Open orig-parity path (build local image first, then pass immutable image id):

```bash
scripts/repro/build_cluster_perf_image.sh --profile orig_parity
IMAGE_ID="$(docker image inspect --format '{{.Id}}' cfregly/cluster_perf_orig_parity:latest)"

scripts/run_cluster_perf_grouped_gemm.sh \
  --runtime container \
  --stack-profile orig_parity_container \
  --image "${IMAGE_ID}" \
  --run-id <run_id> \
  --label <label>
```

Host-only fallback:

```bash
scripts/run_cluster_perf_grouped_gemm.sh \
  --runtime host \
  --stack-profile host_only \
  --run-id <run_id> \
  --label <label>
```
