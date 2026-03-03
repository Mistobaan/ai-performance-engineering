# NVFP4 Group GEMM Worklog

## Scope

This log captures only the final state and the key findings needed to continue tuning.
Historical branch-by-branch details were intentionally removed.

## Canonical Code Surface

- Lab folder: `labs/nvfp4_group_gemm`
- Single runtime module: `custom_cuda_submission.py`
- Single runtime API:
  - `prepare_custom_cuda(...)`
  - `custom_kernel_custom_cuda(...)`
- Single CUDA source: `custom_cuda_group_gemm_kernel.cu`
- Single baseline/optimized pair per case: `baseline_*_case{0..3}.py`, `optimized_*_case{0..3}.py`

## Final Runtime Defaults

Source: `labs/nvfp4_group_gemm/optimized_nvfp4_group_gemm_case{0,1,2,3}.py`

| Key | Value |
|---|---|
| `AISP_NVFP4_GROUP_GEMM_BLOCK_M` | `8` |
| `AISP_NVFP4_GROUP_GEMM_BLOCK_N` | `32` |
| `AISP_NVFP4_GROUP_GEMM_KPACK_TILE` | `64` |
| `AISP_NVFP4_GROUP_GEMM_UNROLL_N` | `2` |
| `AISP_NVFP4_GROUP_GEMM_WS_UNROLL2_MMA` | `0` |
| `AISP_NVFP4_GROUP_GEMM_WS_TMA_PRODUCER` | `1` |
| `AISP_NVFP4_GROUP_GEMM_EPILOGUE_LD_X32` | `1` |
| `AISP_NVFP4_GROUP_GEMM_FUSE_INPUTS` | `1` |
| `AISP_NVFP4_GROUP_GEMM_FUSE_INPUTS_COMPRESS_LIST` | `1` |
| `AISP_NVFP4_GROUP_GEMM_CTA_ORDER` | `tm_major` |
| `AISP_NVFP4_GROUP_GEMM_CLUSTER_DIM_X` | `1` |
| `AISP_NVFP4_GROUP_GEMM_ENABLE_TMA_MULTICAST` | `0` |
| `AISP_NVFP4_GROUP_GEMM_TMA_L2_PROMOTION` | `3` |
| `AISP_NVFP4_GROUP_GEMM_ASSUME_NO_N_TAIL` | `1` |

## Final Snapshot (B200 Expectations)

Source: `labs/nvfp4_group_gemm/expectations_b200.json`

| Example | best_optimized_time_ms |
|---|---:|
| `nvfp4_group_gemm_case0` | 1.8984307425 |
| `nvfp4_group_gemm_case1` | 0.1558052740 |
| `nvfp4_group_gemm_case2` | 0.5010162699 |
| `nvfp4_group_gemm_case3` | 0.4605501887 |

Computed from the four case times:
- geomean: `0.5111241797 ms` (`511.1241797 us`)

Latest per-case provenance timestamps in this snapshot:
- case0: `2026-03-01T15:11:28.664595`
- case1: `2026-02-13T16:29:43.667490`
- case2: `2026-03-01T15:12:24.351388`
- case3: `2026-02-13T13:39:15.497867`

## Key Findings Kept

- Keep one canonical NVFP4 lab target to reduce confusion and accidental regressions.
- Keep harness behavior generic (no lab-specific include/exclude logic in harness code paths).
- Promote changes only with correctness verification plus repeated A/B evidence.

## Repro

List targets:
```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm
```

Run canonical suite:
```bash
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile deep_dive --update-expectations
```
