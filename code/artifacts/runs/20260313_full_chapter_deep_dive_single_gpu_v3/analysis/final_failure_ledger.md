# Final Deep-Dive Failure Ledger

- Original run: `20260313_full_chapter_deep_dive_single_gpu_v3`
- Rechecks: `20260314_deep_dive_failed_targets_recheck_v6_clean, 20260314_deep_dive_failed_targets_recheck_v7_regional_compilation, 20260314_deep_dive_failed_targets_recheck_v13_clean, 20260314_deep_dive_dsmem_v3_recheck_v14, 20260314_deep_dive_dsmem_v3_recheck_v15_direct`
- Total original failures: `18`
- Resolved: `18`
- Unresolved: `0`

| Target | Original | Latest | Latest Run | Best Speedup | Resolved |
| --- | --- | --- | --- | --- | --- |
| `ch09:compute_bound` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `1.714x` | `True` |
| `ch09:cublas_gemm_fp4_perchannel` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v6_clean` | `1.141x` | `True` |
| `ch10:dsmem_reduction` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v6_clean` | `1.549x` | `True` |
| `ch10:dsmem_reduction_v3` | `failed_profiler` | `succeeded` | `20260314_deep_dive_dsmem_v3_recheck_v15_direct` | `1.851x` | `True` |
| `ch10:dsmem_reduction_warp_specialized` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v6_clean` | `2.035x` | `True` |
| `ch12:cuda_graphs` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `3.490x` | `True` |
| `ch12:cuda_graphs_router` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `3.317x` | `True` |
| `ch12:graph_bandwidth` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `2.465x` | `True` |
| `ch12:graph_conditional_runtime` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `3.002x` | `True` |
| `ch12:kernel_launches` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `2.977x` | `True` |
| `ch13:autograd_standard` | `failed_verification` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `6.315x` | `True` |
| `ch13:fp8_static` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `1.257x` | `True` |
| `ch13:matmul_pytorch` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `1.478x` | `True` |
| `ch13:memory_profiling` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `2.777x` | `True` |
| `ch13:precisionfp8_te` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `5.261x` | `True` |
| `ch16:regional_compilation` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v7_regional_compilation` | `1.113x` | `True` |
| `ch17:memory` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `12.216x` | `True` |
| `ch20:end_to_end_bandwidth` | `failed_profiler` | `succeeded` | `20260314_deep_dive_failed_targets_recheck_v13_clean` | `4.508x` | `True` |
