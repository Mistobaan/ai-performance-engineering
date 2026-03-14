# Deep-Dive Expectation Refresh Summary

- Original deep-dive run: `20260313_full_chapter_deep_dive_single_gpu_v3`
- Failure rechecks: `20260314_deep_dive_failed_targets_recheck_v6_clean, 20260314_deep_dive_failed_targets_recheck_v7_regional_compilation, 20260314_deep_dive_failed_targets_recheck_v13_clean, 20260314_deep_dive_dsmem_v3_recheck_v14, 20260314_deep_dive_dsmem_v3_recheck_v15_direct`
- Approved expectation refresh targets: `29`
- Applied records: `29`
- Applied counts: `{'updated': 25, 'unchanged': 1, 'improved': 1, 'regressed': 2}`

## Source Runs

- `artifacts/runs/20260313_full_chapter_deep_dive_single_gpu_v3/results/benchmark_test_results.json` -> counts `{'applied': 11, 'improved': 0, 'regressed': 1, 'unchanged': 0, 'updated': 10, 'rejected': 0, 'skipped': 0, 'total': 11}`
- `artifacts/runs/20260314_deep_dive_failed_targets_recheck_v6_clean/results/benchmark_test_results.json` -> counts `{'applied': 3, 'improved': 0, 'regressed': 0, 'unchanged': 1, 'updated': 2, 'rejected': 0, 'skipped': 0, 'total': 3}`
- `artifacts/runs/20260314_deep_dive_failed_targets_recheck_v7_regional_compilation/results/benchmark_test_results.json` -> counts `{'applied': 1, 'improved': 0, 'regressed': 0, 'unchanged': 0, 'updated': 1, 'rejected': 0, 'skipped': 0, 'total': 1}`
- `artifacts/runs/20260314_deep_dive_failed_targets_recheck_v13_clean/results/benchmark_test_results.json` -> counts `{'applied': 13, 'improved': 1, 'regressed': 1, 'unchanged': 0, 'updated': 11, 'rejected': 0, 'skipped': 0, 'total': 13}`
- `artifacts/runs/20260314_deep_dive_dsmem_v3_recheck_v15_direct/results/benchmark_test_results.json` -> counts `{'applied': 1, 'improved': 0, 'regressed': 0, 'unchanged': 0, 'updated': 1, 'rejected': 0, 'skipped': 0, 'total': 1}`

## Applied Records

| Target | File | Status | Message |
| --- | --- | --- | --- |
| `ch08:tiling_tcgen05` | `ch08/expectations_b200.json` | `updated` | Entry tiling_tcgen05 updated (speed) score 1.153 |
| `ch09:compute_bound` | `ch09/expectations_b200.json` | `updated` | Entry compute_bound updated (speed) score 1.714 |
| `ch09:cublas_gemm_fp4_perchannel` | `ch09/expectations_b200.json` | `updated` | Entry cublas_gemm_fp4_perchannel_cuda updated (speed) score 1.141 |
| `ch09:cutlass_gemm_fp4_perchannel` | `ch09/expectations_b200.json` | `updated` | Entry cutlass_gemm_fp4_perchannel_cuda updated (speed) score 1.126 |
| `ch10:cooperative_persistent` | `ch10/expectations_b200.json` | `updated` | Entry cooperative_persistent_cuda updated (speed) score 1.107 |
| `ch10:dsmem_reduction` | `ch10/expectations_b200.json` | `updated` | Entry dsmem_reduction_cuda updated (speed) score 1.549 |
| `ch10:dsmem_reduction_v3` | `ch10/expectations_b200.json` | `updated` | Entry dsmem_reduction_v3_cuda updated (speed) score 1.851 |
| `ch10:dsmem_reduction_warp_specialized` | `ch10/expectations_b200.json` | `unchanged` | Entry dsmem_reduction_warp_specialized_cuda unchanged (speed) score 2.035 |
| `ch10:matmul_tcgen05_epilogue` | `ch10/expectations_b200.json` | `updated` | Entry matmul_tcgen05_epilogue updated (speed) score 1.148 |
| `ch10:persistent_matmul_tma` | `ch10/expectations_b200.json` | `updated` | Entry persistent_matmul_tma updated (speed) score 1.125 |
| `ch10:tcgen05_cluster_pipeline` | `ch10/expectations_b200.json` | `updated` | Entry tcgen05_cluster_pipeline updated (speed) score 1.136 |
| `ch10:warp_specialized_cluster_pipeline` | `ch10/expectations_b200.json` | `updated` | Entry warp_specialized_cluster_pipeline_cuda updated (speed) score 1.130 |
| `ch12:cuda_graphs` | `ch12/expectations_b200.json` | `updated` | Entry cuda_graphs updated (speed) score 3.490 |
| `ch12:cuda_graphs_router` | `ch12/expectations_b200.json` | `updated` | Entry cuda_graphs_router updated (speed) score 3.317 |
| `ch12:graph_bandwidth` | `ch12/expectations_b200.json` | `updated` | Entry graph_bandwidth updated (speed) score 2.465 |
| `ch12:graph_conditional_runtime` | `ch12/expectations_b200.json` | `updated` | Entry graph_conditional_runtime updated (speed) score 3.002 |
| `ch12:kernel_launches` | `ch12/expectations_b200.json` | `updated` | Entry kernel_launches updated (speed) score 2.977 |
| `ch13:autograd_standard` | `ch13/expectations_b200.json` | `improved` | Entry autograd_standard improved (speed) score 6.315 |
| `ch13:fp8_static` | `ch13/expectations_b200.json` | `updated` | Entry fp8_static updated (speed) score 1.257 |
| `ch13:kv_cache_naive_pool` | `ch13/expectations_b200.json` | `regressed` | Entry kv_cache_naive_pool regressed (speed) score 1.139 |
| `ch13:matmul_pytorch` | `ch13/expectations_b200.json` | `updated` | Entry matmul_pytorch updated (speed) score 1.478 |
| `ch13:memory_profiling` | `ch13/expectations_b200.json` | `regressed` | Entry memory_profiling regressed (speed) score 2.777 |
| `ch13:precisionfp8_te` | `ch13/expectations_b200.json` | `updated` | Entry precisionfp8_te updated (speed) score 5.261 |
| `ch15:disaggregated_inference` | `ch15/expectations_b200.json` | `updated` | Entry disaggregated_inference updated (speed) score 1.081 |
| `ch15:moe_overlap` | `ch15/expectations_b200.json` | `updated` | Entry moe_overlap updated (speed) score 1.093 |
| `ch15:moe_overlap_shared_expert` | `ch15/expectations_b200.json` | `updated` | Entry moe_overlap_shared_expert updated (speed) score 1.091 |
| `ch16:regional_compilation` | `ch16/expectations_b200.json` | `updated` | Entry regional_compilation updated (speed) score 1.113 |
| `ch17:memory` | `ch17/expectations_b200.json` | `updated` | Entry memory updated (speed) score 12.216 |
| `ch20:end_to_end_bandwidth` | `ch20/expectations_b200.json` | `updated` | Entry end_to_end_bandwidth updated (speed) score 4.508 |
