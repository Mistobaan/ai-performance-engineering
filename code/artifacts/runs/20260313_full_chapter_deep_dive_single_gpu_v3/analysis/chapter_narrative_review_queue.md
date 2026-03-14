# Chapter Narrative Review Queue

- Original deep-dive run: `20260313_full_chapter_deep_dive_single_gpu_v3`
- Counts: `{'reframe': 16, 'qualify': 11, 'goal_specific': 1, 'total': 28}`

## ch01

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch01:nvfp4_mlp` | `reframe` | `hold` | `hold_expectations_family_investigation` | All chapters use the same shared NVFP4 MLP benchmark shape and the same Transformer Engine NVFP4 path. On this environment the optimized path shows near-identical kernel metrics and no durable end-to-end win over the BF16 baseline, so the family is weak by construction rather than by one chapter-specific bug. |

## ch04

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch04:symmetric_memory_perf` | `reframe` | `hold` | `hold_expectations_reframe_narrative` | Do not refresh expectations upward or present this as a strong win; either improve the benchmark or demote the narrative. |

## ch05

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch05:ai` | `reframe` | `hold` | `hold_expectations_reframe_narrative` | Do not refresh expectations upward or present this as a strong win; either improve the benchmark or demote the narrative. |
| `ch05:distributed` | `reframe` | `hold` | `hold_expectations_hardware_or_capability_gated` | The optimized path is intentionally multi-GPU/distributed and is skipped on this 1-GPU host, while the baseline still runs as a single-GPU host-staged reduction. This is not an optimization miss; it is a scope/hardware mismatch for the current environment. |

## ch08

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch08:nvfp4_mlp` | `reframe` | `hold` | `hold_expectations_family_investigation` | All chapters use the same shared NVFP4 MLP benchmark shape and the same Transformer Engine NVFP4 path. On this environment the optimized path shows near-identical kernel metrics and no durable end-to-end win over the BF16 baseline, so the family is weak by construction rather than by one chapter-specific bug. |
| `ch08:tiling_tcgen05` | `qualify` | `refresh` | `refresh_with_qualified_narrative` | Keep the example, but describe it as a small or context-dependent win rather than a headline optimization. |

## ch09

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch09:cublaslt_gemm_fp4` | `reframe` | `hold` | `hold_expectations_hardware_or_capability_gated` | The optimized cuBLASLt NVFP4 algorithm is unavailable on the current driver/toolchain stack, so no optimized winner is produced. |
| `ch09:cutlass_gemm_fp4_perchannel` | `qualify` | `refresh` | `refresh_with_qualified_narrative` | Keep the example, but describe it as a small or context-dependent win rather than a headline optimization. |

## ch10

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch10:cooperative_persistent` | `qualify` | `refresh` | `refresh_with_qualified_narrative` | Keep the example, but describe it as a small or context-dependent win rather than a headline optimization. |
| `ch10:matmul_tcgen05_epilogue` | `qualify` | `refresh` | `refresh_with_qualified_narrative` | Keep the example, but describe it as a small or context-dependent win rather than a headline optimization. |
| `ch10:persistent_matmul_tma` | `qualify` | `refresh` | `refresh_with_qualified_narrative` | Keep the example, but describe it as a small or context-dependent win rather than a headline optimization. |
| `ch10:tcgen05_cluster_pipeline` | `qualify` | `refresh` | `refresh_with_qualified_narrative` | Keep the example, but describe it as a small or context-dependent win rather than a headline optimization. |
| `ch10:tcgen05_warp_specialization_cutlass` | `reframe` | `hold` | `hold_expectations_reframe_narrative` | Do not refresh expectations upward or present this as a strong win; either improve the benchmark or demote the narrative. |
| `ch10:warp_specialized_cluster_pipeline` | `qualify` | `refresh` | `refresh_with_qualified_narrative` | Keep the example, but describe it as a small or context-dependent win rather than a headline optimization. |

## ch11

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch11:tensor_cores_streams` | `reframe` | `hold` | `hold_expectations_reframe_narrative` | Do not refresh expectations upward or present this as a strong win; either improve the benchmark or demote the narrative. |

## ch12

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch12:nvfp4_mlp` | `reframe` | `hold` | `hold_expectations_family_investigation` | All chapters use the same shared NVFP4 MLP benchmark shape and the same Transformer Engine NVFP4 path. On this environment the optimized path shows near-identical kernel metrics and no durable end-to-end win over the BF16 baseline, so the family is weak by construction rather than by one chapter-specific bug. |

## ch13

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch13:kv_cache_naive_pool` | `qualify` | `refresh` | `refresh_with_qualified_narrative` | Keep the example, but describe it as a small or context-dependent win rather than a headline optimization. |
| `ch13:training_standard` | `goal_specific` | `goal_specific` | `evaluate_non_speed_goal` | Do not score this example as a speed story; evaluate it against its declared non-speed goal. |

## ch15

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch15:allreduce_rmsnorm` | `reframe` | `hold` | `hold_expectations_reframe_narrative` | Do not refresh expectations upward or present this as a strong win; either improve the benchmark or demote the narrative. |
| `ch15:disaggregated_inference` | `qualify` | `refresh` | `refresh_with_qualified_narrative` | Keep the example, but describe it as a small or context-dependent win rather than a headline optimization. |
| `ch15:moe_overlap` | `qualify` | `refresh` | `refresh_with_qualified_narrative` | Keep the example, but describe it as a small or context-dependent win rather than a headline optimization. |
| `ch15:moe_overlap_shared_expert` | `qualify` | `refresh` | `refresh_with_qualified_narrative` | Keep the example, but describe it as a small or context-dependent win rather than a headline optimization. |
| `ch15:nvfp4_mlp` | `reframe` | `hold` | `hold_expectations_family_investigation` | All chapters use the same shared NVFP4 MLP benchmark shape and the same Transformer Engine NVFP4 path. On this environment the optimized path shows near-identical kernel metrics and no durable end-to-end win over the BF16 baseline, so the family is weak by construction rather than by one chapter-specific bug. |

## ch16

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch16:nvfp4_mlp` | `reframe` | `hold` | `hold_expectations_family_investigation` | All chapters use the same shared NVFP4 MLP benchmark shape and the same Transformer Engine NVFP4 path. On this environment the optimized path shows near-identical kernel metrics and no durable end-to-end win over the BF16 baseline, so the family is weak by construction rather than by one chapter-specific bug. |

## ch17

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch17:nvfp4_mlp` | `reframe` | `hold` | `hold_expectations_family_investigation` | All chapters use the same shared NVFP4 MLP benchmark shape and the same Transformer Engine NVFP4 path. On this environment the optimized path shows near-identical kernel metrics and no durable end-to-end win over the BF16 baseline, so the family is weak by construction rather than by one chapter-specific bug. |
| `ch17:prefill_decode_disagg_tpot_long` | `reframe` | `hold` | `hold_expectations_reframe_narrative` | Do not refresh expectations upward or present this as a strong win; either improve the benchmark or demote the narrative. |

## ch19

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch19:vectorization_memory` | `reframe` | `hold` | `hold_expectations_reframe_narrative` | Do not refresh expectations upward or present this as a strong win; either improve the benchmark or demote the narrative. |

## ch20

| Target | Narrative | Expectations | Disposition | Note |
| --- | --- | --- | --- | --- |
| `ch20:nvfp4_mlp` | `reframe` | `hold` | `hold_expectations_family_investigation` | All chapters use the same shared NVFP4 MLP benchmark shape and the same Transformer Engine NVFP4 path. On this environment the optimized path shows near-identical kernel metrics and no durable end-to-end win over the BF16 baseline, so the family is weak by construction rather than by one chapter-specific bug. |
