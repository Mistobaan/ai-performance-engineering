# Book-After Narrative Edit Plan

Derived from:
- `deep_dive_final_disposition.md`
- `chapter_narrative_review_queue.md`
- `deep_dive_expectation_refresh_summary.md`

## Decision Rules

- `refresh + keep`
  Keep the benchmark and the current chapter story. No narrative demotion needed.
- `refresh + qualify`
  Keep the benchmark and refresh expectations, but rewrite the prose so it reads as a modest or context-dependent win instead of a headline improvement.
- `hold + reframe`
  Do not bless the current benchmark as a strong optimization win. Either demote the claim, change the claim dimension, or replace the benchmark in the chapter story.
- `goal_specific`
  Keep the example, but evaluate it against its declared non-speed goal rather than speedup.

## Chapter Actions

| Chapter | Target | Disposition | Stay Within Current Narrative? | Recommended Edit |
| --- | --- | --- | --- | --- |
| `ch01` | `nvfp4_mlp` | `hold + reframe` | `No` | Reframe as a precision/capability tradeoff story or replace with a stronger FP4/NVFP4 example. Do not present as a durable speed win. |
| `ch04` | `symmetric_memory_perf` | `hold + reframe` | `Maybe` | Keep the symmetric-memory topic, but demote this benchmark from “clear improvement” to “weak/conditional on this environment.” |
| `ch05` | `ai` | `hold + reframe` | `Maybe` | Keep the AI/roofline teaching point, but do not sell this concrete benchmark as a strong speedup. |
| `ch05` | `distributed` | `hold + reframe` | `No` | Scope explicitly to multi-GPU or replace with a single-GPU-compatible example. Current result is a host mismatch, not an optimization miss. |
| `ch08` | `nvfp4_mlp` | `hold + reframe` | `No` | Same family-level action as `ch01`. Do not independently bless this chapter copy. |
| `ch08` | `tiling_tcgen05` | `refresh + qualify` | `Yes` | Keep it, but phrase it as a small/context-sensitive win. Emphasize when tiling helps and when it does not. |
| `ch09` | `cublaslt_gemm_fp4` | `hold + reframe` | `No` | Reframe as a capability-gated path on this stack or replace with an available FP4 winner. |
| `ch09` | `cutlass_gemm_fp4_perchannel` | `refresh + qualify` | `Yes` | Keep it as a modest win with stack/toolchain caveats. Avoid framing it as a dominant FP4 story. |
| `ch10` | `cooperative_persistent` | `refresh + qualify` | `Yes` | Keep, but describe as a small win tied to persistence/launch regime rather than a universal speedup. |
| `ch10` | `matmul_tcgen05_epilogue` | `refresh + qualify` | `Yes` | Keep, but narrow the claim to a modest epilogue-side improvement. |
| `ch10` | `persistent_matmul_tma` | `refresh + qualify` | `Yes` | Keep, but explicitly tie the win to the tested persistence/TMA regime. |
| `ch10` | `tcgen05_cluster_pipeline` | `refresh + qualify` | `Yes` | Keep, but present as a smaller cluster-pipeline win, not a marquee result. |
| `ch10` | `tcgen05_warp_specialization_cutlass` | `hold + reframe` | `Maybe` | Keep the warp-specialization discussion, but demote this specific CUTLASS benchmark or replace it with a stronger representative. |
| `ch10` | `warp_specialized_cluster_pipeline` | `refresh + qualify` | `Yes` | Keep with narrower wording: good example of the technique, but only a modest gain here. |
| `ch11` | `tensor_cores_streams` | `hold + reframe` | `Maybe` | Keep the streams/tensor-core interaction topic, but not as a strong win. Consider rephrasing as a concurrency limit case. |
| `ch12` | `nvfp4_mlp` | `hold + reframe` | `No` | Same family-level action as `ch01`. |
| `ch13` | `kv_cache_naive_pool` | `refresh + qualify` | `Yes` | Keep as a small infrastructure-side win, not a headline acceleration. |
| `ch13` | `training_standard` | `goal_specific` | `Yes` | Judge it on memory/training-goal criteria, not raw speed. The chapter narrative should say that directly. |
| `ch15` | `allreduce_rmsnorm` | `hold + reframe` | `Maybe` | Keep the fused-distributed idea, but do not market this benchmark as a strong speed gain on this environment. |
| `ch15` | `disaggregated_inference` | `refresh + qualify` | `Yes` | Keep, but frame as context-sensitive and architecture-sensitive. |
| `ch15` | `moe_overlap` | `refresh + qualify` | `Yes` | Keep, but qualify as a modest overlap gain in this setup. |
| `ch15` | `moe_overlap_shared_expert` | `refresh + qualify` | `Yes` | Same as `moe_overlap`: keep, refresh, qualify. |
| `ch15` | `nvfp4_mlp` | `hold + reframe` | `No` | Same family-level action as `ch01`. |
| `ch16` | `nvfp4_mlp` | `hold + reframe` | `No` | Same family-level action as `ch01`. |
| `ch17` | `nvfp4_mlp` | `hold + reframe` | `No` | Same family-level action as `ch01`. |
| `ch17` | `prefill_decode_disagg_tpot_long` | `hold + reframe` | `Maybe` | Keep the system tradeoff discussion, but demote the benchmark from “clear win” to “not convincing here.” |
| `ch19` | `vectorization_memory` | `hold + reframe` | `Maybe` | Keep the vectorization topic, but rewrite this example as a weak/conditional result unless a better benchmark is substituted. |
| `ch20` | `nvfp4_mlp` | `hold + reframe` | `No` | Same family-level action as `ch01`. |

## What To Fix Versus What To Rewrite

- Fix first when the benchmark family is supposed to be central and reusable.
  This applies most strongly to the repeated `nvfp4_mlp` family.
- Rewrite first when the benchmark is valid but weak on this environment.
  This applies to `symmetric_memory_perf`, `tensor_cores_streams`, `allreduce_rmsnorm`, and `vectorization_memory`.
- Scope explicitly when the “missing win” is really an environment mismatch.
  This applies to `distributed` and `cublaslt_gemm_fp4`.
- Keep and qualify when the gain is real but small.
  This applies to the `11` `qualify` cases above.

## Recommended Editing Order

1. Update chapter prose/callouts for the `11` `qualify` cases so the refreshed expectations and the narrative match.
2. Reframe or demote the `7` flat/weak local cases.
3. Decide whether `distributed` and `cublaslt_gemm_fp4` stay as scoped examples or get replaced.
4. Do a single family-level decision for `nvfp4_mlp`, then apply that decision consistently across `ch01`, `ch08`, `ch12`, `ch15`, `ch16`, `ch17`, and `ch20`.
5. Keep `training_standard` out of speedup tables and speed-focused prose.
