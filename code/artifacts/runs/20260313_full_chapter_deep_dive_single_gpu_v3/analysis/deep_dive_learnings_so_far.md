# Deep Dive Learnings So Far

Updated: 2026-03-14 UTC
Run ID: `20260313_full_chapter_deep_dive_single_gpu_v3`

## Scope Covered

- Original deep-dive sweep completed all `20` chapters.
- Original sweep outcome: `240` succeeded, `18` failed, `8` skipped.
- Failure rechecks are now complete and the canonical failure ledger is fully green: `18/18` original failures resolved.
- Expected skips remain part of the validated story: `ch04` includes `1` hardware skip and `7` distributed skips from the original single-GPU deep-dive pass.

## Main Learnings

- Several of the biggest reported speedups in `ch06` are not kernel-speed improvements. They are mostly host/runtime overhead elimination.
- The clearest GPU-side optimization story so far is in `ch01` performance examples. Baseline occupancy is very low, while optimized runs materially raise occupancy and SM utilization.
- `ch07` shows strong transfer/bandwidth wins, especially for `hbm_peak` and `tma_copy`.
- A few examples are weak or negative and should be treated as chapter/expectation review candidates rather than clear success stories.

## Strong Signals

- `ch01:performance`
  Deep-dive points to a real utilization improvement, not just Python overhead cleanup.
- `ch01:performance_fusion`
  Same pattern as `performance`: low baseline occupancy, much healthier optimized occupancy.
- `ch06:add*`
  Huge speedups are real for end-to-end latency, but profiler evidence says the kernel itself is basically unchanged.
- `ch07:hbm_peak`
  Large end-to-end win with clear timeline reduction in optimized profiling.
- `ch07:tma_copy`
  Strong bandwidth/transfer improvement story.

## Review Candidates

- Refined review ledger:
  `deep_dive_review_candidates_refined.md`
- Refined machine-readable ledger:
  `deep_dive_review_candidates_refined.json`
- Current count:
  `28` examples should not be blindly blessed as strong optimization stories on this environment.
- Bucket breakdown:
  `15` flat-or-negative (`<=1.05x`), `12` weak-positive (`1.05x - 1.20x`), `1` non-speed-goal example.
- Representative flat-or-negative examples:
  `ch01:nvfp4_mlp`, `ch05:distributed`, `ch09:cublaslt_gemm_fp4`, `ch11:tensor_cores_streams`, `ch15:allreduce_rmsnorm`
- Representative weak-positive examples:
  `ch05:ai`, `ch04:symmetric_memory_perf`, `ch15:disaggregated_inference`, `ch15:moe_overlap`, `ch08:tiling_tcgen05`
- Important nuance:
  `ch13:training_standard` is a memory-goal example, not a speed story, so it should not be narrated as a speedup example.

## Weak-Case Action Plan

- Reproducible classifier:
  `core/analysis/derive_weak_case_actions.py`
- Current action ledger:
  `deep_dive_weak_case_actions.md`
- Current machine-readable action ledger:
  `deep_dive_weak_case_actions.json`
- Root-cause investigation ledger:
  `deep_dive_weak_case_root_causes.md`
- Machine-readable root-cause investigation ledger:
  `deep_dive_weak_case_root_causes.json`
- Current action counts:
  `7` family-level investigations, `2` missing-optimized-win investigations, `7` expectation/story holds, `11` contextual-win qualifications, `1` non-speed example.
- Default policy:
  weak cases are not auto-blessed, and their expectations stay on hold unless they move out of the high-risk buckets.
- High-risk buckets:
  `family_level_investigation_before_blessing`, `investigate_missing_optimized_win`, and `hold_expectations_and_reframe_story`
- What that means in practice:
  repeated weak families like `nvfp4_mlp` need family-level debugging before any chapter instance is blessed; flat examples like `tensor_cores_streams` or `allreduce_rmsnorm` should be demoted or reframed; cases with no credible optimized winner like `distributed` and `cublaslt_gemm_fp4` need root-cause investigation, not expectation refresh.
- Lower-risk bucket:
  `qualify_as_small_or_contextual_win`
- What that means in practice:
  examples like `disaggregated_inference`, `moe_overlap`, and `tiling_tcgen05` can stay, but they should be described as small or context-sensitive wins rather than headline optimization stories.
- Non-speed bucket:
  `treat_as_non_speed_example`
- What that means in practice:
  examples like `training_standard` should be evaluated against the declared goal, not forced into a speedup narrative.
- Repeated-family and no-winner investigations already completed:
  `nvfp4_mlp` is weak as a family on this environment, `distributed` is a single-GPU scope mismatch, and `cublaslt_gemm_fp4` is capability-gated by the current driver/toolchain stack.

## Expectation Drift Seen So Far

- Large drift exists already in completed chapters, but updates are still provenance-gated.
- Biggest currently visible completed-chapter deltas include:
  - `ch01:performance_fp16`
  - `ch01:performance`
  - `ch04:dataparallel`
  - `ch05:decompression`

## Source Artifacts

- Structured run checkpoint:
  `results/benchmark_test_results.json`
- Event stream:
  `logs/benchmark_events.jsonl`
- Human-readable live log:
  `../../../sweeps/20260313_full_chapter_deep_dive_single_gpu_v3/deep_dive.log`
- Per-example Nsight/PyTorch artifacts:
  `profiles/bench/chXX/<example>/...`

## Storage Gap

- Raw evidence is being stored well.
- Interpreted learnings were not previously being stored in a first-class summary artifact.
- This file is the start of that centralized learning log for the live deep-dive run.

## Post-Run Failure Learnings

- The final deep-dive `v3` run finished with `18` failures, but most were not algorithmic benchmark failures. The dominant class was profiler-path failure, especially `nsys`.
- Benchmark-local deep-dive timeout needs are real. Several examples in `ch09`, `ch10`, `ch12`, `ch13`, `ch16`, `ch17`, and `ch20` need materially larger `nsys` budgets than repo defaults.
- The original config merge policy in `core/harness/run_benchmarks.py` was wrong for deep-dive reliability. It silently clamped benchmark-local timeout overrides back down to the run-level default, which made local timeout fixes ineffective.
- The corrected rule is:
  benchmark-local timeouts may widen repo defaults, but explicit CLI timeout overrides stay authoritative.
- This is a core trust lesson: if structured config merge semantics are wrong, the logs can make it look like a benchmark-local fix landed when the effective runtime config still disagrees.

## Profiler Wrapper Learnings

- Python profiler subprocesses must register imported benchmark modules in `sys.modules` before `exec_module()`.
- Without that, import-time introspection can fail under Python 3.12, especially for `@dataclass`-based modules. This was the root cause behind the `ch13:fp8_static` deep-dive profiler failure.
- This was correctly fixed as cross-cutting infrastructure in:
  `core/utils/python_entrypoints.py`
- The learning is simple:
  import behavior in profiler wrappers is part of benchmark correctness, not just tooling convenience.

## Verification Learnings

- Verification payload semantics matter as much as timing semantics.
- `ch13:autograd_standard` showed that returning raw benchmark output instead of the normalized verification payload can produce false verification failures (`Half` vs `Float`) even when the underlying computation is fine.
- The stable rule is:
  benchmarks should verify against their explicit payload contract, not whatever incidental dtype happens to be left in `self.output`.

## Process Learnings

- A rerun started before a source fix is effectively stale for root-cause validation, even if it is still alive and making progress.
- For profiler-path bugs, it is better to kill the stale rerun and relaunch from a clean run id than to keep reading misleading live progress.
- Provenance needs to include not just run id and commit, but also effective runtime config. The critical field here was the actual launched `timeout_seconds` in the event stream.

## Current Validation Thread

- Focused recheck run:
  `20260313_deep_dive_failed_targets_recheck_v3`
- Current live validation has already confirmed one key fix:
  `optimized_compute_bound` now launches `nsys` with `timeout_seconds=1200` instead of the incorrect `540`.
- Additional reliability lesson from that rerun:
  a timed-out `nsys` capture can still yield a usable `.nsys-rep` artifact a few seconds later.
- Concrete case:
  `ch09:compute_bound` optimized `nsys` timed out at `1200s`, but the `.nsys-rep` file materialized on disk about `3s` later.
- Resulting harness fix:
  `core/profiling/nsight_automation.py` now waits briefly for a late-finalizing `.nsys-rep` artifact before classifying the capture as failed.
- Trust rule:
  classify capture success from the report artifact itself first, and treat metric extraction as best-effort augmentation rather than a reason to lose a valid capture.

## Final Recheck Outcome

- Canonical deep-dive failure ledger:
  `final_failure_ledger.md`
- Canonical deep-dive disposition ledger:
  `deep_dive_final_disposition.md`
- Canonical expectation refresh summary:
  `deep_dive_expectation_refresh_summary.md`
- Canonical chapter narrative review queue:
  `chapter_narrative_review_queue.md`
- Final rerun coverage:
  original run `20260313_full_chapter_deep_dive_single_gpu_v3` plus rechecks `v6`, `v7`, `v13`, `v14`, and `v15_direct`
- Final hard outcome:
  `18` original deep-dive failures, `18` resolved on clean reruns, `0` unresolved
- Final blocker closure:
  `ch10:dsmem_reduction_v3` is now green in `20260314_deep_dive_dsmem_v3_recheck_v15_direct`, with baseline `nsys`, `ncu`, and torch all succeeding and the optimized path validating at `~1.85x`.

## Final Weak-Case Policy

- Weak and flat examples are not refreshed blindly just because the repo is now more stable.
- Repeated-family weak cases:
  keep expectations on hold and reframe chapter narrative until the shared family is improved.
- Hardware/capability-gated no-winner cases:
  keep expectations on hold and reframe them as scoped/environment-limited examples rather than optimization misses.
- Weak-but-positive cases:
  expectation refresh is acceptable only with qualified narrative that explicitly presents them as small/contextual wins.
- Non-speed examples:
  do not force them into a speedup story.

## Final Decision Counts

- `18` cases: refresh expectations and keep chapter narrative
- `11` cases: refresh expectations with qualified/smaller-win narrative
- `7` cases: hold expectations and reframe due to flat/weak story
- `7` cases: hold expectations and reframe due to repeated weak family (`nvfp4_mlp`)
- `2` cases: hold expectations and reframe due to hardware/capability gating (`distributed`, `cublaslt_gemm_fp4`)
- `1` case: evaluate against non-speed goal (`training_standard`)
- Expectation refreshes were actually applied for the approved `29` cases and validated successfully across:
  `ch08`, `ch09`, `ch10`, `ch12`, `ch13`, `ch15`, `ch16`, `ch17`, `ch20`
