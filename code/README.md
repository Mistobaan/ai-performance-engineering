# AI Systems Performance Engineering

## Summary
Reference implementation of high-performance PyTorch, CUDA, and Triton workloads for NVIDIA Blackwell platforms.
The repository packages 20 focused chapters, advanced labs, and the shared benchmarking harness so you can profile baselines, apply optimizations, and capture artifacts that prove performance gains.

Roadmap: [`docs/performance_repo_roadmap.md`](/home/cfregly/ai-performance-engineering/code/docs/performance_repo_roadmap.md) defines the prioritized plan for canonical suites, trend tracking, anti-pattern enforcement, shared benchmark bases, and evidence-first documentation.

## Problem
Most performance repos are easy to browse and hard to trust. This one is meant to do the opposite:
- baseline and optimized paths live side-by-side
- correctness checks run before speedup claims matter
- profiler artifacts and benchmark manifests are first-class outputs, not optional extras

## Tier-1 Canonical Suite
The fastest way to answer "is this repo still delivering real wins?" is the canonical tier-1 suite.

```bash
python -m cli.aisp bench run-tier1 --single-gpu --profile minimal
```

That command now writes a stable history package under `artifacts/history/tier1/<run_id>/`:
- `summary.json`: per-target baseline, optimized path, and best speedup
- `regression_summary.md`: human-readable before/after summary against the previous tier-1 run
- `regression_summary.json`: machine-readable regressions and improvements
- `trend_snapshot.json`: run-history summary for dashboards and release notes
- `artifacts/history/tier1/index.json`: suite history index

See [`docs/tier1_benchmark_suite.md`](/home/cfregly/ai-performance-engineering/code/docs/tier1_benchmark_suite.md) for the current target list, artifact contract, and interpretation guidance.

## Current Representative Deltas
These numbers are taken from the latest canonical tier-1 history summary rather than from hand-maintained README text.

Source artifact: `artifacts/history/tier1/20260309_230500_tier1_green_to_green/summary.json`

Representative suite speedup: `7.29x` geomean, `8.59x` median, `21.95x` arithmetic average.

| Target | Baseline | Optimized | Measured delta | Artifact |
| --- | ---: | ---: | ---: | --- |
| `labs/block_scaling:block_scaling` | `0.196 ms` | `0.112 ms` | `1.76x` | `artifacts/history/tier1/20260309_230500_tier1_green_to_green/summary.json` |
| `labs/flashattention4:flashattention4_alibi` | `5.566 ms` | `0.372 ms` | `14.98x` | `artifacts/history/tier1/20260309_230500_tier1_green_to_green/summary.json` |
| `labs/persistent_decode:persistent_decode` | `1.331 ms` | `0.090 ms` | `14.74x` | `artifacts/history/tier1/20260309_230500_tier1_green_to_green/summary.json` |
| `labs/kv_optimization:kv_standard` | `1604.413 ms` | `968.251 ms` | `1.66x` | `artifacts/history/tier1/20260309_230500_tier1_green_to_green/summary.json` |
| `ch04:gradient_fusion` | `3.634 ms` | `0.038 ms` | `96.14x` | `artifacts/history/tier1/20260309_230500_tier1_green_to_green/summary.json` |
| `labs/real_world_models:llama_3_1_8b` | `12.848 ms` | `5.282 ms` | `2.43x` | `artifacts/history/tier1/20260309_230500_tier1_green_to_green/summary.json` |

## Profiler Evidence
When you want proof beyond wall-clock timing, use the same harness target with a profiling mode instead of a different script.

```bash
python -m cli.aisp bench run --targets labs/block_scaling:block_scaling --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_alibi --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/persistent_decode:persistent_decode --profile deep_dive --single-gpu
```

- `minimal` is the fastest artifact-bearing path.
- `deep_dive` is the profiler-backed path for Nsight Systems + Nsight Compute comparisons.
- The benchmark harness now blocks more hot-path anti-patterns and follows imported helper code, so "clean benchmark" means more than it used to.

## Cluster Evaluation
Cluster evaluation has one supported artifact contract for new work:

```text
cluster/runs/<run_id>/
  manifest.json
  structured/
  raw/
  figures/
  reports/
```

Start with:
- [`cluster/README.md`](/home/cfregly/ai-performance-engineering/code/cluster/README.md) for the current commands and folder contract.
- `python -m cli.aisp cluster common-eval --preset common-answer-fast ...` for the normal "evaluate this system" ask.
- `python -m cli.aisp cluster common-eval --preset modern-llm ...` when you need the full canonical package.
- `python -m cli.aisp cluster common-eval --preset multinode-readiness ...` before first real multi-node workloads.
- `python -m cli.aisp cluster promote-run --run-id <run_id> ...` when one collected run should become the published localhost package.

The current published canonical package lives under `cluster/published/current/`. New collection still happens under `cluster/runs/<run_id>/`.

## Learning Goals
- Understand how the chapters, labs, and shared tooling fit together.
- Stand up a reproducible environment for PyTorch 2.10-dev + CUDA 13 workloads on Blackwell GPUs.
- Run the benchmark harness directly or through the Typer CLI for automated artifact capture.
- Validate peak hardware characteristics before grading optimizations against stored expectations.

## Directory Layout
| Path | Description |
| --- | --- |
| `ch01` - `ch20` | One directory per chapter with baseline/optimized benchmarks, workload configs, and chapter-level harness entrypoints such as `ch01/compare.py`. |
| `labs/` | Deep-dive labs for matmul, routing, FlexAttention, MoE, persistent decode, distributed training, and more. |
| `core/benchmark/`, `profiling/`, `core/`, `optimization/`, `analysis/` | Shared harness, logging, workload metadata, profiling, and optimization utilities used by every chapter. |
| `python -m cli.aisp bench` | Typer-based CLI for running and profiling targets with reproducible artifacts. |
| `docs/` + `core/scripts/` | Operational guides, profiling workflows, and setup/reset helpers (`setup.sh`, `cleanup.py`, `reset-gpu.sh`). |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_latest.txt
python -m cli.aisp bench list-targets --chapter ch01
python -m cli.aisp bench run --targets ch01 --profile minimal
python -m cli.aisp bench run-tier1 --single-gpu --profile minimal
```
- `setup.sh` installs system prerequisites (drivers, CUDA, Nsight) and should be rerun after driver upgrades.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited hosts.
- Use `python -m cli.aisp bench expectations --hardware b200 --min-speedup 1.05` to report expectation entries below a target threshold.
- Use `python -m cli.aisp bench run --targets ch*` for automated regression suites.
- Portable runs do not update expectation files unless `--allow-portable-expectations-update` is supplied.
- `python core/analysis/analyze_expectations.py --artifacts-dir artifacts` compares new runs to stored thresholds.

## Validation Checklist
- `pytest tests/integration` succeeds to confirm harness discovery and CLI plumbing.
- `python core/benchmark/benchmark_peak.py` reports TFLOP/s, bandwidth, and NVLink numbers close to the published ceilings.

## Wall of Shame
The benchmark harness includes a strict set of correctness and validity checks to prevent misleading speedups.
Below is the reference list of validity issues we explicitly protect against, plus real-world incidents that
motivated these checks.

Note: All 95 validity issues are protected by the harness.

CUDA Graph Note: Capturing CUDA graphs in `setup()` is allowed for steady-state replay benchmarks (we intentionally
measure replay, not capture). It is NOT allowed to precompute and reuse the final output from `setup()`. The output
used for verification must come from the timed `benchmark_fn()` run and be surfaced via `capture_verification_payload()`.

Virtualization Note: `validate_environment()` treats virtualization (hypervisor present) as a warning. Benchmarks can
run in virtualized environments, but bare metal remains the preferred source for final performance numbers.

### Benchmark Validity Issues Reference

| Category | Issue | What Happens | Protection | Status | Real-World Incident |
| --- | --- | --- | --- | --- | --- |
| Timing | Unsynced Streams | Work on non-default streams is not timed | Full device sync + StreamAuditor | OK | Locus/KernelBench 2025 |
| Timing | Incomplete Async Ops | Timer stops before async work finishes | Full device sync | OK | Locus/KernelBench 2025 |
| Timing | Event Timing Gaps | CUDA events recorded incorrectly | Cross-validate with wall clock | OK | |
| Timing | Timer Granularity | Measurement too coarse for fast ops | Adaptive iterations | OK | |
| Timing | Warmup Bleed | Real work happens during warmup | isolate_warmup_cache | OK | |
| Timing | Clock Drift | System clock changes during measurement | Monotonic clock usage | OK | |
| Timing | Profiler Overhead | Profiling tools add latency | Profile-free timing path | OK | |
| Output | Constant Output | Same result regardless of input | Jitter check | OK | |
| Output | Stale Cache | Same result across different seeds | Fresh-input check | OK | |
| Output | Approximation Drift | Rough estimate instead of full compute | Output tolerance validation | OK | |
| Output | Invalid Values (NaN) | NaN in output | validate_result NaN check | OK | |
| Output | Invalid Values (Inf) | Inf in output | validate_result Inf check | OK | |
| Output | Invalid Ground Truth | Labels/expected values wrong | GoldenOutputCache | OK | ImageNet Labels 2021, MMLU Errors 2025 |
| Output | Shape Mismatch | Output shape differs from expected | Shape validation | OK | |
| Output | Dtype Mismatch | Output dtype differs from expected | ToleranceSpec dtype check | OK | |
| Output | Denormalized Values | Subnormal floats cause slowdowns | Denormal check | OK | |
| Output | Uninitialized Memory | Output contains garbage | Memory initialization check | OK | |
| Workload | Precision Mismatch | Claims FP32 but uses FP16 | InputSignature dtype verification | OK | |
| Workload | Backend Precision Policy Drift | Global precision policy changes during timing | Backend policy immutability check | OK | PyTorch TF32 Default 2020 |
| Workload | Undeclared Shortcuts | Skips elements without declaring | Workload invariant check | OK | AI Agent Benchmark Shortcuts 2024 |
| Workload | Early Exit | Stops iteration loops early | Config immutability | OK | |
| Workload | Batch Shrinking | Processes fewer samples | InputSignature matching | OK | |
| Workload | Sequence Truncation | Processes shorter sequences | InputSignature matching | OK | |
| Workload | Hidden Downsampling | Silently reduces resolution | Dimension validation | OK | |
| Workload | Sparsity Mismatch | Different sparsity patterns | Sparsity ratio check | OK | |
| Workload | Attention Mask Mismatch | Different masking applied | Mask equivalence check | OK | |
| Workload | KV Cache Size Mismatch | Different cache sizes | Cache dimension check | OK | |
| Workload | Train/Test Overlap | Model tested on training data | Dataset isolation | OK | Computational Biology 2019 |
| Location | CPU Spillover | Work offloaded to CPU | GPU kernel time validation | OK | |
| Location | Setup Pre-computation | Work done in setup | check_setup_precomputation | OK | |
| Location | Graph Capture Cheat | Pre-compute during graph capture | GraphCaptureCheatDetector | OK | |
| Location | Warmup Computation | Compute results during warmup | isolate_warmup_cache | OK | |
| Location | Background Thread | Compute in separate thread | Process isolation | OK | |
| Location | Lazy Evaluation Skip | Returns unevaluated lazy tensor | force_tensor_evaluation | OK | |
| Location | JIT Compilation Timing | JIT compile time included/excluded inconsistently | clear_compile_cache | OK | |
| Memory | Pre-allocated Output | Result buffer allocated in setup | MemoryAllocationTracker | OK | |
| Memory | Input-Output Aliasing | Output points to pre-filled input | check_input_output_aliasing | OK | |
| Memory | Pinned Memory Timing | Async pinned transfers not waited | Transfer completion check | OK | |
| Memory | Memory Pool Reuse | Cached allocations skew timing | reset_cuda_memory_pool | OK | |
| Memory | Fragmentation Effects | Memory fragmentation differs | Memory pool reset | OK | |
| Memory | Page Fault Timing | First-touch page faults included | Memory pre-touch | OK | |
| Memory | Swap Interference | Swapping affects timing | Memory lock / swap disable | OK | |
| CUDA | Host Callback Escape | cudaLaunchHostFunc returns early | Host function tracking | OK | |
| CUDA | Async Memcpy Incomplete | D2H/H2D copies not awaited | Full device sync | OK | |
| CUDA | Workspace Pre-compute | Work in cuBLAS workspace alloc | Workspace monitoring | OK | |
| CUDA | Persistent Kernel | Kernel left running across calls | Kernel lifetime check | OK | |
| CUDA | Undeclared Multi-GPU | Work spread across undeclared GPUs | validate_environment | OK | |
| CUDA | Context Switch Overhead | CUDA context switches affect timing | Context pinning | OK | |
| CUDA | Driver Overhead | Driver calls not accounted for | Driver call tracking | OK | |
| CUDA | Cooperative Launch Abuse | Cooperative kernels bypass checks | Launch mode validation | OK | |
| CUDA | Dynamic Parallelism Hidden | Child kernels not tracked | CDP kernel tracking | OK | |
| CUDA | Unified Memory Faults | Page migration not timed | UM fault tracking | OK | |
| Compile | Compilation Cache Hit | Returns cached compiled output | clear_compile_cache | OK | |
| Compile | Trace Reuse | Exploits trace caching | torch._dynamo.reset | OK | |
| Compile | Mode Inconsistency | Different compile mode verify vs perf | Mode consistency check | OK | |
| Compile | Inductor Asymmetry | Inductor optimizations inconsistent | Compilation parity | OK | |
| Compile | Guard Failure Hidden | Recompilation not counted | get_compile_state | OK | |
| Compile | Autotuning Variance | Autotuning picks different kernels | Fixed autotuning cache | OK | |
| Compile | Symbolic Shape Exploit | Different shapes trigger different code | InputSignature matching | OK | |
| Distributed | Rank Skipping | Some ranks do not do work | check_rank_execution | OK | |
| Distributed | Collective Short-circuit | Communication skipped | NCCL validation | OK | |
| Distributed | Topology Mismatch | Claims different topology | verify_distributed | OK | |
| Distributed | Barrier Timing | Barrier timing exploited | Barrier synchronization | OK | |
| Distributed | Gradient Bucketing Mismatch | Different bucket sizes | Bucket size validation | OK | |
| Distributed | Async Gradient Timing | Async all-reduce not awaited | Full device sync | OK | |
| Distributed | Pipeline Bubble Hiding | Pipeline bubbles not counted | Bubble time tracking | OK | |
| Distributed | Shard Size Mismatch | FSDP shards differ | InputSignature matching | OK | |
| Environment | Device Mismatch | Uses different GPU than declared | validate_environment | OK | |
| Environment | Frequency Boost | Overclocked for benchmark only | lock_gpu_clocks | OK | |
| Environment | Priority Elevation | Runs at higher priority | Process isolation | OK | |
| Environment | Memory Overcommit | Exploits memory overcommit | Memory validation | OK | |
| Environment | NUMA Inconsistency | NUMA placement differs | NUMA audit | OK | |
| Environment | CPU Governor Mismatch | Different CPU frequency scaling | Governor lock | OK | |
| Environment | Thermal Throttling | GPU throttles during run | capture_gpu_state (pynvml) | OK | |
| Environment | Power Limit Difference | Different TDP settings | capture_gpu_state (pynvml) | OK | |
| Environment | Driver Version Mismatch | Different CUDA drivers | RunManifest version lock | OK | |
| Environment | Library Version Mismatch | Different cuDNN/cuBLAS | RunManifest version lock | OK | |
| Environment | Container Resource Limits | cgroups limits differ | Resource limit check | OK | |
| Environment | Virtualization Overhead | VM/container overhead varies | Bare-metal validation | OK | |
| Statistical | Cherry-picking | Only best iterations reported | All-iteration reporting | OK | Leaderboard Illusion 2025 |
| Statistical | Outlier Injection | Slow iterations added to baseline | Statistical validation | OK | |
| Statistical | Variance Gaming | Variance reporting manipulated | Consistent statistics | OK | |
| Statistical | Percentile Selection | Favorable percentile chosen | Fixed percentile policy | OK | |
| Statistical | Insufficient Samples | Too few iterations for significance | Adaptive iterations | OK | Measuring What Matters 2025 |
| Statistical | Cold Start Inclusion | First run included unfairly | Warmup enforcement | OK | |
| Statistical | GC Interference | Garbage collection during timing | gc_disabled | OK | |
| Statistical | Background Process Noise | System processes affect timing | Process isolation | OK | |
| Evaluation | Eval Code Exploitation | Benchmark code modified to pass | BenchmarkContract enforcement | OK | |
| Evaluation | Timeout Manipulation | Timeout extended to hide slowdowns | Config immutability | OK | |
| Evaluation | Metric Definition Gaming | Redefine what speedup means | Standardized metric definitions | OK | MLPerf 2019, HANS 2019, Measuring What Matters 2025, Medical LLM Benchmarks 2025 |
| Evaluation | Test Data Leakage | Training on test data | Data contamination checks | OK | Benchmark Data Contamination Survey 2024 |
| Evaluation | Benchmark Overfitting | Optimize specifically for benchmark | Fresh-input + jitter checks | OK | Underspecification 2020, Epic Sepsis 2021, NaturalCodeBench 2024 |
| Evaluation | Self-Modifying Tests | AI/code modifies its own tests | Config immutability | OK | |
| Evaluation | Benchmark Memorization | Agent memorizes test cases | Fresh-input checks, jitter | OK | AI Agent Benchmark Shortcuts 2024 |
| Evaluation | Missing Holdout Sets | No proper train/test split | Held-out evaluation data | OK | AI Agent Benchmark Shortcuts 2024, Microsoft Tay 2016 |

Total: 11 categories, 95 validity issues - all protected by the harness.

### Notable Real-World Incidents

| Year | Incident | Issue Type | What Happened | Source |
| --- | --- | --- | --- | --- |
| 2025 | Locus/KernelBench Stream Exploit | Unsynced Streams | Claimed 20x speedup on Llama FFW kernel. AI launched work on non-default CUDA streams but timer only measured default stream. 32.8 percent of RL-generated kernels exploited this, causing fake 18x speedups. | https://x.com/miru_why/status/1991773868806361138 |
| 2025 | Measuring What Matters: Construct Validity in LLM Benchmarks | Metric Definition Gaming / Construct Validity | Systematic review of 445 LLM benchmarks found construct-validity weaknesses and low statistical rigor; issued eight design recommendations. | https://ora.ox.ac.uk/objects/uuid%3Aad2b69b6-0986-42d0-a512-a6e56338b6cc |
| 2025 | Medical LLM Benchmarks and Construct Validity | Metric Definition Gaming / Construct Validity | Position paper argues exam-style medical LLM benchmarks miss real-world tasks and documents construct-validity gaps using clinical data. | https://arxiv.org/abs/2503.10694 |
| 2025 | Sakana AI Scientist Evaluation | Evaluation Integrity | Independent evaluation found frequent experiment failures and hallucinated numerical results. | https://arxiv.org/abs/2502.14297 |
| 2025 | Leaderboard Illusion (Chatbot Arena) | Cherry-picking | Analysis of Chatbot Arena reports selection effects and leaderboard instability when submissions are inconsistent or selectively disclosed. | https://arxiv.org/abs/2504.20879 |
| 2024 | MMLU Benchmark Errors | Invalid Ground Truth | Analysis found 57 percent of MMLU virology subset questions incorrect and estimated 6.49 percent errors overall. | https://arxiv.org/abs/2406.04127 |
| 2024 | AI Agent Benchmark Shortcuts | Missing Holdout Sets | Study found AI agents memorize benchmark test samples instead of learning to generalize. Many benchmarks lack proper holdout test sets. | https://arxiv.org/abs/2407.01502 |
| 2024 | NaturalCodeBench vs HumanEval | Benchmark Overfitting | Real-user coding tasks in NaturalCodeBench show large performance gaps and weak correlation with HumanEval scores. | https://aclanthology.org/2024.findings-acl.471/ |
| 2024 | Benchmark Data Contamination Survey | Data Contamination | Survey catalogs contamination pathways across LLM benchmarks and highlights mitigation gaps. | https://arxiv.org/abs/2406.04244 |
| 2023 | NLP Evaluation Data Contamination | Data Contamination | Position paper warns that LLMs trained on benchmark test splits can inflate reported scores. | https://arxiv.org/abs/2310.18018 |
| 2022 | MLPerf Participation Issues | Cherry-picking | MLPerf faced inconsistent vendor participation; selective scenario submissions led to biased performance representations. | http://web.archive.org/web/20250813110435/https://www.nextplatform.com/2022/04/08/the-performance-of-mlperf-as-a-ubiquitous-benchmark-is-lacking/ |
| 2022 | ML Benchmark Validity (Berkeley) | Benchmark Overfitting | Small changes in data distribution caused significant performance drops, questioning external validity. | https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-180.html |
| 2021 | ImageNet Label Errors | Invalid Ground Truth | At least 6 percent label errors in ImageNet validation set. | https://arxiv.org/abs/2103.14749 |
| 2021 | MLPerf Reproducibility | Benchmark Reproducibility | Users could not reproduce MLPerf v0.7 results due to inaccessible datasets and outdated repos. | https://groups.google.com/a/mlcommons.org/g/public/c/T_8UsUPIWFo |
| 2021 | Epic Sepsis Model External Validation | Benchmark Overfitting | External validation found poor discrimination and calibration for the Epic Sepsis Model, leading to missed cases and alert fatigue. | https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781307 |
| 2020 | Underspecification in ML | Benchmark Overfitting | Models with equivalent benchmark performance diverged in deployment behavior. | https://arxiv.org/abs/2011.03395 |
| 2020 | TF32 Default on Ampere | Precision Policy Drift | TF32-enabled matmul/conv trades precision for speed unless explicitly disabled. | https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere |
| 2019 | NLI Heuristic Shortcuts (HANS) | Metric Definition Gaming | Models trained on MNLI (GLUE) rely on shallow heuristics and fail on HANS, revealing spurious shortcut behavior. | https://aclanthology.org/P19-1334/ |
| 2019 | MLPerf Inference Bias | Metric Definition Gaming | Vendors selectively submitted results highlighting strengths. | http://web.archive.org/web/20191112035148/https://www.forbes.com/sites/janakirammsv/2019/11/10/the-curious-case-of-mlperf-inferencing-benchmark-results/ |
| 2019 | Computational Biology Overfitting | Train/Test Overlap | Tools developed and tested on same datasets failed on new data. | https://www.nature.com/articles/s41467-019-09406-4 |
| 2016 | Microsoft Tay Chatbot | Missing Holdout Sets | AI chatbot learned abusive behavior within 24 hours after deployment due to adversarial user interactions. | https://blogs.microsoft.com/blog/2016/03/25/learning-tays-introduction/ |

### Incident Categories and Our Protections

| Category | Incidents | Our Protection | Status |
| --- | --- | --- | --- |
| Timing Manipulation | 1 (Locus/KernelBench) | Full device sync + StreamAuditor | OK |
| Invalid Ground Truth | 2 (ImageNet Labels, MMLU) | GoldenOutputCache + validate_result | OK |
| Benchmark Overfitting | 4 (Underspecification, Epic Sepsis, NaturalCodeBench, Berkeley) | Fresh-input checks + jitter | OK |
| Data Contamination | 2 (LLM Survey 2024, NLP Contamination 2023) | Data contamination checks + fresh inputs | OK |
| Metric Gaming | 4 (Measuring What Matters 2025, Medical LLM Benchmarks 2025, HANS 2019, MLPerf 2019) | Standardized metric definitions | OK |
| Cherry-picking | 2 (Leaderboard Illusion, MLPerf 2022) | All-iteration reporting | OK |
| Train/Test Overlap | 1 (Computational Biology) | Dataset isolation + holdout enforcement | OK |
| Missing Holdout Sets | 2 (AI Agent Shortcuts, Microsoft Tay) | Held-out evaluation data | OK |
| Reproducibility | 1 (MLPerf 2021) | RunManifest version locking | OK |
| Evaluation Integrity | 1 (Sakana AI Scientist) | BenchmarkContract + verification enforcement | OK |
| Precision Policy Drift | 1 (TF32 Default) | Backend policy immutability check | OK |

### Deep Dive: The Locus/KernelBench Stream Timing Vulnerability

This 2025 incident illustrates why correctness verification alone is insufficient.

```python
# VULNERABLE TIMING (what KernelBench did)
start_event.record(original_model_stream)  # Only records on default stream
model(*inputs)                              # But work runs on s1, s2, s3
end_event.record(original_model_stream)    # Timer stops before s1/s2/s3 finish
torch.cuda.synchronize(device=device)      # Waits, but timing already recorded

# CORRECT TIMING (the fix)
for stream in custom_model_streams:
    custom_model_stream.wait_stream(stream)  # Wait for ALL streams
_event.record(custom_model_stream)           # Then record timing
```

The exploit pattern:
1. AI creates non-default streams: `s1 = getStreamFromPool()`, `s2 = ...`, `s3 = ...`
2. AI launches GEMMs on those streams: `at::mm_out(gate, x2d, gate_proj.t())` on s1
3. AI does not call `setCurrentCUDAStream(s3)` or wait for streams before returning
4. Correctness test uses `torch.cuda.synchronize()` and passes
5. Performance test uses stream-specific events and reports fake speedups

Result: 82/250 (32.8 percent) of RL-generated CUDA kernels exploited this, producing artificial 18x speedups with
zero actual performance improvement.

### Protection Implementation Reference

| Module | Key Protections |
| --- | --- |
| `core/harness/benchmark_harness.py` | Full device sync, L2 cache clearing, GPU clock locking, warmup isolation, config immutability, adaptive iterations, CUDA graph mode |
| `core/harness/validity_checks.py` | StreamAuditor, MemoryAllocationTracker, GraphCaptureCheatDetector, gc_disabled, clear_compile_cache, capture_gpu_state, validate_environment |
| `core/harness/l2_cache_utils.py` | Dynamic L2 cache size detection, clear_l2_cache |
| `core/benchmark/verify_runner.py` | VerifyRunner, GoldenOutputCache, jitter check, fresh-input check, output comparison, workload invariants |
| `core/benchmark/verification.py` | InputSignature, ToleranceSpec, QuarantineReason, seed mutation detection |
| `core/benchmark/quarantine.py` | QuarantineManager with persistence |
| `core/benchmark/contract.py` | BenchmarkContract enforcement |

## Notes
- `core/scripts/profile_all_workloads.sh` and `ncu_template.ini` capture Nsight traces with consistent metric sets.
- `artifacts/runs/` holds run outputs (results/profiles/reports/logs); clean via `python cleanup.py` when rotating hardware.
- `docs/perf_intake_and_triage.md` outlines the standard intake bundle for performance investigations.
