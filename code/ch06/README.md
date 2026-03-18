# Chapter 6 - CUDA Programming Fundamentals

## Summary
Moves from Python into CUDA C++: write first kernels, reason about occupancy, control memory layouts, and experiment with ILP, launch bounds, and unified memory on Blackwell devices.

## Problem
Chapter 6 is where kernel mechanics stop being theoretical. The real question is which low-level changes register pressure, vector width, launch bounds, ILP, and memory layout actually show up as measured improvement under the harness.

## Baseline Path
- simple kernels with minimal attention to occupancy or memory layout
- scalar or poorly amortized execution paths
- examples that surface launch and memory inefficiency clearly

## Optimized Path
- vectorized and parallelized kernels
- ILP- and launch-bound-aware variants
- autotuned or occupancy-tuned schedules where the hardware payoff is visible

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `add` | `172.202 ms` | `0.044 ms` | `3881.04x` | naive add path replaced with a true CUDA implementation |
| `attention_ilp` | `140.603 ms` | `0.529 ms` | `265.82x` | the attention-score inner loop moves from one dependent chain per thread to four independent chains |
| `autotuning` | `63.881 ms` | `16.310 ms` | `3.92x` | schedule selection finds a materially better kernel config |

This chapter has the biggest synthetic-looking wins in the repo because many baselines are intentionally pedagogical. They are still useful, but they should be read as controlled teaching deltas, not production uplift guarantees.

## Profiler Evidence
This is a profiler-heavy chapter by design. Use deep-dive runs when you want to connect the wall-clock delta to occupancy, memory throughput, and launch behavior:

```bash
python -m cli.aisp bench run --targets ch06:add --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch06:attention_ilp --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch06:autotuning --profile deep_dive --single-gpu
```

Expected profiler story:
- `add`: removal of pure-framework overhead and better GPU utilization
- `attention_ilp`: higher effective work per thread inside an attention-shaped score microbenchmark, not a different attention algorithm
- `autotuning`: better schedule choice rather than different math

## Repro Commands
```bash
python -m ch06.compare
python -m cli.aisp bench list-targets --chapter ch06
python -m cli.aisp bench run --targets ch06 --profile minimal
python -m cli.aisp bench run --targets ch06:attention_ilp --profile deep_dive --single-gpu
```

## Learning Goals
- Write and launch custom kernels that mirror the harness workloads.
- Understand how occupancy, launch bounds, and register pressure interact.
- Use ILP and vectorized memory ops to increase throughput per thread.
- Validate unified memory and allocator tuning on Blackwell GPUs.

## Directory Layout
| Path | Description |
| --- | --- |
| `my_first_kernel.cu`, `simple_kernel.cu`, `baseline_add_cuda.cu`, `optimized_add_cuda_parallel.cu`, `baseline_add.py`, `optimized_add.py`, `baseline_add_cuda.py`, `optimized_add_cuda_parallel.py` | Hello-world kernels plus Python wrappers for the canonical `add` / `add_cuda` targets and the underlying CUDA launch-path verification. |
| `baseline_attention_ilp.py`, `optimized_attention_ilp.py`, `baseline_gemm_ilp.py`, `optimized_gemm_ilp.py`, `ilp_low_occupancy_vec4_demo.cu`, `ilp_extreme_low_occupancy_vec4_demo.cu` | Instruction-level parallelism studies that keep the math fixed while changing independent chains per thread, register pressure, and vector width. |
| `baseline_bank_conflicts.cu`, `optimized_bank_conflicts.cu`, `baseline_launch_bounds*.{py,cu}`, `optimized_launch_bounds*.{py,cu}` | Bank conflict and launch-bound exercises to highlight shared memory layouts and CTA sizing. |
| `baseline_autotuning.py`, `optimized_autotuning.py`, `memory_pool_tuning.cu`, `stream_ordered_allocator/` | Autotuning harness plus allocator experiments for controlling fragmentation and stream ordering. |
| `unified_memory.cu`, `occupancy_api.cu`, `baseline_quantization_ilp.py`, `optimized_quantization_ilp.py` | Unified memory demo, occupancy calculator sample, and quantization-focused ILP workloads. |
| `compare.py`, `Makefile`, `expectations_{hardware_key}.json`, `workload_config.py` | Harness entry, build scripts, expectation baselines, and workload settings. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch06.compare
python -m cli.aisp bench list-targets --chapter ch06
python -m cli.aisp bench run --targets ch06 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `nvcc -o baseline_add_cuda_sm121 baseline_add_cuda.cu` vs the optimized vectorized version shows a clear bandwidth delta when inspected with Nsight Compute.
- `python optimized_autotuning.py --search` converges to the same schedule as the curated preset and logs the score table under `artifacts/`.
- `python -m ch06.compare` confirms the chapter baseline/optimized pairs stay runnable through the harness after ILP and launch-bound refactors.

## Notes
- `arch_config.py` forces SM-specific compile flags (e.g., disabling pipelines on unsupported GPUs) so targets fail gracefully on older hardware.
- The canonical public add targets are `add` and `add_cuda`; the older add-tensors wrapper names were removed instead of kept as legacy aliases.
- `attention_ilp` is an attention-score preprocessing microbenchmark. It is intentionally not a fused SDPA or multi-stream overlap example.
- CUDA extensions in `cuda_extensions/` can be imported directly into notebooks for interactive prototyping.
