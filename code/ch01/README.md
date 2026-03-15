# Chapter 1 - Performance Fundamentals

## Summary
Establishes the baseline benchmarking discipline with a simple training-loop goodput benchmark and a small CUDA GEMM case study. The goal is to ground later optimizations in repeatable measurement, equivalent workloads, and verifiable outputs.

## Problem
Chapter 1 sets the measurement contract for the rest of the repo. The useful question here is not "can I make something faster?" but "can I show a repeatable before/after delta without changing the workload or hiding correctness problems?"

## Baseline Path
- eager FP32 or minimally optimized Python training loops
- one-launch-per-work-item CUDA examples
- benchmark setups that make launch overhead and framework overhead visible

## Optimized Path
- mixed-precision and fused microbatch execution for the training loop
- separate precision-only and fusion-only variants so the training-loop story is decomposable
- batched or strided CUDA launches to amortize dispatch cost
- memory-reduction variants where the main win is footprint, not raw speed

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `gemm` | `0.364 ms` | `0.012 ms` | `29.51x` | strided batched GEMM removes launch overhead |
| `performance` | `68.836 ms` | `14.286 ms` | `4.82x` | FP16 + fused microbatches raise goodput |
| `nvfp4_mlp` | `1.130 ms` | `1.167 ms` | `0.97x` | near-flat latency, but `37.9%` lower memory use |

This chapter intentionally includes both pure speedup examples and one memory-oriented tradeoff so later chapters do not overfit on "speedup only" thinking.

## Training-Loop Variants
The Chapter 1 training loop is intentionally split into three related targets:

| Target | Isolated change | Intended lesson |
| --- | --- | --- |
| `performance` | FP16 math + fused microbatches | the combined goodput story |
| `performance_fp16` | FP16 math only | what tensor-core-friendly precision buys you without changing batching |
| `performance_fusion` | fused microbatches only | what launch amortization buys you without changing math precision |

## Profiler Evidence
Use deep-dive harness runs when you want proof of where the win comes from instead of only a runtime delta:

```bash
python -m cli.aisp bench run --targets ch01:gemm --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch01:performance --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch01:nvfp4_mlp --profile deep_dive --single-gpu
```

The expected profiler story is straightforward:
- `gemm`: fewer launches and lower dispatch overhead
- `performance`: fewer launches plus faster tensor-core math
- `performance_fp16`: faster GEMMs from FP16 tensor-core math with the same microbatch structure
- `performance_fusion`: fewer forward/backward launches at unchanged FP32 math
- `nvfp4_mlp`: reduced memory footprint rather than a large wall-clock win

## Repro Commands
```bash
python -m ch01.compare
python -m cli.aisp bench list-targets --chapter ch01
python -m cli.aisp bench run --targets ch01 --profile minimal
python -m cli.aisp bench run --targets ch01:gemm --profile deep_dive --single-gpu
```

## Learning Goals
- Profile a minimal PyTorch training loop with the shared harness and reason about throughput vs latency.
- Separate precision wins from batching wins instead of treating all training-loop speedups as one bundle.
- Compare hand-written GEMM kernels in batched vs. strided forms to understand arithmetic intensity.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_performance.py`, `optimized_performance.py`, `optimized_performance_fp16.py`, `optimized_performance_fusion.py` | Training-loop variants covering the baseline, the combined FP16+fusion path, the FP16-only path, and the fusion-only path. |
| `baseline_gemm.cu`, `optimized_gemm_batched.cu`, `optimized_gemm_strided.cu` | CUDA GEMM variants (single, batched, strided) used to illustrate launch amortization and memory coalescing. |
| `compare.py`, `workload_config.py`, `arch_config.py`, `expectations_{hardware_key}.json` | Harness entrypoint, workload shapes, architecture overrides, and stored expectation thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch01.compare
python -m cli.aisp bench list-targets --chapter ch01
python -m cli.aisp bench run --targets ch01 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m ch01.compare` reports the chapter baseline/optimized training loop pair through the shared harness with consistent workloads.
- Running `make && ./baseline_gemm_sm100` vs `./optimized_gemm_batched_sm100` shows a substantial drop in launch count and total runtime.

## Notes
- `requirements.txt` pins lightweight extras (Typer, tabulate) used by helper scripts.
- `Makefile` builds the CUDA GEMM binaries with SM-specific suffixes for quick diffing.
