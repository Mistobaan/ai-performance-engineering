# Chapter 20 - End-to-End Case Studies

## Summary
Combines kernel, memory, pipeline, and inference optimizations into holistic case studies: take a baseline pipeline, apply staged improvements, and capture proof-of-benefit artifacts for every major subsystem.

## Learning Goals
- Chain memory, pipeline, and KV-cache optimizations together to see cumulative impact.
- Generate automatic reports that compare baseline vs tuned end-to-end runs.
- Prototype new kernels via the AI kernel generator and slot them into the harness.
- Validate improvements with workload-specific acceptance tests.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_multiple_unoptimized.py`, `optimized_multiple_unoptimized.py`, `ai_kernel_generator.py`, `inductor_guard.py` | Composite workloads that stack several bottlenecks plus helpers for generating candidate kernels safely. |
| `baseline_pipeline_sequential.py`, `optimized_pipeline_sequential.py`, `baseline_end_to_end_bandwidth.py`, `optimized_end_to_end_bandwidth.py` | Pipeline and bandwidth case studies showing how optimizations interact across stages. |
| `baseline_integrated_kv_cache.py`, `optimized_integrated_kv_cache.py` | Integrated KV-cache demos that merge allocator, overlap, and NVLink pooling tricks. |
| `baseline_memory_standard.py`, `optimized_memory_standard.py` | Memory-focused harness verifying allocator changes at system level. |
| `baseline_training_single.py`, `optimized_training_single.py`, `test.cu`, `Makefile` | Single-device training case study plus CUDA kernels used in the final report. |
| `compare.py`, `arch_config.py`, `expectations_{hardware_key}.json` | Harness driver, architecture settings, and expectation baselines. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch20.compare --profile none
python -m cli.aisp bench list-targets --chapter ch20
python -m cli.aisp bench run --targets ch20 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m ch20.compare` emits per-stage summaries that show each optimized variant meeting or exceeding stored expectations.
- `python -m ch20.ai_kernel_generator --emit test.cu` produces CUDA kernels that compile via `nvcc` and integrate into the harness without manual edits.
- `python -m ch20.optimized_pipeline_sequential --trace` shows smooth NVTX ranges covering the entire pipeline, demonstrating overlap success.

## Notes
- `inductor_guard.py` provides convenience toggles for gating experimental kernels behind feature flags.
- `ai_kernel_generator.py` logs generated code to `artifacts/` for reproducibility; capture the log with your proof-of-benefit bundle.
