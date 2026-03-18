# Lab - Ozaki Scheme on B200

## Summary
This lab demonstrates cuBLAS floating-point emulation for FP64 GEMM on Blackwell using the two practical Ozaki-style variants exposed by NVIDIA's CUDA 13 stack:

- dynamic retained-bit control
- fixed retained-bit control

The baseline is native FP64 GEMM. The two optimized variants use `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` through the explicit cuBLAS emulation-handle API and fail fast if the library silently falls back to native FP64, so the measured delta reflects real Ozaki-style emulation instead of a mislabeled no-op.

The default lab scenario keeps the original `4096 x 4096 x 4096` GEMM size but narrows the operand range with `input_scale=1e-3`. That puts the dynamic controller into a regime where it can materially reduce the retained-bit budget instead of behaving like an accuracy-only fallback.

## Measured B200 Result
Strict benchmark-harness run on the local B200 with repo clock locking:

| Variant | Time (ms) | Speedup vs native | Retained bits | Max abs error | Mean abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native FP64 | `6.714` | `1.00x` | `-1` | `0` | `0` |
| Ozaki dynamic | `1.130` | `5.94x` | `4` | `3.0e-06` | `0` |
| Ozaki fixed | `0.842` | `7.98x` | `12` | `0` | `0` |

This tuned B200 result is useful because the two Ozaki variants now both beat native FP64 on the same shared workload:

- dynamic retained-bit control adapts down to a 4-bit retained budget on this workload and clears the native FP64 baseline by nearly `6x`
- fixed retained-bit control remains the fastest path and clears the same baseline by nearly `8x`
- both optimized paths stay verification-clean within the lab tolerance (`rtol=1e-2`, `atol=1e-2`)

## Why This Lab Exists
Ozaki-style emulation matters because low-precision tensor-core hardware is abundant, but strict FP64 accuracy remains expensive. NVIDIA now exposes fixed-point FP64 emulation directly in cuBLAS, which makes the scheme practical to benchmark without writing a custom segmented-arithmetic kernel from scratch.

Relevant references:

- NVIDIA blog: <https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/>
- cuBLAS floating-point emulation docs: <https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation>

## Benchmark Structure
The lab exposes one baseline and two optimized variants:

| Path | Role |
| --- | --- |
| `baseline_ozaki_scheme.py` | Native FP64 accuracy and performance anchor |
| `optimized_ozaki_scheme_dynamic.py` | Ozaki-style dynamic retained-bit control |
| `optimized_ozaki_scheme_fixed.py` | Ozaki-style fixed retained-bit control |

## Direct Runner
Use the direct lab runner when you want a single three-way table:

```bash
python labs/ozaki_scheme/run_lab.py
python labs/ozaki_scheme/run_lab.py --m 6144 --n 6144 --k 6144 --iters 6
```

## Harness Targets
Use the benchmark harness when you want verification, expectations, and profiler integration:

```bash
python -m cli.aisp bench list-targets --chapter labs/ozaki_scheme
python -m cli.aisp bench run --targets labs/ozaki_scheme --profile minimal
python -m cli.aisp bench run --targets labs/ozaki_scheme:ozaki_scheme_dynamic --profile minimal
python -m cli.aisp bench run --targets labs/ozaki_scheme:ozaki_scheme_fixed --profile minimal
```

The canonical expectation file for this host is `labs/ozaki_scheme/expectations_b200.json`, generated from a strict `--profile none --update-expectations` run.

## Why Dynamic Wins Here
The profiled `ozaki_scheme_dynamic` pair on the tuned default scenario points to a reduction in emulation work, not a bandwidth-driven win:

- strict timing run (`20260318_183903__bench__profile_none_targets_labs_ozaki_scheme`) measured native FP64 at `6.714 ms` and dynamic Ozaki at `1.130 ms`
- profiled dynamic pair run (`20260318_185200__bench__profile_minimal_targets_labs_ozaki_scheme_ozaki_scheme_dynamic`) kept the same verification-clean result while showing baseline `nsys total_gpu_time_ms ~= 258.4` versus dynamic `~= 157.0`
- the matched NCU kernel metrics stayed almost flat on DRAM and L2 throughput, while active warps rose from about `44.85%` to `50.08%`

That combination suggests the speedup is primarily coming from the dynamic controller collapsing the amount of fixed-point emulation work needed for this low-range input distribution, rather than from a memory-system optimization.

## What The Binary Emits
Each CUDA binary prints:

- `TIME_MS`
- `TFLOPS`
- `RETAINED_BITS`
- `EMULATION_USED`
- `MAX_ABS_ERROR`
- `MEAN_ABS_ERROR`
- `VERIFY_CHECKSUM`

That keeps the harness integration thin while still surfacing the Ozaki-specific knobs that matter.

## Validation Goal
The optimized variants only count if both conditions hold:

- `EMULATION_USED: 1`
- numerical error remains small relative to the native FP64 reference

If cuBLAS floating-point emulation falls back to native FP64, the optimized binary exits non-zero.
