# Lab - Blackwell Hardware Block Scaling

## Summary
Recreates the practical flow of Colfax Research's article on hardware-supported block scaling with NVIDIA Blackwell GPUs inside this repo's lab structure. The baseline path is intentionally conservative: it materializes block scales in BF16, applies them as explicit elementwise multiplies, and then calls `matmul`. The optimized path compiles the CUTLASS/CuTe blockscaled GEMM once during setup and measures only the Blackwell hardware-supported execution path.

The lab now defaults to the larger Colfax-style workload:
- `MNKL = 8192,8192,1024,1`
- `mma_tiler_mn = 256,128`
- `cluster_shape_mn = 2,1`
- `sf_vec_size = 16`

## Credit
- Source article: Colfax Research, ["CUTLASS Tutorial: Hardware-supported Block Scaling with NVIDIA Blackwell GPUs"](<https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/>)
- Kernel/source inspiration: NVIDIA CUTLASS `examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py` and the related `72_blackwell_narrow_precision_gemm` examples

## Learning Goals
- See how Blackwell's blockscaled tensor-core path changes the cost model versus a software dequantize-and-matmul baseline.
- Run a CUTLASS/CuTe blockscaled kernel from the article in a repo-native, repeatable lab.
- Validate the numerical output against a software reference before trusting the timing.
- Sweep matrix shapes, tile shapes, and cluster shapes without rewriting the kernel code.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_block_scaling.py` | Conservative software baseline: expand scales, multiply in BF16, then matmul. |
| `optimized_block_scaling.py` | Compile-once Blackwell hardware blockscaled GEMM benchmark. |
| `block_scaling_common.py` | Shared config parsing, tensor prep, CUTLASS example loading, and timing helpers. |
| `compare_block_scaling.py` | Reproducible three-path runner: software blockscaled ref, pre-scaled BF16 GEMM, and hardware blockscaled GEMM. |
| `microbenchmark_block_scaling.py` | Apples-to-apples microbenchmark that adds the direct Colfax/CUTLASS `run()` path. |

## Running the Lab
Use the comparison runner when you want a one-command answer on correctness plus speedup:
```bash
python labs/block_scaling/compare_block_scaling.py
python labs/block_scaling/compare_block_scaling.py --json-out /tmp/block_scaling_compare.json
```

Use the microbenchmark when you want the direct CUTLASS article path side-by-side with the lab wrapper and PyTorch ranges:
```bash
python labs/block_scaling/microbenchmark_block_scaling.py
python labs/block_scaling/microbenchmark_block_scaling.py --warmup 2 --iterations 10 --json-out /tmp/block_scaling_microbench.json
```

Use the benchmark harness when you want the lab to participate in the repo's standard benchmark flows:
```bash
python -m cli.aisp bench list-targets --chapter labs/block_scaling
python -m cli.aisp bench run -t labs/block_scaling:block_scaling -p none --iterations 10 --warmup 5 --timeout-seconds 900 --single-gpu
```

## Recommended Knobs
The defaults are tuned for the larger Colfax-style B200 workload:
- `AISP_BLOCK_SCALING_MNKL=8192,8192,1024,1`
- `AISP_BLOCK_SCALING_MMA_TILER_MN=256,128`
- `AISP_BLOCK_SCALING_CLUSTER_SHAPE_MN=2,1`
- `AISP_BLOCK_SCALING_SF_VEC_SIZE=16`

You can still override them explicitly:
```bash
AISP_BLOCK_SCALING_MNKL=8192,8192,1024,1 \
AISP_BLOCK_SCALING_MMA_TILER_MN=256,128 \
AISP_BLOCK_SCALING_CLUSTER_SHAPE_MN=2,1 \
python labs/block_scaling/compare_block_scaling.py
```

## Default Tuning Pass
The current default tile/cluster pair came from a first-pass direct CUTLASS sweep on the article-sized workload (`warmup=1`, `iterations=10`, `skip_ref_check=True`):

| `mma_tiler_mn` | `cluster_shape_mn` | Direct CUTLASS latency |
| --- | --- | --- |
| `128,128` | `1,1` | `78.33 us` |
| `128,128` | `1,2` | `74.96 us` |
| `128,128` | `2,1` | `74.75 us` |
| `128,256` | `1,2` | `84.68 us` |
| `256,128` | `2,1` | `71.68 us` |
| `256,128` | `2,2` | `88.06 us` |

That keeps the default aligned with the article's representative command line while also matching the best result from the local sweep.

## Microbenchmark View
The microbenchmark reports four distinct numbers on the same logical workload:

| Path | What it measures |
| --- | --- |
| `Software blockscaled ref` | PyTorch scale multiply plus BF16 GEMM every iteration. |
| `PyTorch BF16 GEMM` | BF16 GEMM after the scales were already applied. |
| `Lab CUTLASS hardware` | The repo-native compile-once wrapper around the blockscaled tensor-core kernel. |
| `Colfax/CUTLASS direct` | The original CUTLASS example's `run()` benchmark path. |

This is the apples-to-apples interpretation:
- `Software blockscaled ref` vs `Lab CUTLASS hardware` shows the real improvement from Blackwell's hardware-supported block scaling.
- `PyTorch BF16 GEMM` isolates how much of the software path is just GEMM versus scale application overhead.
- `Lab CUTLASS hardware` vs `Colfax/CUTLASS direct` checks whether the lab wrapper is staying in the same performance range as the original example.

### Representative B200 Ranges
On this B200, with `python labs/block_scaling/microbenchmark_block_scaling.py --warmup 2 --iterations 10`, the lab produced:

| Path | Latency | TFLOP/s | Relative to lab hardware |
| --- | --- | --- | --- |
| `Software blockscaled ref` | `0.1566 ms` | `877.8` | `2.34x slower` |
| `PyTorch BF16 GEMM` | `0.1199 ms` | `1145.8` | `1.79x slower` |
| `Lab CUTLASS hardware` | `0.0670 ms` | `2050.5` | `1.00x` |
| `Colfax/CUTLASS direct` | `0.0711 ms` | `1934.0` | `1.06x slower` |

Derived takeaways from that run:
- Hardware block scaling was `2.34x` faster than the software blockscaled PyTorch reference.
- Hardware block scaling was `1.79x` faster than pre-scaled BF16 GEMM, so the hardware kernel is not just saving the explicit scale multiplies.
- The software block-scaling overhead above plain BF16 GEMM was about `0.0366 ms` or `30.53%`.
- The lab wrapper stayed within about `5.7%` of the direct Colfax/CUTLASS path.

## Harness vs Microbenchmark
The repo harness and the standalone microbenchmark answer slightly different questions:
- `microbenchmark_block_scaling.py` uses CUDA-event timing around the direct call sites, so it is the right place to compare against Colfax/CUTLASS and PyTorch-reported kernel-adjacent ranges.
- `bench run` measures the benchmark pair through the generic harness, including the per-iteration synchronization the harness uses for correctness and stability. That number is expected to be higher than the direct microbenchmark.

On this B200, the harness run:
- `python -m cli.aisp bench run -t labs/block_scaling:block_scaling -p none --iterations 10 --warmup 5 --timeout-seconds 900 --single-gpu`
- reported `0.198 ms` baseline, `0.113 ms` optimized, and `1.76x` speedup
- updated `labs/block_scaling/expectations_b200.json` from `1.669x` to `1.762x`

## Validation Checklist
- `python labs/block_scaling/compare_block_scaling.py` reports a hardware speedup greater than `1.0x` on a B200 / Blackwell system.
- `python labs/block_scaling/microbenchmark_block_scaling.py` keeps the lab wrapper in the same range as the direct Colfax/CUTLASS example.
- The comparison runner's correctness check passes before timing is reported.
- `python -m cli.aisp bench run -t labs/block_scaling:block_scaling -p none ...` executes both paths without recompiling the hardware kernel inside each measured iteration.

## Notes
- This lab is intentionally conservative on the baseline side. It pre-expands scale factors once, but still pays the per-iteration BF16 scale multiply and matmul cost, so the hardware speedup is not inflated by Python-side packing overhead.
- The standalone runners lock GPU clocks through the repo harness by default. Use `--no-lock-gpu-clocks` only for quick local iteration when repeatability is not the goal.
- The optimized benchmark validates correctness during setup by default. Set `AISP_BLOCK_SCALING_SKIP_VERIFY=1` only when you are explicitly doing timing-only sweeps.
- The optimized path requires a Blackwell-class GPU (`sm100+`). The software baseline still requires CUDA because the lab is meant to be compared on the same device.
