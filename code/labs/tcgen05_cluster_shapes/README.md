# tcgen05 Cluster Shapes

## Summary
Exploratory lab for the CUTLASS tcgen05 dense-GEMM cluster-shape question: when does `1-SM` beat `2-SM`, and is `4-SM` even available in the current SM100 CUTLASS schedule set?

## Why This Is A Lab
This is not a canonical baseline/optimized chapter pair.
It is a shape-and-schedule probe that benchmarks the shipped `1-SM` and `2-SM` wrappers and reports whether `4-SM` is supported by the current CUTLASS headers.
It uses the repo's harness clock-lock helper by default so the direct script is still measured under the same app-clock discipline as the benchmark harness.

## What It Benchmarks
- `1sm`: [`ch10/tcgen05_warp_specialized_cutlass.cu`](../../ch10/tcgen05_warp_specialized_cutlass.cu)
- `2sm`: [`ch10/tcgen05_warpgroup_specialized.cu`](../../ch10/tcgen05_warpgroup_specialized.cu)
- `4sm`: capability check only unless a future wrapper is added

## Repro
```bash
python -m labs.tcgen05_cluster_shapes.run_cluster_shape_sweep
python -m labs.tcgen05_cluster_shapes.run_cluster_shape_sweep --m 4096 --n 4096 --k 1024 --iterations 10 --warmup 3
python -m labs.tcgen05_cluster_shapes.run_cluster_shape_sweep --json-out artifacts/tcgen05_cluster_shapes/latest.json
```

## Notes
- The current CUTLASS SM100 dense GEMM dispatch policy only exposes dedicated `1-SM` and `2-SM` warp-specialized schedules.
- If `4-SM` support is absent, this lab reports that directly instead of inventing a synthetic path.
- Use `--no-lock-gpu-clocks` only for local debugging; it is not the default measurement path.
