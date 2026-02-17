# Plan

- [x] Add CUDA extension with three kernels:
  - `kernel_bias_relu` (separate)
  - `kernel_residual_add` (separate)
  - `kernel_bias_relu_residual_fused`
- [x] Implement baseline path calling separate kernels.
- [x] Implement fused path calling one kernel.
- [x] Add script to benchmark baseline vs fused, verify correctness, and dump raw metrics to `benchmark_output.txt`.
- [x] Run benchmark and document measured speedup.

# Runlog
- 2026-02-16T16:33:14Z: initial plan drafted.
- 2026-02-16T16:36:55Z: implementation completed (extension + benchmark script).
- 2026-02-16T16:36:37Z: benchmark executed and artifact written.
