# Chapter 11 Benchmark Pair Validity Review

**Review date:** 2026-03-17  
**Scope:** All canonical baseline_*/optimized_* pairs in `/home/cfregly/ai-performance-engineering/code/ch11/`

---

## Summary

| Pair | Result | Notes |
|------|--------|-------|
| `streams` | PASS | Same copy/compute overlap workload; optimized overlaps H2D and compute. |
| `stream_ordered` | PASS | Same allocator workload; optimized switches to stream-ordered allocation. |
| `tensor_cores_streams` | PASS | Same segmented GEMM work; optimized overlaps copy and GEMM across streams. |
| `adaptive_streams` | PASS | Legacy target name; docs now state the optimized path is fixed round-robin multi-stream overlap, not runtime-adaptive scheduling. |
| `gemm_streams` | PASS | Legacy target name; docs now state the workload is the shared copy+elementwise overlap base, not GEMM. |
| `stream_ordered_kv_cache` | PASS | Same overlap workload; target name remains conceptual. |
| `distributed_streams` | PASS | Same overlap workload; "distributed" refers to the overlap pattern, not multi-GPU execution. |
| `warp_specialized_multistream` | PASS | CUDA binary pair with matching workload params. |
| `warp_specialization_multistream` | PASS | Same stream-overlap workload with identical segment counts. |
| `warp_specialized_two_pipelines_multistream` | FLAG | Optimized path still loads the extension and DSMEM guard in `__init__`, unlike the baseline's `setup()` load path. |
| `warp_specialized_two_pipelines_driver` | PASS | CUDA binary pair with matching workload params. |

---

## Remediation Notes

- The shared stream base now uses the same harness iteration count on both sides, so the chapter no longer has the baseline `iterations=20` versus optimized `iterations=16` asymmetry.
- `adaptive_streams` and `gemm_streams` remain stable public target names, but their module docstrings and README text now describe the actual copy+compute overlap workload rather than implying a different scheduling algorithm or a GEMM kernel.
- No workload-equivalence issues remain in the stream-overlap family after the iteration-parity fix.

---

## Remaining Follow-Up

1. Move the optimized extension load in `warp_specialized_two_pipelines_multistream` from `__init__` into `setup()` to match the baseline lifecycle and avoid discovery-time failures on unsupported hosts.
