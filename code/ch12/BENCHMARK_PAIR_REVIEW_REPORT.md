# Ch12 Benchmark Pair Validity Review

**Chapter 12**: Dynamic Scheduling, CUDA Graphs, and Device-Initiated Kernel Orchestration  
**Review date**: 2026-03-17

---

## Summary

| Pair | Status | Notes |
|------|--------|------|
| `work_queue` | PASS | Same workload and same reset pattern. |
| `cuda_graphs` | PASS | Same math; optimized captures the launch sequence once and replays it. |
| `graph_bandwidth` | PASS | Same copy workload with graph capture versus fresh launches. |
| `kernel_launches` | PASS | Same 1000-stage add/mul/ReLU chain per harness iteration; optimized graph replay now starts from immutable input and no longer compounds work. |
| `kernel_fusion` | PASS | Same math; optimized fuses the kernel sequence. |
| `uneven_static` | PASS | CUDA binary pair with matching workload params. |
| `uneven_partition` | PASS | CUDA binary pair with matching workload params. |
| `dynamic_parallelism_host` | PASS | CUDA binary pair with matching workload params. |
| `dynamic_parallelism_device` | PASS | CUDA binary pair with matching workload params. |
| `cuda_graphs_conditional` | PASS | CUDA binary pair with matching workload params. |
| `cuda_graphs_conditional_enhanced` | PASS | CUDA binary pair with matching workload params. |
| `graph_conditional_runtime` | PASS | Same runtime-controlled work graph; optimized captures it once. |
| `nvfp4_mlp` | PASS | Shared benchmark class with matched config. |

---

## Key Remediation

- `kernel_launches` was the only fairness blocker in the canonical chapter surface. The baseline no longer performs an extra hot-path clone, and the optimized path now captures a deterministic graph over immutable input buffers, so each harness iteration performs the same amount of work on both sides.
- Chapter-level pair validation is now clean: `python -m core.scripts.validate_benchmark_pairs --chapter ch12` reports 17/17 valid pairs with no signature mismatches.

---

## Informational Variants

- `optimized_kernel_fusion_llm_reuse_static_tensor_and_simplify_setup.py`
- `optimized_kernel_fusion_llm_persistent_buffer_and_stream_friendly_setup.py`
- `optimized_kernel_fusion_llm_dedicated_stream_and_prefetch_for_blackwell.py`

These remain alternative optimized implementations for the same `kernel_fusion` control pair and do not change the canonical pair-validity result.
