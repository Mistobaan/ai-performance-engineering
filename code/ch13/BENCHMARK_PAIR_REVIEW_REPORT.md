# Ch13 Benchmark Pair Validity Review Report

**Date:** 2026-03-17  
**Scope:** All baseline_*.py / optimized_*.py pairs in `/home/cfregly/ai-performance-engineering/code/ch13/`

---

## Summary

| Status | Count | Notes |
|--------|-------|-------|
| PASS | 27 | Canonical same-work pairs validated cleanly. |
| INFO | 2 | Informational compound variants retained but excluded from canonical speedup claims. |
| REQUIRES >=2 GPUs | 2 | `context_parallel_multigpu`, `expert_parallel_multigpu` need a multi-GPU host for full load-time validation. |

---

## Remediated Canonical Pairs

| Pair | Status | Notes |
|------|--------|-------|
| `regional_compile` | PASS | Both sides now run the same BF16 workload. Baseline compiles the full block; optimized compiles only the MLP region. |
| `matmul_pytorch` | PASS | Both sides now run the same FP16 compute path. The canonical story is eager versus compiled PyTorch, not a precision drop. |
| `kv_cache_naive` | PASS | Both sides now use the same token-by-token decode loop and backend semantics. The optimized path isolates paged-cache allocation rather than a blockwise algorithm change. |
| `torchao_quantization` | PASS | Canonical pair is quantization-only. |
| `precisionfp8_pad_inner` | PASS | Each timed iteration now performs a single forward pass, and verification reuses that output. |
| `dataloader_default` | PASS | Same dataset and batch semantics; optimized path improves loader configuration only. |
| `attention_standard` | PASS | Same attention math; optimized path uses the higher-level backend implementation. |
| `autograd_standard` | PASS | Same model and optimizer state; optimized path reduces framework overhead. |

---

## Informational Variants

| Target | Role | Notes |
|--------|------|-------|
| `torchao_quantization_compiled` | INFO | Preserves the old quantization-plus-compile story without polluting the canonical quantization pair. |
| `kv_cache_naive_flash_blockwise` | INFO | Preserves the old blockwise + FlashAttention story as an explicitly noncanonical exemplar. |

---

## Additional Notes

- `memory_profiling` now runs the same FP32 workload on both sides. Treat it as a memory-goal checkpointing study, not as a throughput headline pair.
- Chapter-level pair validation is clean on this host for all single-GPU pairs: `python -m core.scripts.validate_benchmark_pairs --chapter ch13` reports 29 valid pairs and only the expected `>=2 GPU` load guards for `context_parallel_multigpu` and `expert_parallel_multigpu`.
- `optimized_kv_cache_naive_pool` remains a valid same-work alternative to the canonical paged-cache pair: same token-by-token loop, different allocation strategy.

---

## Follow-Up

1. Run the two multi-GPU chapter targets on a `>=2 GPU` host when you want full end-to-end validation for the distributed pair surfaces.
