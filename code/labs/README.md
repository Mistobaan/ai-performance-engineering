# Labs

Labs are end-to-end optimization stories that combine multiple chapter techniques (kernel + runtime + system) into realistic workflows. Each lab directory is self-contained and typically includes `baseline_*.py` / `optimized_*.py` benchmark pairs plus a `README.md` with prerequisites and run commands.

## Lab Index (Best-Effort Chapter Mapping)

| Lab | Summary | Suggested Chapters |
| --- | --- | --- |
| `labs/nvfp4_gemv/` | GPUMODE `nvfp4_gemv` challenge workspace | ch06, ch10 |
| `labs/nvfp4_gemm/` | GPUMODE `nvfp4_gemm` challenge workspace | ch06, ch09, ch10 |
| `labs/async_input_pipeline/` | Async CPU→GPU input overlap | ch02, ch05, ch11 |
| `labs/block_scaling/` | Blackwell hardware-supported block scaling with direct CUTLASS vs PyTorch microbenchmarks | ch06, ch09 |
| `labs/blackwell_matmul/` | Matmul suite focused on Blackwell | ch06, ch09, ch10 |
| `labs/cudnn_sdpa_bench/` | cuDNN SDPA benchmarking | ch10, ch18 |
| `labs/custom_vs_cublas/` | Custom kernel vs cuBLAS parity | ch06, ch09 |
| `labs/cutlass_profiler_kernel_selector/` | CUTLASS profiler-based kernel selection | ch06, ch09 |
| `labs/decode_optimization/` | Decoder hot-path optimization | ch18, ch19 |
| `labs/dynamic_router/` | Dynamic prefill/decode routing | ch17, ch19 |
| `labs/flashattention4/` | FlashAttention-4 pipeline co-design | ch10, ch18 |
| `labs/flashattention_gluon/` | FlashAttention experimentation | ch18 |
| `labs/flashinfer_attention/` | FlashInfer block-sparse attention lab | ch16 |
| `labs/flexattention/` | FlexAttention harness and sweeps | ch18 |
| `labs/fullstack_cluster/` | Full-stack cluster + DSMEM workflows | ch10 |
| `labs/kv_cache_compression/` | KV-cache compression/quantization | ch18, ch19 |
| `labs/kv_optimization/` | KV-cache performance optimization | ch15, ch18, ch19 |
| `labs/moe_cuda/` | CUDA MoE decode toolkit | ch06, ch10, ch15 |
| `labs/moe_optimization_journey/` | MoE optimization narrative | ch15, ch19 |
| `labs/moe_parallelism/` | MoE parallelism planning | ch04, ch15 |
| `labs/nanochat_fullstack/` | End-to-end inference stack (nanochat) | ch16 |
| `labs/occupancy_tuning/` | Triton occupancy/schedule sweeps | ch08, ch14 |
| `labs/persistent_decode/` | Persistent decode + TMA prefill | ch10, ch11 |
| `labs/python_concurrency/` | Python concurrency control-plane playbook (`asyncio`, retries, idempotency, hybrid pipelines) | ch03, ch11, ch16 |
| `labs/real_world_models/` | Real-world model optimization playbook | ch20 |
| `labs/speculative_decode/` | Speculative decoding | ch15, ch18 |
| `labs/trtllm_phi_3_5_moe/` | TensorRT-LLM Phi-3.5-MoE comparison | ch16, ch18 |
| `labs/train_distributed/` | Distributed training workflows | ch03, ch04 |
| `labs/uma_memory/` | UMA / unified memory diagnostics | ch02, ch07 |

Notes:
- “Suggested Chapters” is a best-effort map; when a lab has an explicit mapping, it is documented in that lab’s `README.md`.
- `labs/common/` contains shared helpers and is not a standalone lab target.
