# Lab - FlashAttention-4 Pipeline Co-Design

## Summary
Recreates the practical shape of the Colfax FlashAttention-4 article on top of this repo's harness. The baseline path uses eager FlexAttention, which materializes scores and behaves like the unfused scalar-heavy path the article is trying to escape. The optimized path compiles a Blackwell-friendly kernel configuration, prefers the experimental FLASH backend when available, and falls back to a compiled TMA path when the local toolchain cannot lower the FLASH backend cleanly. The default benchmark focuses on `ALiBi`, which is one of the Flex-only patterns the upstream FA4 integration explicitly targets and is stable on this local stack.

## Learning Goals
- Measure how much of the win comes from moving from eager score materialization to a fused, compiled attention kernel.
- Exercise FA4-style workloads with non-trivial score modifiers such as ALiBi and soft-capped logits, and optionally probe sliding-window masks on a best-effort basis.
- Inspect how TMA-oriented kernel options change the provider selected on Blackwell.
- Use the coarse pipeline model to reason about why overlap matters more as tensor-core throughput scales faster than scalar/SFU throughput.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flashattention4.py` | Eager FlexAttention baseline that materializes the score matrix. |
| `optimized_flashattention4.py` | Compiled, Blackwell-oriented path that tries the FLASH backend first, then falls back to compiled FlexAttention+TMA. |
| `flashattention4_common.py` | Shared QKV generation, mask/score-mod builders, and provider resolution. |
| `pipeline_model.py` | Coarse latency model for FA4-style overlap and asymmetric scaling. |
| `tflops_microbench.py` | Clock-locked TFLOPs/s microbenchmark for Colfax/PyTorch-style backend comparisons. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/flashattention4
python -m cli.aisp bench run --targets labs/flashattention4 --profile minimal
python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_alibi --profile minimal
python -m cli.aisp bench run --targets labs/flashattention4:best_available_attention_dense --profile minimal
python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_softcap --profile minimal
python labs/flashattention4/pipeline_model.py --tiles 32 --tensor-core-scale 4 --scalar-scale 2
python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal alibi
python labs/flashattention4/tflops_microbench.py --preset peak_probe --mode dense causal --backends flash_backend triton_flex cudnn_sdpa
```
- Harness workflows use explicit targets such as `flashattention4_dense`, `flashattention4_causal`, `flashattention4_alibi`, `flashattention4_softcap`, `flashattention4_windowed`, `flashattention4_alibi_windowed`, and the matching `best_available_attention_*` variants.
- On the local `torch 2.9.1+cu130` build, `windowed` and `alibi_windowed` are experimental: the optimized path can produce non-finite outputs on a fresh compile even though upstream FA4 supports sliding-window patterns.
- `tflops_microbench.py` locks GPU clocks through `core.harness.benchmark_harness.lock_gpu_clocks()` by default; use `--no-lock-gpu-clocks` only for local debugging.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/flashattention4 --profile minimal` shows a large baseline-to-optimized gap on B200 because the eager path materializes scores while the compiled path does not.
- `python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_alibi --profile minimal` succeeds on a cold-start process and exercises the FA4 score-mod path without relying on env vars.
- `python -m cli.aisp bench run --targets labs/flashattention4:best_available_attention_dense --profile minimal` gives the clearest absolute-performance path for standard attention on this stack.
- `python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_windowed --profile minimal` and `labs/flashattention4:flashattention4_alibi_windowed` remain explicit experimental probes; treat failures there as a PyTorch/FA4 integration limitation on this stack rather than as a lab bug.
- `python labs/flashattention4/pipeline_model.py --tiles 64 --tensor-core-scale 4 --scalar-scale 2` reports a larger overlap speedup than a balanced-scaling scenario.
- `python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal alibi` runs the public-shape comparison against the local FLASH backend, the local Triton-style proxy, and cuDNN where supported.
- `python labs/flashattention4/tflops_microbench.py --preset peak_probe --mode dense causal --backends flash_backend triton_flex cudnn_sdpa` checks whether a larger compute-bound shape moves the local stack toward the Colfax/PyTorch reported envelope.

## TFLOPs/s Microbenchmark
Use `tflops_microbench.py` when you want something closer to the published Colfax and PyTorch comparisons than the harness benchmark pair. The harness pair is intentionally end-to-end and compares eager score materialization against a fused kernel; the microbenchmark instead compares backend implementations on the same attention workload.

| Published comparison target | Local command | Notes |
| --- | --- | --- |
| Colfax B200 BF16 forward envelope (`1605 TFLOPs/s`, up to `1.3x` over cuDNN 9.13, up to `2.7x` over Triton) | `python labs/flashattention4/tflops_microbench.py --preset peak_probe --mode dense causal --backends flash_backend triton_flex cudnn_sdpa` | Uses a larger shape to push the local stack harder. |
| PyTorch GB200 standard-attention forward envelope (`1.6x-3.2x` over Triton) | `python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal` | Uses the public blog shape `B=2, H=8, S=2048, D=128`. |
| PyTorch GB200 ALiBi forward envelope (`1.2x-2.1x` over Triton) | `python labs/flashattention4/tflops_microbench.py --preset public_blog --mode alibi --backends flash_backend triton_flex flex_tma` | cuDNN SDPA is not applicable to ALiBi. |

The FLOP accounting matches the common SDPA forward convention used in vendor/blog comparisons:
`forward_flops = 4 * batch * heads * head_dim * nonmasked_attention_elements`

- For `dense`, `alibi`, and `softcap`, `nonmasked_attention_elements = q_seq_len * kv_seq_len`.
- For `causal`, `windowed`, and `alibi_windowed`, only the unmasked score matrix entries are counted.
- `triton_flex` is the closest local proxy for the blog's Triton baseline: compiled FlexAttention with `USE_TMA=False`.

## Current Local Results
These measurements were taken on March 5, 2026 on the current local `torch 2.9.1+cu130` stack with harness clock locking enabled. This host is still virtualized, so treat the numbers as directional rather than canonical.

### Public Blog Shape (`B=2, H=8, S=2048, D=128`)
| Mode | Backend | Median (ms) | TFLOPs/s | Flash vs Triton | Flash vs cuDNN | Published check |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `dense` | `flash_backend` | 0.224 | 153.6 | `1.02x` | `0.40x` | Outside Colfax and PyTorch ranges |
| `dense` | `triton_flex` | 0.229 | 150.1 | `1.00x` | `0.39x` | Local Triton-style proxy |
| `dense` | `cudnn_sdpa` | 0.090 | 382.5 | `2.55x` | `1.00x` | Local cuDNN leader |
| `causal` | `flash_backend` | 0.238 | 72.1 | `14.84x` | `0.37x` | Beats local Triton-style proxy, still far below cuDNN |
| `causal` | `triton_flex` | 3.538 | 4.9 | `1.00x` | `0.02x` | Local Triton-style proxy collapses on this stack |
| `causal` | `cudnn_sdpa` | 0.088 | 195.5 | `40.25x` | `1.00x` | Local cuDNN leader |
| `alibi` | `flash_backend` | 6.221 | 5.5 | `1.02x` | n/a | Outside PyTorch ALiBi range |
| `alibi` | `triton_flex` | 6.323 | 5.4 | `1.00x` | n/a | Local Triton-style proxy |
| `alibi` | `flex_tma` | 6.169 | 5.6 | `1.03x` | n/a | Slightly ahead locally, still not near published envelope |

### Peak Probe Shape (`B=8, H=16, S=4096, D=128`)
| Mode | Backend | Median (ms) | TFLOPs/s | % of Colfax 1605 | Flash vs Triton | Flash vs cuDNN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dense` | `flash_backend` | 3.576 | 307.5 | 19.2% | `1.01x` | `0.34x` |
| `dense` | `triton_flex` | 3.614 | 304.2 | 19.0% | `1.00x` | `0.34x` |
| `dense` | `cudnn_sdpa` | 1.222 | 899.8 | 56.1% | `2.96x` | `1.00x` |
| `causal` | `flash_backend` | 2.264 | 242.9 | 15.1% | `0.97x` | `0.36x` |
| `causal` | `triton_flex` | 2.200 | 250.0 | 15.6% | `1.00x` | `0.37x` |
| `causal` | `cudnn_sdpa` | 0.814 | 675.1 | 42.1% | `2.70x` | `1.00x` |

The local conclusion is straightforward: this stack does not currently reproduce the published Colfax or PyTorch FlashAttention-4 envelope. The larger probe rules out a pure small-shape saturation explanation because the local FLASH path still tops out at `307.5 TFLOPs/s` on dense and `242.9 TFLOPs/s` on causal, well below both Colfax's `1605 TFLOPs/s` peak and the local cuDNN path.

## Notes
- Primary source: [Colfax Research, "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/).
- Integration source: [PyTorch, "FlexAttention + FlashAttention-4: Fast and Flexible"](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/).
- Colfax reports up to `1605 TFLOPs/s` on B200 BF16 at roughly `71%` utilization, plus up to `1.3x` over cuDNN 9.13 and `2.7x` over Triton for forward passes.
- The PyTorch post reports `1.6x-3.2x` forward speedup over Triton for standard dense/causal attention on GB200, `1.2x-2.1x` for ALiBi, and `1.4x-2.1x` for sliding-window attention.
- On the local `torch 2.9.1+cu130` stack, the experimental FLASH backend needs a quoted backend literal when passed through `kernel_options`; the lab hides that workaround and falls back automatically if the backend still fails.
- The lab pins float32 accumulation to IEEE mode (`TF32` disabled) because the current `sm_100` FLASH/FlexAttention lowering produced non-finite outputs for this workload under TF32 accumulation.
- The upstream PyTorch FA4 integration supports `ALiBi`, sliding-window masks, soft-capping, and combinations of those patterns; this lab exposes all of them, but the cold-start-stable benchmark default is `ALiBi`.
- The optimized path is intentionally provider-aware. Check the benchmark metrics or NVTX range name to see whether the run used `flash_backend`, `flex_tma`, or the plain compiled FlexAttention fallback.
