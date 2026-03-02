# Lab - TensorRT-LLM Phi-3.5-MoE

## Summary
Compares Hugging Face Transformers generation against TensorRT-LLM inference for Phi-3.5-MoE, using identical prompts and greedy decoding.

## Learning Goals
- Measure runtime speedups from TensorRT-LLM kernels and engine optimizations.
- Validate output logits for a fixed prompt and decoding length.
- Exercise TRT-LLM generation APIs in a harness-comparable workflow.

## Files
| File | Description |
| --- | --- |
| `baseline_trtllm_phi_3_5_moe.py` | Transformers baseline (eager attention). |
| `optimized_trtllm_phi_3_5_moe.py` | TensorRT-LLM optimized generation. |
| `trtllm_common.py` | Shared prompt/token helpers. |

## Running
```bash
# Baseline vs optimized (pass engine path for TRT-LLM)
python -m cli.aisp bench run --targets labs/trtllm_phi_3_5_moe \
  --target-extra-arg labs/trtllm_phi_3_5_moe:optimized_trtllm_phi_3_5_moe="--engine-path /path/to/engine.plan"
```

## Notes
- Requires local Phi-3.5-MoE weights at `phi-3.5-moe/original` (override with `--model-path` or `AISP_PHI35_MOE_MODEL_PATH`).
- Optimized path also requires a TensorRT-LLM engine plan/directory (pass `--engine-path` or set `AISP_PHI35_MOE_ENGINE_PATH`).
- Recommended source: `microsoft/Phi-3.5-MoE-instruct`.
- TRT-LLM must be built with `output_generation_logits=True` support; the benchmark validates `generation_logits`.
- Keep TRT-LLM precision aligned with the baseline (e.g., FP16) to pass output verification.
- `trtllm_common.load_trtllm_runtime()` loads the runtime submodule directly and calls `_common._init()` to avoid full-package imports.
- Required runtime deps: TensorRT libs + python bindings, `tensorrt-llm`, `nvtx`, `mpi4py`, `nvidia-modelopt`, `onnx-graphsurgeon`, `h5py`, `pulp`, `soundfile`.

## Related Chapters
- **Ch16**: Production inference systems and engine comparisons.
- **Ch18**: TensorRT-LLM and NVFP4 workflows.
