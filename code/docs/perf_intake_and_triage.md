# Performance Intake & Triage Bundle

This repo already collects benchmark artifacts. The missing step for most investigations is turning those artifacts into a repeatable method instead of an ad hoc debug session.

Use this doc with [`docs/benchmark_methodology.md`](/home/cfregly/ai-performance-engineering/code/docs/benchmark_methodology.md). The methodology tells you which benchmark layer to run and what evidence is required. This doc is the fast intake + first-pass collection path.

## Quick Actions
- Fill the intake in [`templates/performance_intake.yaml`](/home/cfregly/ai-performance-engineering/code/templates/performance_intake.yaml).
- Freeze the workload in [`templates/benchmark_workload_spec.yaml`](/home/cfregly/ai-performance-engineering/code/templates/benchmark_workload_spec.yaml).
- If the run needs to survive CI, scheduling automation, or external publication review, copy [`templates/benchmark_run.yaml`](/home/cfregly/ai-performance-engineering/code/templates/benchmark_run.yaml) and validate it with `python -m core.scripts.validate_benchmark_run --file <your-file>.yaml`.
- Choose the benchmark layer: `micro`, `component`, or `end_to_end`.
- Run the triage bundle to gather a clean baseline: `core/scripts/profiling/perf_triage_bundle.sh --output-root ./artifacts/runs --tag baseline -- <your command>`.
- Pick 2-3 experiments that change one variable at a time and rerun with the same workload spec.

## One-Page Intake
Copy [`templates/performance_intake.yaml`](/home/cfregly/ai-performance-engineering/code/templates/performance_intake.yaml) and fill it for the workload under test. The fields cover KPIs, workload shape, SLOs, benchmark layer, comparison axis, current baseline, and guardrails.

Then copy [`templates/benchmark_workload_spec.yaml`](/home/cfregly/ai-performance-engineering/code/templates/benchmark_workload_spec.yaml) and freeze the actual benchmark contract:
- model and weights source
- sequence-length mix
- precision policy
- batching policy
- concurrency model
- image digest
- driver/CUDA/NCCL/runtime versions
- topology and scheduler path
- outlier/confidence policy

If the result is going to be scheduled declaratively or cited outside engineering, also copy [`templates/benchmark_run.yaml`](/home/cfregly/ai-performance-engineering/code/templates/benchmark_run.yaml). That file adds the layer stack, distributed diagnosis policy, provenance requirements, and publication-vs-realism execution mode.

Do not compare runs until the intake and workload spec are filled. Do not publish or automate a run until the `BenchmarkRun` file validates cleanly.

## Choose The Right Layer

- `micro`: isolate a subsystem such as NCCL, disk, PCIe, NVLink, or tensor-core throughput. Use `python -m cli.aisp benchmark ...` or the cluster microbench scripts.
- `component`: isolate serving, dataloaders, train-step execution, job startup, or a single benchmark pair. Use `python -m cli.aisp bench run --targets ... --profile minimal|deep_dive`.
- `end_to_end`: measure a realistic workflow or release gate. Use `python -m cli.aisp bench run-tier1` or `python -m cli.aisp cluster common-eval --preset ...`.

## 30-Minute Triage Bundle
The bundle captures hardware/software facts plus a short run with either Nsight Systems (when present) or `nvidia-smi dmon` as a fallback.

Usage:
```bash
# Snapshots only (no runtime command)
core/scripts/profiling/perf_triage_bundle.sh --output-root ./artifacts/runs

# Snapshots + runtime capture for a representative command
core/scripts/profiling/perf_triage_bundle.sh --output-root ./artifacts/runs --tag baseline --nsys -- \
  python ch10/baseline_matmul.py --batch-size 32
```

What it does:
- Creates `artifacts/runs/perf_triage_<host>_<timestamp>[_<tag>]` with GPU/CPU/memory/storage/network snapshots, CUDA/PyTorch versions, and manifest metadata.
- If Nsight Systems is available (and not disabled), runs the provided command under `nsys profile -t cuda,nvtx,osrt,cudnn,cublas` and emits both the `.nsys-rep` and a text summary.
- Otherwise, samples `nvidia-smi dmon` while the command runs and stores a CSV timeseries for SM%, mem BW, power, and utilization.
- Packs everything into a `.tgz` for sharing.

The triage bundle is for decomposition, not publication. Use it to classify the bottleneck and decide which benchmark layer and profiler depth come next.

### PyTorch Compiler Diagnostics (optional)
If you use `torch.compile`, enable graph-break diagnostics before the run:
```bash
export TORCH_LOGS="+dynamo,+inductor,perf_hints,output_code"
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_BENCHMARK_KERNEL=1
```
Inside Python you can inspect breaks with:
```python
torch._dynamo.explain(model)(*example_inputs)
```

## High-ROI Experiment Menu (run A/B with identical inputs)
- **Inference:** continuous batching and scheduling, speculative decoding (EAGLE/Medusa), quantization (FP8/INT8/INT4 with quality gates), grammar/constraint overhead checks, topology-aware placement (NVLink/NVSwitch first), and engine comparisons (vLLM/SGLang/TensorRT-LLM).
- **Training:** dataloader throughput (prefetch, pinned memory, CPU workers), `torch.compile` with graph-break fixes or regional compilation, overlap of compute/communication (bucket sizing + NCCL env), parallelism mix (DP/FSDP/TP/PP/MoE balance), mixed precision + fused ops, and small autotuning sweeps for batch/seq length/kernel configs.
- **Both:** measure goodput (useful GPU work / wall time), track throughput/$ and throughput/W across instance types, and keep lightweight continuous monitors (DCGM/Prometheus) to catch regressions.

## What to Return
- Completed intake YAML.
- Completed workload spec YAML.
- The triage bundle `.tgz` containing system snapshots and Nsight/dmon outputs.
- A/B table with the variable under test called out explicitly, plus throughput, p50/p99 latency, GPU SM%, NIC GB/s, and any accuracy deltas.
- Written explanation of whether the bottleneck is compute-bound, comm-bound, input-bound, or control-plane-bound.
