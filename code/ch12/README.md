# Chapter 12 - CUDA Graphs & Dynamic Workloads

## Summary
Covers modern CUDA Graph capabilities-conditional capture, graph memory tuning, dynamic parallelism, and work queues-to keep irregular workloads performant without per-launch overhead.

## Problem
Chapter 12 is where launch overhead and dynamic work management have to justify themselves with measured wins. The useful question is not "can we capture a graph?" but "which graph or dynamic-work techniques actually reduce the real runtime once correctness and workload shape stay fixed?"

## Baseline Path
- eager launches and more CPU-visible scheduling work
- less reuse of graph capture or GPU-resident work management
- easy to inspect, but often too expensive for irregular steady-state workloads

## Optimized Path
- CUDA Graph replay where the steady-state workload is stable enough
- fused or GPU-resident queueing/dispatch where it actually removes launch overhead
- measured through the shared harness instead of hand-timed one-off scripts

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `cuda_graphs` | `529.532 ms` | `125.874 ms` | `4.21x` | graph replay instead of repeated eager launch overhead |
| `kernel_fusion` | `1.776 ms` | `0.654 ms` | `2.72x` | fewer launches through fused graph-friendly execution |
| `work_queue` | `2.100 ms` | `0.442 ms` | `4.75x` | GPU-resident work queue path |

This chapter is useful because it separates "graphs help" from "graphs help on a workload that is actually stable enough to benefit." The work-queue target also keeps the chapter from being only about graph replay.

## Profiler Evidence
Use deep-dive runs when you want hard evidence for launch reduction and work scheduling changes:

```bash
python -m cli.aisp bench run --targets ch12:cuda_graphs --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch12:kernel_fusion --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch12:work_queue --profile deep_dive --single-gpu
```

Those targets answer slightly different questions:
- `cuda_graphs`: replay payoff
- `kernel_fusion`: launch-count reduction
- `work_queue`: GPU-side dynamic dispatch effectiveness

## Repro Commands
```bash
python -m ch12.compare
python -m cli.aisp bench list-targets --chapter ch12
python -m cli.aisp bench run --targets ch12 --profile minimal
python -m cli.aisp bench run --targets ch12:cuda_graphs --profile deep_dive --single-gpu
```

## Learning Goals
- Capture steady-state workloads into CUDA Graphs and study the delta vs eager launches.
- Use conditional nodes and graph memory pools for adaptive pipelines.
- Experiment with device-side launches (dynamic parallelism) to reduce CPU involvement.
- Implement GPU-resident work queues and uneven partition schedulers.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_cuda_graphs.py`, `optimized_cuda_graphs.py`, `baseline_cuda_graphs_conditional*.cu`, `optimized_cuda_graphs_conditional*.cu` | Graph capture demos that evolve from simple replay to conditional and DSM-aware execution. |
| `baseline_graph_bandwidth.{py,cu}`, `optimized_graph_bandwidth.{py,cu}`, `baseline_kernel_launches.py`, `optimized_kernel_launches.py` | Launch- and bandwidth-focused studies illustrating how graphs reduce CPU overhead. |
| `baseline_dynamic_parallelism_host.cu`, `baseline_dynamic_parallelism_device.cu`, `optimized_dynamic_parallelism_host.cu`, `optimized_dynamic_parallelism_device.cu`, `dynamic_parallelism_sm121/` | Device-side launch samples showing when dynamic parallelism helps or hurts. |
| `baseline_work_queue.{py,cu}`, `optimized_work_queue.{py,cu}`, `work_queue_common.cuh` | GPU work queues for irregular batch sizes, including NVTX instrumentation. |
| `baseline_uneven_partition.cu`, `optimized_uneven_partition.cu`, `baseline_uneven_static.cu`, `optimized_uneven_static.cu` | Uneven workload partitioners that rebalance CTA assignments at runtime. |
| `baseline_kernel_fusion.py`, `optimized_kernel_fusion.py`, `kernel_fusion_cuda_demo.cu` | Kernel fusion exercises within graph capture so you can remove CPU synchronization entirely. (`kernel_fusion_cuda_demo.cu` is a standalone tool; not a benchmark target.) |
| `compare.py`, `cuda_extensions/`, `expectations_{hardware_key}.json` | Harness entry, extension stubs, and expectation thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch12.compare
python -m cli.aisp bench list-targets --chapter ch12
python -m cli.aisp bench run --targets ch12 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python optimized_cuda_graphs.py --iterations 100` should report lower wall-clock time than the baseline while matching outputs.
- Device-side dynamic parallelism samples emit warnings on unsupported hardware, ensuring you only trust data from GPUs with the feature enabled.
- `python optimized_work_queue.py --trace` exposes balanced dequeue times across CTAs when compared to the baseline's stragglers.

## Notes
- `cuda_graphs_workload.cuh` holds reusable graph capture helpers when you want to wrap your own kernels.
- `helper_*.cu` files contain host/device glue for the dynamic-parallelism case studies-copy them when bootstrapping new experiments.
- For a smaller supporting lab that keeps Chapter 12's CUDA Graph story primary while adding fused reduction and padding-aware projection examples, see `labs/training_hotpath`.
