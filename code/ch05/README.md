# Chapter 5 - Storage and IO Optimization

## Summary
Focuses on feeding GPUs efficiently: tune DataLoader workers, vectorize preprocessing, overlap IO with compute, and adopt GPUDirect Storage when NVMe traffic becomes the bottleneck.

## Problem
Chapter 5 exists because GPUs do not care how elegant a kernel is if the input path is late. The useful question is which storage and preprocessing changes actually turn an IO-bound workload into a compute-bound one.

## Baseline Path
- CPU-heavy preprocessing and unvectorized parsing
- storage paths that serialize work on the host
- dataloading behavior that leaves visible GPU idle time

## Optimized Path
- vectorized preprocessing and overlap between IO and compute
- tuned worker/prefetch settings
- GPUDirect Storage or cleaner staging paths where the platform supports them

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `vectorization` | `3.861 ms` | `0.053 ms` | `72.64x` | Python-heavy preprocessing becomes vectorized |
| `storage_cpu` | `111.652 ms` | `53.898 ms` | `2.07x` | storage path stops starving the device |
| `ai` | `63.724 ms` | `47.702 ms` | `1.34x` | streaming/inference pipeline overlaps IO better |

The headline win here is often preprocessing, not raw storage hardware. That is why both vectorization and storage-path examples belong in the same chapter.

## Profiler Evidence
Use deep-dive runs to distinguish host-side preprocessing waste from actual storage limits:

```bash
python -m cli.aisp bench run --targets ch05:vectorization --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch05:storage_cpu --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch05:ai --profile deep_dive --single-gpu
```

The expected evidence is:
- `vectorization`: dramatically less CPU time in preprocessing
- `storage_cpu`: fewer long idle gaps between batches
- `ai`: better overlap between read/decode work and compute

## Repro Commands
```bash
python -m ch05.compare
python -m cli.aisp bench list-targets --chapter ch05
python -m cli.aisp bench run --targets ch05 --profile minimal
python -m ch05.gds_cufile_minimal /tmp/gds_test_file.bin 1073741824 --generate
```

## Learning Goals
- Detect IO stalls via harness metrics and restructure pipelines to keep GPUs busy.
- Tune PyTorch DataLoader knobs (workers, prefetch, pinned memory) for large-batch training.
- Evaluate GPUDirect Storage paths vs traditional CPU-mediated reads.
- Benchmark remote storage and distributed data reading strategies.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_storage_cpu.py`, `optimized_storage_cpu.py` | Single-node dataloader comparison covering worker count, pinned memory, and caching strategies. |
| `baseline_vectorization.py`, `optimized_vectorization.py` | Vectorized parsing and memory-map examples that remove Python loops from preprocessing. |
| `baseline_ai.py`, `optimized_ai.py`, `storage_io_optimization.py` | LLM-style token pipelines showcasing overlapping compute with streaming reads and prefetch. |
| `baseline_distributed.py`, `optimized_distributed.py` | Single-GPU host-staged reduction vs on-device reduction. |
| `baseline_distributed_multigpu.py`, `optimized_distributed_multigpu.py` | Actual multi-GPU reduction baseline (CPU staging) vs GPU-side reduce_add. |
| `gds_cufile_minimal.py`, `gpudirect_storage_example.py` | GPUDirect Storage samples for verifying cuFile setup, buffer alignment, and throughput. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json` | Harness entrypoint plus expectation baselines for spotting regressions. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch05.compare
python -m cli.aisp bench list-targets --chapter ch05
python -m cli.aisp bench run --targets ch05 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python baseline_storage_cpu.py --inspect` exposes CPU wait time > GPU time; `optimized_storage_cpu.py` reverses the ratio with >=80% GPU utilization.
- `python -m ch05.gds_cufile_minimal /tmp/gds_test_file.bin 1073741824 --generate` sustains multi-GB/s throughput when `/etc/cufile.json` is configured and NVMe advertises GPUDirect support.
- `python -m ch05.compare` shows optimized_ai eliminating CPU-side preprocessing from the critical path.

## Notes
- GPUDirect scripts fall back to host-mediated reads when `libcufile.so` is unavailable, making it safe to run on dev laptops.
- `requirements.txt` captures the limited extra deps (like `lmdb`) needed for the dataset shims.
