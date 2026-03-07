# Cluster Scorecard: `2026-03-05_localhost_modern_profile_r24_full20b`

Generated: `2026-03-05T15:43:55.689955+00:00`
Workload KPI label: `localhost`

## Canonical Completeness

| Field | Value |
|---|---|
| Overall score | `100.0` |
| Pass/fail | `pass` |
| Coverage score | `100%` |
| Advanced coverage score | `100%` |
| Coverage maturity | `high` |
| MLPerf overall status | `aligned` |
| MLPerf inference track ready | `True` |
| MLPerf training track ready | `True` |
| Gate: coverage >= min | `True` |
| Gate: advanced >= min | `True` |
| Gate: MLPerf alignment minimum | `True` |
| Gate: canonical complete | `True` |

## Unified KPIs

| Domain | KPI | Value |
|---|---|---:|
| Compute | GEMM max TFLOPS | `1272.3` |
| Memory | nvbandwidth HBM GB/s | `0.0` |
| Memory | STREAM-like triad GB/s | `6184.7` |
| Communication | NCCL single-node peak busbw GB/s | `0.0` |
| Communication | NCCL multi-node peak busbw GB/s | `0.0` |
| Communication | Multi/single busbw ratio | `0.00` |
| Communication | NCCL all-to-all single-node peak busbw GB/s | `0.0` |
| Communication | NCCL all-to-all multi-node peak busbw GB/s | `0.0` |
| Communication | NCCL all-to-all multi/single busbw ratio | `0.00` |
| Communication | NCCL algo winner | `n/a (single-rank)` |
| Communication | NCCL algo spread % | `n/a` |
| Communication | NCCL auto gap % | `n/a` |
| Communication | Allreduce stability CV % | `n/a` |
| Communication | Allreduce stability p99/p50 | `n/a` |
| Communication | Allreduce jitter assessment | `n/a (world_size<=1)` |
| Communication | Allreduce latency comp (small/large duration ratio) | `378.54` |
| Communication | Allreduce latency comp one-large duration ms | `0.0272` |
| Communication | Allreduce latency comp many-small duration ms | `10.2964` |
| Communication | all_gather_object vs tensor speedup | `4.78x` |
| Communication | all_gather_object vs all_reduce speedup | `8.36x` |
| Communication | Control-plane fastest method | `all_reduce_tensor` |
| Communication | Control-plane fastest latency ms | `0.0283` |
| Host transfer | nvbandwidth H2D GB/s | `55.6` |
| Workload | vLLM throughput gain ratio | `13.64` |
| Workload | vLLM p99 TTFT ratio | `9.67` |
| Workload | vLLM max SLO goodput tok/s | `5840.97` |
| Workload | vLLM goodput efficiency ratio | `0.27` |
| Workload | vLLM knee concurrency | `256` |
| Workload | vLLM request-rate max tok/s | `7056.09` |
| Workload | vLLM request-rate at max tok/s | `16.00` |
| Efficiency | vLLM tok/J @ max tok/s | `69.024` |
| Efficiency | vLLM request-rate tok/J @ max tok/s | `30.488` |
| Efficiency | Cost USD / 1M tok (concurrency) | `n/a` |
| Efficiency | Cost USD / 1M tok (request-rate) | `n/a` |
| Workload Stability | vLLM conc tok/s CV p95 % | `2.52` |
| Workload Stability | vLLM conc p99 TTFT CV p95 % | `6.99` |
| Workload Stability | vLLM rate tok/s CV p95 % | `1.16` |
| Storage Stability | fio seq-read BW CV % | `1.66` |
| Storage Stability | fio seq-write BW CV % | `2.21` |

## Bottleneck Classification

| Classifier | Value |
|---|---|
| Dominant bottleneck | `host-bound` |
| Confidence | `high` |

| Evidence |
|---|
| Only 0.27 of peak token throughput meets SLO (goodput efficiency), indicating host/scheduler latency pressure. |
| vLLM tail latency growth is steep (p99 TTFT ratio=9.67) with limited throughput gain (13.64x). |

| Recommended next actions |
|---|
| Inspect scheduler/data-path overhead and CPU-side batching limits. |
| Run Nsight Systems + PyTorch trace to attribute host gaps and launch overhead. |

## Per-Node Metrics

| Label | GEMM max TFLOPS | nvbandwidth HBM GB/s | STREAM triad GB/s | vLLM tok/s gain | vLLM p99 TTFT ratio | vLLM max SLO goodput tok/s | vLLM knee concurrency | vLLM conc tok/s CV p95 % | fio seq read MB/s | fio seq read CV % | fio seq write MB/s | fio seq write CV % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `localhost` | `1272.3` | `0.0` | `6184.7` | `13.64` | `9.67` | `5840.97` | `256` | `2.52` | `1466.6` | `1.66` | `697.3` | `2.21` |
