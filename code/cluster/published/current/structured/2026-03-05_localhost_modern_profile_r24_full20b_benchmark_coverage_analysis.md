# Benchmark Coverage Analysis: `2026-03-05_localhost_modern_profile_r24_full20b`

Generated: `2026-03-05T15:42:16.023468+00:00`

| Field | Value |
|---|---|
| Labels | `localhost` |
| Coverage score | `100%` |
| Maturity | `high` |
| Advanced coverage score | `100%` |

## Subsystem Coverage

| Subsystem | Covered |
|---|---|
| `sm_compute` | `yes` |
| `hbm_memory` | `yes` |
| `gpu_gpu_communication` | `yes` |
| `gpu_cpu_transfer` | `yes` |
| `ai_workloads` | `yes` |

Missing: `none`

## Advanced Coverage

| Advanced signal | Covered |
|---|---|
| `vllm_request_rate_sweep` | `yes` |
| `vllm_concurrency_repeat_stability` | `yes` |
| `vllm_request_rate_repeat_stability` | `yes` |
| `fio_repeat_stability` | `yes` |
| `allreduce_stability` | `yes` |
| `allreduce_latency_comp` | `yes` |
| `allgather_control_plane` | `yes` |
| `nccl_alltoall` | `yes` |
| `nccl_algo_comparison` | `yes` |
| `train_step_workload` | `yes` |
| `mlperf_alignment` | `yes` |

## Recommended Next Runs

- Coverage is complete across the five major subsystems.
