# MLPerf Alignment: `2026-03-05_localhost_modern_profile_r24_full20b`

Generated: `2026-03-05T15:34:42.103022+00:00`

| Field | Value |
|---|---|
| Overall status | `aligned` |
| Inference track ready | `True` |
| Training track ready | `True` |

## Inference Track

| Signal | Ready |
|---|---|
| `concurrency_sweep` | `yes` |
| `request_rate_sweep` | `yes` |
| `concurrency_slo_goodput` | `yes` |
| `request_rate_slo_goodput` | `yes` |
| `concurrency_repeat_stability` | `yes` |
| `request_rate_repeat_stability` | `yes` |

## Training Track

| Signal | Ready |
|---|---|
| `train_step_workload` | `yes` |
| `nccl_collectives_single_or_multi` | `yes` |
| `allreduce_stability` | `yes` |
| `alltoall_moe_coverage` | `yes` |
| `multinode_train_step` | `no` |

## References

- Inference: `MLPerf Inference Datacenter (LLM-style serving: throughput + TTFT/TPOT tails)`
- Training: `MLPerf Training (LLM-style train-step + distributed collective behavior)`
- Future-facing LLM set: `llama3.1_8b, llama3.1_405b, gpt_oss_20b`

## Recommendations

- Add multinode train-step evidence for distributed training alignment.
