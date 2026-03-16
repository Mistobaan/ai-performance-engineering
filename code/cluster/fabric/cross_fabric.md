# Cross-Fabric Interpretation

## Why This Exists

Fabric validation in this repo is not a separate networking lab. It is tied directly to AI Systems Performance artifacts:

- NCCL all-reduce
- NCCL all-to-all
- torchrun connectivity
- train-step
- vLLM concurrency and request-rate sweeps
- nvbandwidth

The fabric question is always: what does the control plane say, what does the runtime say, and how does that line up with the workload outcome?

## Completeness Model

Each family lands in one of four states:

- `not_present`
- `present_unverified`
- `runtime_verified`
- `full_stack_verified`

Management endpoints that are missing or inaccessible produce structured `not_configured` or `unavailable` states. They do not disappear from the report.

## Fast Interpretation Rules

- `runtime_verified` + weak AI workload scaling usually means the runtime path is alive but something about routing, congestion, queueing, or placement still needs work.
- `full_stack_verified` + weak scaling shifts suspicion toward host/runtime choices, launch shape, or application behavior.
- `present_unverified` means the hardware is there but the evaluation still lacks enough evidence for a publish-grade claim.

## Core Questions To Answer

- Is the topology what the operator expects?
- Is the management plane visible and healthy?
- Are the runtime collectives and connectivity probes healthy?
- Do train-step or vLLM results show a knee or collapse that matches the fabric evidence?

## Artifact Reading Order

1. `<run_id>_fabric_capability_matrix.json`
2. `<run_id>_fabric_verification.json`
3. `<run_id>_fabric_ai_correlation.json`
4. `<run_id>_fabric_scorecard.json`
5. `<run_id>_cluster_scorecard.json`

That sequence moves from "what is present" to "what was verified" to "what it means for AI workloads".
