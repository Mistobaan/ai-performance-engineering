# Kubernetes-Native Benchmark Service

## Goal
Make benchmark execution declarative instead of script-by-script, while preserving the repo's existing artifact and methodology contracts.

The service contract is a `BenchmarkRun` custom resource:

- workload definition
- benchmark layer selection
- comparison variable
- scheduler path
- measurement policy
- provenance policy
- publication-grade versus realism-grade mode

The CRD lives at [`cluster/configs/benchmarkrun-crd.yaml`](/home/cfregly/ai-performance-engineering/code/cluster/configs/benchmarkrun-crd.yaml). The local example lives at [`templates/benchmark_run.yaml`](/home/cfregly/ai-performance-engineering/code/templates/benchmark_run.yaml).
Use [`templates/performance_intake.yaml`](/home/cfregly/ai-performance-engineering/code/templates/performance_intake.yaml) plus [`templates/benchmark_workload_spec.yaml`](/home/cfregly/ai-performance-engineering/code/templates/benchmark_workload_spec.yaml) to freeze the human-reviewed workload contract before generating or submitting a `BenchmarkRun`.
Use [`docs/performance_warehouse.md`](/home/cfregly/ai-performance-engineering/code/docs/performance_warehouse.md) plus [`templates/performance_warehouse_contract.yaml`](/home/cfregly/ai-performance-engineering/code/templates/performance_warehouse_contract.yaml) to keep raw evidence, warehouse lineage, and retention policy consistent across operator implementations.

## Control Loop
1. A user, CI workflow, or release process submits a `BenchmarkRun`.
2. The operator validates that the workload is frozen, the workload spec is pinned, and only one comparison variable is changing.
3. The operator resolves the requested layer set:
   - micro
   - component
   - end-to-end
4. The operator chooses execution primitives:
   - SLINKY for topology-aware placement and isolation policy
   - Kueue for queueing, admission, cohorting, and canary/nightly/pre-release scheduling
5. The operator launches the matching repo entrypoints:
   - `bench run-tier1`
   - `cluster common-eval`
   - targeted cluster scripts for NCCL, fio, vLLM sweeps, and startup checks
6. Raw artifacts land in immutable object storage plus the run directory contract already used by the repo.
7. Results are copied into:
   - a hot time-series system for operational alerting and debugging
   - a long-term analytical warehouse for regression analysis across releases, hardware generations, and data centers
8. The operator updates `status.conditions`, links artifact locations, and records audit metadata.

## Warehouse Contract
The operator should treat the warehouse as part of the product surface, not an afterthought.

Required split:

- raw logs, traces, profiler outputs, manifests, and exported metrics in immutable low-cost storage
- reduced-cardinality operational metrics in a hot time-series system
- curated fact and dimension tables in a columnar analytical store

Required curated dimensions:

- software versions
- hardware topology
- cluster and region metadata
- workload parameters
- artifact lineage

Required curated facts:

- benchmark run facts
- serving outcomes
- training outcomes when applicable
- telemetry slices for comparison and regression detectors

## Observability Contract
`BenchmarkRun.spec.observability` should give the operator enough structure to join service, job, and hardware telemetry without ad hoc heuristics.

Required signals include:

- service metrics and request traces
- `dcgm-exporter`
- `nvlink-exporter`
- `node-pci-exporter`
- `ping-exporter`
- `node-problem-detector`
- `hpc-verification`
- Kubernetes, Kueue, and SLINKY scheduling signals

Required stable join keys include:

- `run_id`
- `benchmark_case_id`
- `scheduler_run_id`
- `job_uid`
- `pod_uid`
- `node_name`
- `gpu_uuid`
- `rank_id` for distributed paths
- `request_id` and `trace_id` for serving paths

## Scenario Debuggability
The service should make these questions cheap to answer from the warehouse and raw evidence:

- inference p99 doubled but average barely moved
- training throughput dropped while GPU utilization stayed low
- distributed runs show periodic spikes and one node looks slow
- jobs spend too long pending and cluster utilization regresses

Those scenarios are modeled directly in `spec.observability.scenarioPlaybooks` so the operator and downstream warehouse know which pivots must remain intact.

## CRD Fields That Matter
| Field | Why it exists |
| --- | --- |
| `spec.layers` | Expresses the micro/component/end-to-end stack explicitly. |
| `spec.workload` | Freezes model, sequence-length mix, precision, batching policy, and concurrency model. |
| `spec.comparison.variableUnderTest` | Forces one-variable-at-a-time comparisons. |
| `spec.metrics` | Separates training and inference success criteria. |
| `spec.trials` | Declares replicates, confidence level, and outlier policy. |
| `spec.bottleneckAnalysis` | Binds taxonomy to instrumentation instead of guessing from GPU utilization alone. |
| `spec.distributed` | Requires rank-level visibility and node diagnosis for distributed runs. |
| `spec.provenance` | Makes image digests, topology, audit trails, and signed attestations first-class policy. |
| `spec.executionPolicy` | Separates publication-grade isolation from realism-grade customer-experience testing. |
| `spec.observability` | Declares telemetry joins, exporter coverage, and required debug playbooks. |
| `spec.sinks` | Splits raw evidence, hot operational metrics, and long-term regression analytics. |

## SLINKY + Kueue Mapping
| Concern | SLINKY role | Kueue role |
| --- | --- | --- |
| Placement | Enforce topology-aware placement, GPU/NIC affinity, exclusive placement for publication-grade runs | Admit the run into the correct queue and cohort |
| Isolation | Choose dedicated or topology-exclusive placement classes | Coordinate admission limits and multi-tenant fairness |
| Cadence | N/A | schedule canary, nightly, and pre-release runs |
| Backpressure | N/A | queue depth, retries, and scheduling policy |

The operator should translate the `BenchmarkRun` execution mode into scheduler semantics rather than forcing workload owners to understand queueing internals.

## Artifact Flow
Use the existing repo unit of record:

```text
cluster/runs/<run_id>/
  manifest.json
  structured/
  raw/
  figures/
  reports/
```

For single-node or non-cluster paths, keep using the existing benchmark history package under `artifacts/history/tier1/<run_id>/`.

## Publication-Grade Mode
Publication-grade runs should resolve to:

- dedicated nodes
- fixed topology
- stable background load
- topology-exclusive scheduling when appropriate
- image digests, not mutable tags
- signed provenance policy enabled

If those guarantees are unavailable, the operator should mark the run as realism-grade or reject it. It should not silently continue and let the user believe the result is publication-safe.

## Realism-Grade Mode
Realism-grade runs should intentionally model customer conditions:

- multi-tenant nodes when requested
- realistic background load
- cluster context capture preserved
- outliers explained with scheduler, topology, and node telemetry

This mode is not weaker. It answers a different question.

## Recommended Status Fields
Recommended `status` additions for a future operator implementation:

- `status.phase`
- `status.conditions`
- `status.selectedSchedulerPath`
- `status.runId`
- `status.runDirectory`
- `status.hotMetricsRef`
- `status.warehouseRef`
- `status.outlierSummary`
- `status.stragglerNodes`
- `status.provenanceRef`

## CI/CD Integration
Recommended automation policies:

- canary: smallest useful micro or component run after merge
- nightly: tier-1 plus selected cluster components
- pre-release: publication-grade end-to-end suite with full provenance requirements

These schedules should create `BenchmarkRun` resources or equivalent generated specs, not opaque shell scripts with hidden defaults.
