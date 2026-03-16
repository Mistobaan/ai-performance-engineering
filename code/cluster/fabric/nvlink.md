# NVLink Track

## What This Track Covers

Use this track for intra-node fabric questions:

- NVLink and NVSwitch topology correctness
- NMX or NetQ visibility for services, GPUs, switches, and partitions
- Single-node NCCL bandwidth sanity
- Correlation between NVLink topology and AI workload behavior

## Inventory

Primary evidence comes from the host metadata snapshot and the generated topology summary:

- `nvidia-smi topo -m`
- `<run_id>_<label>_meta_nvlink_topology.json`
- `<run_id>_<label>_meta_nvlink_topology.png`

What to look for:

- expected GPU count
- expected NVLink pair count
- peer links aligned with the intended topology

## Management Plane

Set:

- `AISP_FABRIC_NMX_URL`
- `AISP_FABRIC_NMX_TOKEN`

The evaluator uses these NMX endpoints:

- `/services`
- `/compute-nodes`
- `/gpus`
- `/switches`
- `/switch-nodes`
- `/ports`
- `/partitions`
- `/metrics`

These checks are read-only. Partition create/delete operations from the archived course docs are cataloged as `lab_only` and excluded from canonical runs.

## Scenario: Capacity Planning With NMX Topology

Use this path when the question is "can this GB200/NVL72 domain support two teams at once, and how should I split it?"

Commands:

- `curl -k https://seat##-nvlink.nvacademy.dev/nmx/v1/compute-nodes | jq`
- `curl -k https://seat##-nvlink.nvacademy.dev/nmx/v1/gpus | jq`
- `curl -sk https://seat##-nvlink.nvacademy.dev/nmx/v1/gpus | jq length`
- `curl -k https://seat##-nvlink.nvacademy.dev/nmx/v1/switches | jq`
- `curl -sk https://seat##-nvlink.nvacademy.dev/nmx/v1/switches | jq length`
- `curl -k https://seat##-nvlink.nvacademy.dev/nmx/v1/switch-nodes | jq`
- `curl -k https://seat##-nvlink.nvacademy.dev/nmx/v1/ports | jq`
- `curl -sk https://seat##-nvlink.nvacademy.dev/nmx/v1/ports | jq length`
- `echo "GPUs: $(curl -sk https://seat##-nvlink.nvacademy.dev/nmx/v1/gpus | jq length)"; echo "Switches: $(curl -sk https://seat##-nvlink.nvacademy.dev/nmx/v1/switches | jq length)"`

What the repo now answers in structured output:

- compute-node count, GPU count, switch-ASIC count, switch-tray count, and port count
- which GPUs map to which compute nodes via `SystemUID` plus `LocationInfo`
- candidate 4-GPU allocations for Team Alpha and Team Beta based on discovered node-to-GPU mapping
- switch-ASIC grouping into switch trays by `LocationInfo`
- GPU-facing vs switch-facing port counts and whether `BaseLID` is present

Fields worth interpreting explicitly:

- `DeviceID` is the per-node GPU index
- `SystemUID` ties a GPU back to its owning compute node
- `DomainUUID` identifies the NVLink domain
- `PartitionID` tells you whether the GPU is still in the default partition or already carved up
- `LocationInfo` gives the physical `chassis.slot.host.gpu` addressing used by partition commands
- `PortIDList` tells you how many NVLink ports belong to the entity

The fabric evaluator snapshots this into `<run_id>_fabric_verification.json` under `families.nvlink.control_plane.nmx.topology`.

## Scenario: Tenant Partition Planning And Safe Update Flow

Canonical fabric runs do not mutate partitions. They document the workflow and verify that the management surface required for it is visible.

Inspect:

- `curl -k https://seat##-nvlink.nvacademy.dev/nmx/v1/partitions | jq`
- `curl -k https://seat##-nvlink.nvacademy.dev/nmx/v1/operations/<operationID> | jq`

Documented lab-only commands:

- `curl -k -X POST https://seat##-nvlink.nvacademy.dev/nmx/v1/partitions -H "Content-Type: application/json" -d '{"Name":"AlphaPartition","DomainUUID":"<DomainUUID>","Members":{"locations":["<chassis.slot.host.gpu>"]}}' | jq`
- `curl -k -X PUT https://seat##-nvlink.nvacademy.dev/nmx/v1/partitions/<partition-id> -H "Content-Type: application/json" -d '{"DomainUUID":"<DomainUUID>","Members":{"locations":["<chassis.slot.host.gpu>"]}}' | jq`
- `curl -k -X DELETE https://seat##-nvlink.nvacademy.dev/nmx/v1/partitions/<partition-id> | jq`

Safe operational order:

- inspect the current partition owner first
- remove GPUs from the source partition before reassigning them
- poll `/operations/<operation-id>` until `status=completed`
- only then update the target partition
- deleting a partition returns its GPUs to the unassigned/default pool

The evaluator now captures:

- total partition count
- whether a default partition is present
- default partition member count
- unassigned GPU count
- the exact operation poll path and update-flow guidance

This lands in `<run_id>_fabric_verification.json` under `families.nvlink.control_plane.nmx.partitions`.

## Scenario: Telemetry Monitoring With NMX Metrics

Use the metrics endpoint when you need switch health and port/cable evidence to explain a runtime result.

Commands:

- `curl -sk https://seat##-nvlink.nvacademy.dev/nmx/v1/metrics | grep "^switch_temperature" | head -5`
- `curl -sk https://seat##-nvlink.nvacademy.dev/nmx/v1/metrics | grep "^PortXmitDataExtended" | head -5`
- `curl -sk https://seat##-nvlink.nvacademy.dev/nmx/v1/metrics | grep "^PortRcvDataExtended" | head -5`
- `curl -sk https://seat##-nvlink.nvacademy.dev/nmx/v1/metrics | grep "^PortLocalPhysicalErrors" | head -5`
- `curl -sk https://seat##-nvlink.nvacademy.dev/nmx/v1/metrics | grep "^CableInfoTemperature" | head -5`
- `curl -sk https://seat##-nvlink.nvacademy.dev/nmx/v1/metrics | grep "^CableInfoRxPower" | head -5`
- `curl -sk https://seat##-nvlink.nvacademy.dev/nmx/v1/metrics | grep "^CableInfoTxPower" | head -5`

What the evaluator records:

- whether the metrics endpoint responded
- how many temperature, throughput, physical-error, and cable-diagnostic series were visible
- sample metric queries so operators can pivot from the scorecard back to raw telemetry quickly

This lands in `<run_id>_fabric_verification.json` under `families.nvlink.control_plane.nmx.telemetry`.

## Runtime Verification

Runtime evidence is pulled from:

- `<run_id>_node1_nccl.json`
- `<run_id>_nccl_env_sensitivity.json`

Interpretation:

- high single-node NCCL algbw with expected topology means the NVLink path is behaving normally
- missing management-plane access downgrades the family to `runtime_verified`, not `full_stack_verified`

## Typical Failure Shapes

- Strong single-node NCCL algbw but no NMX visibility: runtime is healthy, control-plane is unverified.
- Weak single-node algbw with valid topology: inspect clock lock, NCCL env sensitivity, and topology parsing before blaming the fabric.
- Partition or service inventory missing: check NMX reachability and token scope.
