# Fabric Evaluation Handbook

This directory is the maintained fabric surface for the repo.

The archived NVIDIA course exports under [nvidia-advanced-networking-for-ai-infra](/home/cfregly/ai-performance-engineering/code/cluster/fabric/nvidia-advanced-networking-for-ai-infra) stay as source material only. The operator-facing entrypoints are:

- [nvlink.md](/home/cfregly/ai-performance-engineering/code/cluster/fabric/nvlink.md)
- [infiniband.md](/home/cfregly/ai-performance-engineering/code/cluster/fabric/infiniband.md)
- [spectrum_x.md](/home/cfregly/ai-performance-engineering/code/cluster/fabric/spectrum_x.md)
- [cross_fabric.md](/home/cfregly/ai-performance-engineering/code/cluster/fabric/cross_fabric.md)
- [fabric_command_catalog.json](/home/cfregly/ai-performance-engineering/code/cluster/fabric/fabric_command_catalog.json)

## Entry Points

Use the dedicated fabric path when the question is "is the fabric healthy, correctly configured, and fast enough for AI workloads?":

```bash
python -m cli.aisp cluster fabric-eval \
  --run-id <run_id> \
  --hosts localhost \
  --labels localhost \
  --ssh-user "$(id -un)" \
  --primary-label localhost
```

This path defaults to capability-aware partial completion. It keeps `not_present`, `not_configured`, and `unavailable` signals visible in structured outputs instead of failing on broader publish-grade completeness gates.

Use the preset when staying inside the common-eval surface:

```bash
python -m cli.aisp cluster common-eval \
  --preset fabric-systems \
  --run-id <run_id> \
  --hosts <h1,h2> \
  --labels <l1,l2> \
  --ssh-user <user> \
  --ssh-key <key>
```

## Structured Outputs

Fabric runs emit these artifacts under `cluster/runs/<run_id>/structured/`:

- `<run_id>_fabric_command_catalog.json`
- `<run_id>_fabric_capability_matrix.json`
- `<run_id>_fabric_verification.json`
- `<run_id>_fabric_ai_correlation.json`
- `<run_id>_fabric_scorecard.json`
- `<run_id>_fabric_scorecard.md`

These artifacts are also folded into:

- `manifest.json`
- `<run_id>_cluster_scorecard.json`
- `reports/field-report-localhost*.md` when localhost report rendering is enabled

For NMX-backed NVLink domains, the verification payload now answers the operator scenarios directly:

- topology and capacity-planning counts for compute nodes, GPUs, switch ASICs, switch trays, and ports
- GPU-to-node mapping and candidate Alpha/Beta 4-GPU allocations
- partition inventory, default-partition membership, and unassigned capacity
- telemetry endpoint coverage for switch temperature, throughput, physical errors, and cable diagnostics

## Management-Plane Inputs

The fabric evaluator is capability-aware. If management endpoints are missing, the run records `not_configured` instead of silently skipping.

Environment variables:

- `AISP_FABRIC_NMX_URL`
- `AISP_FABRIC_NMX_TOKEN`
- `AISP_FABRIC_IB_MGMT_HOST`
- `AISP_FABRIC_IB_MGMT_USER`
- `AISP_FABRIC_IB_MGMT_SSH_KEY`
- `AISP_FABRIC_CUMULUS_HOSTS`
- `AISP_FABRIC_CUMULUS_USER`
- `AISP_FABRIC_CUMULUS_SSH_KEY`

Use `--require-management-plane` for publish-grade runs when missing management access should fail the fabric step instead of downgrading completeness.

## Design Rules

- Canonical evaluation is read-only.
- Mutating lab commands remain documented in the catalog as `lab_only=true`.
- NVLink, InfiniBand, and Spectrum-X must be interpreted alongside NCCL, all-to-all, train-step, vLLM, and nvbandwidth artifacts.
- The run directory remains the only contract surface that matters.
