# InfiniBand Track

## What This Track Covers

Use this track for lossless multi-node fabric questions:

- HCA state and link-layer verification
- subnet-manager discovery and routing checks
- switch, host, path, and counter visibility from the IB control plane
- NCCL and all-to-all correlation over the IB path

## Inventory

Host-side detection uses the meta snapshot:

- `ibstat`
- `rdma link`
- `ibv_devinfo`

Expected signals:

- `Link layer: InfiniBand`
- active `mlx5_*` HCAs
- multi-node NCCL artifacts when world size is greater than 1

## Control-Plane Verification

Set:

- `AISP_FABRIC_IB_MGMT_HOST`
- optionally `AISP_FABRIC_IB_MGMT_USER`
- optionally `AISP_FABRIC_IB_MGMT_SSH_KEY`

The evaluator runs read-only commands on the management host:

- `ibstat`
- `ibswitches`
- `ibhosts`
- `iblinkinfo`
- `ibnetdiscover`
- `saquery`
- `ibdiagnet -r`
- opportunistic follow-ons when LIDs are available:
  - `ibaddr`
  - `ibtracert`
  - `ibroute`
  - `perfquery`

These are the go-to checks for routing correctness, link-state visibility, and counter-based health.

## Runtime Verification

Runtime evidence is pulled from:

- `<run_id>_2nodes_nccl.json`
- `<run_id>_2nodes_alltoall_nccl_alltoall.json`
- `<run_id>_torchrun_connectivity_probe.json`

Interpretation:

- multi-node NCCL algbw or world size > 1 yields `runtime_verified`
- successful management-plane checks plus runtime evidence yields `full_stack_verified`
- `ibdiagnet` is the broadest single control-plane check; start there when the fabric is present but behavior is unclear

## Typical Failure Shapes

- HCAs present but no management host configured: `present_unverified` or `runtime_verified`, never a silent skip.
- Connectivity probe green but poor multi-node NCCL ratio: inspect routing, congestion, HCA binding, or link health.
- `ibdiagnet` or `saquery` failing from the management host: treat as control-plane visibility loss, not a runtime pass.
