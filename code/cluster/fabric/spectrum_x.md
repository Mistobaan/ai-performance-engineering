# Spectrum-X / RoCE Track

## What This Track Covers

Use this track for Ethernet-based AI fabrics:

- high-speed Ethernet plus RDMA over Converged Ethernet detection
- NVUE and Cumulus checks for RoCE, adaptive routing, and BGP
- runtime correlation with multi-node NCCL and all-to-all
- explicit distinction between management coverage and runtime-only evidence

## Inventory

The evaluator flags Spectrum-X when the host shows:

- high-speed Ethernet from `ethtool`
- RDMA-over-Ethernet-like signals from `ibstat`, `rdma link`, or `ibv_devinfo`

Host-side evidence is read from the existing meta artifact.

## Control-Plane Verification

Set:

- `AISP_FABRIC_CUMULUS_HOSTS`
- optionally `AISP_FABRIC_CUMULUS_USER`
- optionally `AISP_FABRIC_CUMULUS_SSH_KEY`

The evaluator runs read-only commands on the listed Cumulus switches:

- `nv show router adaptive-routing`
- `nv show qos roce`
- `nv show vrf default router bgp neighbor`
- `vtysh -c "show bgp ipv4 unicast summary"`
- `vtysh -c "show ip route vrf default bgp"`

These checks are intended to surface:

- RoCE QoS state
- adaptive routing configuration
- BGP control-plane health
- route visibility for the Ethernet fabric

## Runtime Verification

Runtime evidence is shared with the multi-node communication path:

- `<run_id>_2nodes_nccl.json`
- `<run_id>_2nodes_alltoall_nccl_alltoall.json`
- `<run_id>_torchrun_connectivity_probe.json`

Interpretation:

- runtime-only evidence without NVUE access stays `runtime_verified`
- NVUE plus runtime evidence upgrades the family to `full_stack_verified`

## Typical Failure Shapes

- high-speed Ethernet detected, but no switch access: runtime may be healthy while congestion features remain unverified
- BGP or adaptive-routing checks failing: treat this as control-plane degradation even if workloads still run
- weak multi-node NCCL ratio with Spectrum-X present: correlate QoS, route state, and adaptive-routing checks before tuning workloads
