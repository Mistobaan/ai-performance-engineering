# Lab - Dynamic Prefill/Decode Router

## Summary
Simulates and benchmarks dynamic routing policies for large-scale inference: split GPUs into prefill/decode pools, monitor TTFT/TPOT, honor KV locality, and migrate traffic only when the score gap warrants it.

## Learning Goals
- Compare naive round-robin routing with telemetry-driven policies that stabilize TTFT.
- Prototype migration budgets, KV-locality boosts, and per-pool thresholds.
- Drive the router against synthetic workloads or real vLLM engines.
- Export detailed metrics (TTFT, TPOT, queue depth) for visualization.

## Directory Layout
| Path | Description |
| --- | --- |
| `router_round_robin.py`, `router_policy.py`, `driver.py`, `eval_stack.py` | Core router logic plus a synthetic simulator for deterministic comparisons. |
| `baseline_dynamic_router_vllm.py`, `optimized_dynamic_router_vllm.py`, `vllm_runner.py` | Integrations for running the routing policy against vLLM instances. |
| `baseline_dual_pool_vllm.py`, `optimized_dual_pool_vllm.py` | Shared-pool vs dual-pool TTFT benchmarks that reuse `vllm_runner.py`. |
| `topology.py`, `topology_probe.py` | NUMA-aware GPU mapping helpers and a target that emits topology JSON under `artifacts/topology/` for routing hints. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/dynamic_router
python -m cli.aisp bench run --targets labs/dynamic_router --profile minimal
```
- Targets follow the `labs/dynamic_router:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/dynamic_router:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

### ABI Preflight For vLLM Targets
Run this once per host before long sweeps that include `dynamic_router_vllm` or `dual_pool_vllm`:
```bash
python -c "import importlib, importlib.metadata as md, torch, vllm; importlib.import_module('vllm._C'); print(torch.__version__, md.version('vllm'), vllm.__version__)"
```
If this fails with `undefined symbol` from `vllm/_C.abi3.so`, the host has a torch/vLLM ABI mismatch.
Use the pinned benchmark stack:
- `torch==2.9.1+cu130`
- `vllm==0.16.0`
- `flashinfer-python==0.6.3`

## Validation Checklist
- `python labs/dynamic_router/driver.py --mode baseline` vs `--mode optimized` shows lower TTFT variance and higher TPOT for the optimized policy.
- `python -m cli.aisp bench run --targets labs/dynamic_router --profile minimal` records artifacts comparing baseline/optimized harness runs.
- `python -m cli.aisp bench run --targets labs/dynamic_router:dynamic_router_vllm --target-extra-arg labs/dynamic_router:dynamic_router_vllm="--model /path/to/model --decode-gpus 0,1"` succeeds on hosts with at least two GPUs and a local model copy.
- `python -m cli.aisp bench run --targets labs/dynamic_router:dual_pool_vllm --target-extra-arg labs/dynamic_router:dual_pool_vllm="--model /path/to/model --prefill-gpus 0 --decode-gpus 1"` contrasts shared versus dual pools and emits per-pool TTFT and queue depth.
- `python -m cli.aisp bench run --targets labs/dynamic_router:topology_probe` captures GPU↔NUMA mappings and distance matrices for consumption by the router.

## Notes
- `driver.py` accepts knobs such as `--prefill-gpus`, `--decode-gpus`, and `--migration-budget` to stress different regimes.
- vLLM integration now takes flags (`--model`, `--prefill-gpus`, `--decode-gpus`, etc.) plus locally available tokenizer/model weights.
- Router scoring incorporates pinned-host KV slab availability and NUMA-locality bias; feed it real topology via `topology_probe.py` or NVML when available.
