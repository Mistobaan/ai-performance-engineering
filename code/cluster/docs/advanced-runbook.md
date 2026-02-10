# Cluster Evaluation Harness: Advanced Runbook

This runbook is the detailed operator reference for discovery, benchmark execution, plotting, and reproducibility artifacts.

Key rule (GPU benchmarks):
- GPU clock locking is **mandatory**. GPU benchmark scripts fail if clock locking cannot be acquired via the repo harness (`lock_gpu_clocks`).
- Practically, this usually means you must configure passwordless sudo for `nvidia-smi` clock locking (so `sudo -n true` succeeds).

## Validated Reference Package
- Canonical validated run (used by the current field report): `2026-02-09_fresh_full_suite_e2e_green_rootfix`.
- Manifest: `results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_manifest.json`.
- Sanitized cluster metadata: `results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_meta.json`, `results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node2_meta.json`.
- Multi-node vLLM path result (strict-lock + digest-pinned image parity): `results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_multinode_serve.json`.
- NVLink/NVSwitch topology summaries: `results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_meta_nvlink_topology.json`, `results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node2_meta_nvlink_topology.json`.
- Dedicated nvbandwidth bundle: `results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_nvbandwidth.json`, `results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node2_nvbandwidth.json`.
- Storage parity fio artifacts: `results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_fio.json`, `results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node2_fio.json`.
- Narrative report: `field-report.md`.
- Claim-to-evidence ledger: `field-report-notes.md`.

## Core Run Flow

### 1) Run The Full Suite
This runs discovery + NCCL (1 node + 2 nodes) + vLLM serving sweep + GEMM sanity + fio + plots + manifest refresh:
```bash
scripts/run_cluster_eval_suite.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --oob-if <iface> \
  --socket-ifname <iface> \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5 \
  --health-suite collectives \
  --model openai/gpt-oss-120b \
  --isl 1024 \
  --osl 1024 \
  --concurrency-range "32 64 128 256 512"
```
Optional GB200-focused diagnostics can be toggled in the same suite run:
```bash
  --fp4-runtime host \
  --health-suite extended \
  --health-gdr --health-gdr-gpu 0 --health-gdr-mem-types 0,1 \
  --run-c2c --c2c-device 0 \
  --run-numa-mem-bw --numa-bytes 1073741824 --numa-iters 10 \
  --run-train-step --train-step-single-node --train-step-multi-node \
  --run-checkpoint-io --checkpoint-test-dir /tmp --checkpoint-bytes 4G
```
FP4 checks are enabled by default in `run_cluster_eval_suite.sh`. To skip them, pass `--disable-fp4`.
FP4 now includes a default paired-smoke skew guard (`--fp4-smoke-rounds 3`, `--fp4-smoke-skew-threshold-pct 5`): the run fails only when sustained cross-host skew is detected (max pairwise median gap > threshold).
Node bootstrap is also enabled by default (`scripts/bootstrap_cluster_nodes.sh` via the suite), so dependency/setup drift is corrected before checks run. Per-node bootstrap artifacts are written as:
`results/structured/<run_id>_<label>_bootstrap_status.json`.
To skip bootstrap explicitly, pass `--skip-bootstrap-nodes`.
Optional high-impact cross-reference diagnostics:
```bash
  --enable-mamf --mamf-mode quick --mamf-concurrent \
  --enable-allreduce-stability --allreduce-payload-gib 2.0 --allreduce-iters 200 \
  --enable-allreduce-latency-comp --allreduce-latency-payload-gib 4.0 --allreduce-latency-chunks 1000 \
  --enable-allgather-control-plane --allgather-control-iters 2000 --allgather-control-warmup 200 \
  --enable-nccl-algo-comparison --nccl-algos Ring,Tree,NVLS,auto
```

Optional multi-node vLLM serving path (Ray + TP across both nodes):
```bash
  --run-vllm-multinode \
  --vllm-multinode-concurrency 16 \
  --vllm-multinode-num-prompts 64 \
  --vllm-multinode-ray-port 6379
```

Portable baseline profile (recommended first run when FP4 deps are not available):
```bash
scripts/run_cluster_eval_suite.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --oob-if <iface> \
  --socket-ifname <iface> \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5 \
  --health-suite extended \
  --disable-fp4
```

### 2) Identity Snapshot And Uniqueness (Break-Glass Rotation Only)
Capture identity state (machine-id, hostname, SSH host keys) and log it for the field report:
```bash
scripts/setup.sh --label node1
```
Apply uniqueness fixes only with explicit approval (break-glass), then capture post-state evidence. Rotation is blocked unless explicitly overridden:
```bash
ALLOW_ID_ROTATION=1 ALLOW_SSH_KEY_ROTATION=1 scripts/setup.sh --label node2 --set-hostname node2 --regenerate-machine-id --regenerate-ssh-hostkeys --apply
```
Include peer ping checks in the readiness output:
```bash
scripts/setup.sh --label node1 --peers <peer_ip1,peer_ip2>
```
Append operator actions to a per-run JSONL log:
```bash
scripts/setup.sh --label node1 --log-ops
```
Outputs:
`results/structured/<run_id>_<label>_identity_pre.json`, `results/structured/<run_id>_<label>_identity_post.json`, `results/structured/<run_id>_<label>_readiness.json`, `results/raw/<run_id>_<label>_setup.log`, `results/raw/<run_id>_operator_actions.jsonl` (when `--log-ops` is used).

Validate operator log schema:
```bash
python3 scripts/validate_operator_log.py --input results/raw/<run_id>_operator_actions.jsonl
```

### 3) Discovery + Metadata
```bash
scripts/collect_system_info.sh --output results/structured/<run_id>_meta.json --label node1
```
For all nodes (requires SSH access):
```bash
RUN_ID=<run_id> \
  scripts/run_discovery_all_nodes.sh \
    --hosts node1,node2 \
    --ssh-key ~/.ssh/ssh_key.pem
```

TCP sysctl snapshots (structured JSON for diffing):
```bash
RUN_ID=<run_id> \
  scripts/collect_tcp_sysctl_all_nodes.sh \
    --hosts node1,node2 \
    --ssh-key ~/.ssh/ssh_key.pem
```

One-shot: discovery + tcp sysctl + storage layout + manifest
```bash
RUN_ID=<run_id> \
  scripts/collect_discovery_and_tcp_sysctl.sh \
    --hosts node1,node2 \
    --ssh-key ~/.ssh/ssh_key.pem
```
This writes a manifest JSON to `results/structured/<run_id>_manifest.json` (includes `manifest_version`, file hashes, and artifact counts).
Schema: `docs/manifest_schema.md`.

After generating plots, refresh the manifest to include figures:
```bash
python3 scripts/write_manifest.py --run-id <run_id> --hosts node1,node2 --include-figures
```

### 3a) Runtime/CVE Evidence (Required by Default)
Runtime/CVE checks are collected in both:
- `scripts/collect_discovery_and_tcp_sysctl.sh` (discovery flow).
- `scripts/run_cluster_health_suite.sh` (health flow, unless explicitly skipped).
`run_cluster_health_suite.sh` supports explicit opt-out with `--skip-runtime-cve-check` (default is enabled).

Direct runtime/CVE collection command:
```bash
RUN_ID=<run_id> \
  scripts/collect_container_runtime_all_nodes.sh \
    --hosts node1,node2 \
    --ssh-key ~/.ssh/ssh_key.pem
```
Artifacts:
- `results/structured/<run_id>_<label>_container_runtime.txt` (includes CVE-2025-23266 and CVE-2025-23267 status fields).

### 3b) Enable Researcher Stack (Optional)
Dry-run first:
```bash
scripts/enable_researcher_stack.sh
```
Apply on a node:
```bash
scripts/enable_researcher_stack.sh --apply
```

### 4) Cluster Health Suite
Runs `iperf3` + `ib_write_bw` + `nccl-tests` + `torchrun` and writes raw logs under `results/raw/` and a single JSON summary under `results/structured/`:
```bash
scripts/run_cluster_health_suite.sh --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem
```
Run suite on a subset of GPUs (example: exclude GPU0 on each node):
```bash
scripts/run_cluster_health_suite.sh --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --gpus-per-node 3 --cuda-visible-devices 1,2,3
```
Extended run (also adds `ib_read_bw` + `ib_send_bw` + NCCL `alltoall_perf`):
```bash
scripts/run_cluster_health_suite.sh --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --extended
```
If you hit an NCCL NVLS failure like `transport/nvls.cc: NCCL WARN Cuda failure 801 'operation not supported'`, rerun with NVLS disabled:
```bash
scripts/run_cluster_health_suite.sh --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --extended --nccl-nvls-enable 0
```

Repeat the suite to quantify variance (base + extended per repetition):
```bash
scripts/run_cluster_health_suite_repeats.sh --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --repeats 3 --mode both --prefix <run_id>_suite_variance
```
Pass extra args through to the suite with `--` (example: NCCL-only repeats):
```bash
scripts/run_cluster_health_suite_repeats.sh --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --repeats 3 --mode base --prefix <run_id>_nccl_only -- --skip-iperf3 --skip-ib --skip-torchdist
```

Summarize variance across multiple suite summaries:
```bash
python3 analysis/summarize_cluster_health_suite_variance.py --glob 'results/structured/<run_id>_suite_variance*_cluster_health_suite_summary.json' --output-md results/structured/<run_id>_suite_variance.md --output-json results/structured/<run_id>_suite_variance.json
```
Plot key metrics across repeats:
```bash
python3 analysis/plot_cluster_health_suite_variance.py --glob 'results/structured/<run_id>_suite_variance*_cluster_health_suite_summary.json' --output docs/figures/<run_id>_suite_variance_metrics.png
```
Compare two suite summaries (flags regressions/improvements):
```bash
python3 analysis/compare_cluster_health_summaries.py --baseline results/structured/<baseline>_cluster_health_suite_summary.json --candidate results/structured/<candidate>_cluster_health_suite_summary.json --threshold 0.05 --output-md results/structured/<baseline>_vs_<candidate>.md --output-json results/structured/<baseline>_vs_<candidate>.json
```

### 5) Plotting (after results exist)
```bash
python3 analysis/plot_nccl.py --input results/structured/<run_id>_nccl.json --out-dir docs/figures --run-id <run_id>
python3 analysis/plot_vllm.py --input results/structured/<run_id>_vllm.csv --out-dir docs/figures --run-id <run_id>
python3 analysis/plot_vllm_serve_sweep.py --input results/structured/<run_id>_<label>_vllm_serve_sweep.csv --out-dir docs/figures --run-id <run_id>_<label>
python3 analysis/plot_fio.py --input results/structured/<run_id>_<label>_fio.json --out docs/figures/<run_id>_<label>_fio.png
```

### 6) Benchmark A (Networking): NCCL `all_reduce_perf`
Single-node:
```bash
scripts/run_nccl_all_reduce.sh --run-id <run_id>_node1 --hosts localhost --label node1
python3 analysis/plot_nccl.py --input results/structured/<run_id>_node1_nccl.json --out-dir docs/figures --run-id <run_id>_node1
```

Multi-node (recommended explicit settings):
```bash
scripts/run_nccl_all_reduce.sh \
  --run-id <run_id>_2nodes \
  --hosts node1,node2 \
  --label node1node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --oob-if <iface> \
  --socket-ifname <iface> \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5
python3 analysis/plot_nccl.py --input results/structured/<run_id>_2nodes_nccl.json --out-dir docs/figures --run-id <run_id>_2nodes
```

### 7) Storage (fio)
```bash
scripts/run_fio_bench.sh --run-id <run_id> --label <label> --test-dir <path>
python3 analysis/plot_fio.py --input results/structured/<run_id>_<label>_fio.json --out docs/figures/<run_id>_<label>_fio.png
```

### 8) Inference (vLLM online serving sweep)
```bash
scripts/repro/run_vllm_serve_sweep_container.sh \
  --run-id <run_id> \
  --label <label> \
  --model openai/gpt-oss-120b \
  --isl 1024 \
  --osl 1024 \
  --concurrency-range "32 64 128 256 512"
python3 analysis/plot_vllm_serve_sweep.py \
  --input results/structured/<run_id>_<label>_vllm_serve_sweep.csv \
  --out-dir docs/figures \
  --run-id <run_id>_<label>
```
This benchmark self-locks clocks (strict) and writes a clock-lock artifact to:
`results/structured/<run_id>_<label>_vllm_serve_sweep_clock_lock.json`.

### 8b) Inference (vLLM multi-node serving, Ray)
```bash
scripts/repro/run_vllm_serve_multinode_container.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --model openai/gpt-oss-120b \
  --tp 8 \
  --isl 512 \
  --osl 256 \
  --concurrency 16 \
  --num-prompts 64 \
  --socket-ifname <iface> \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5
```
`--image <tag>` is supported; the runner auto-resolves a single repo digest and enforces that digest on both leader and worker before launch.

Artifacts:
- `results/structured/<run_id>_<leader_label>_vllm_multinode_serve.json`
- `results/structured/<run_id>_<leader_label>_vllm_multinode_serve.csv`
- `results/structured/<run_id>_<leader_label>_vllm_multinode_serve.jsonl`
- `results/structured/<run_id>_<leader_label>_vllm_multinode_leader_clock_lock.json`
- `results/structured/<run_id>_<worker_label>_vllm_multinode_worker_clock_lock.json`
- `results/raw/<run_id>_<leader_label>_vllm_multinode_serve/leader.log`

Latest known state for the canonical run (`2026-02-09_fresh_full_suite_e2e_green_rootfix`):
- command path executed correctly with strict lock on both nodes
- benchmark completed successfully (`status=ok`) at TP=8 across both nodes
- result includes image provenance showing identical digest on both nodes (`sha256:2338992e8e6413ba65a768e8e767a8092037316107739fa41057be3b0aaa0f90`)

### 8c) NVLink/NVSwitch Topology Artifact
Generate topology figure + structured summary from node meta (`nvidia-smi topo -m`):
```bash
python3 analysis/plot_nvlink_topology.py \
  --meta results/structured/<run_id>_<label>_meta.json \
  --fig-out docs/figures/<run_id>_<label>_nvlink_topology.png \
  --summary-out results/structured/<run_id>_<label>_nvlink_topology.json
```

### 8d) Dedicated nvbandwidth Bundle
Run a strict-lock `nvbandwidth` bundle and emit structured summaries:
```bash
scripts/repro/run_nvbandwidth_bundle.sh \
  --run-id <run_id> \
  --label <label> \
  --runtime host \
  --quick
```
Optional container runtime:
```bash
scripts/repro/build_cluster_perf_image.sh --profile open --tag cfregly/cluster_perf:latest
scripts/repro/run_nvbandwidth_bundle.sh --run-id <run_id> --label <label> --runtime container --image cfregly/cluster_perf@sha256:f9b2f503384d1780206dda1435cc2fb4eebe43bb15ff4b040a3601356af63a42 --quick
```
Artifacts:
- `results/structured/<run_id>_<label>_nvbandwidth.json`
- `results/structured/<run_id>_<label>_nvbandwidth_sums.csv`
- `results/structured/<run_id>_<label>_nvbandwidth_clock_lock.json`
- `results/raw/<run_id>_<label>_nvbandwidth/nvbandwidth.log`

### 9) Compute Sanity (GEMM per GPU, all nodes)
```bash
scripts/run_gemm_sanity_all_nodes.sh --run-id <run_id> --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem
python3 analysis/plot_gemm_bar.py --inputs results/structured/<run_id>_*_gemm_gpu_sanity.csv --output docs/figures/<run_id>_gemm_gpu_sanity.png --filter-m 16384
```

### 10) Optional Diagnostics

Optional: Long GEMM + 1 Hz telemetry (useful for chasing a few-% per-GPU or per-node deltas, or diagnosing power-cap behavior):
```bash
scripts/run_gemm_with_telemetry_all_nodes.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --gpus 0 \
  --iters 10000

python3 analysis/plot_gpu_telemetry.py \
  --csv results/raw/<run_id>_node1_gpu0_gemm_telemetry_query.csv --label node1_gpu0 \
  --csv results/raw/<run_id>_node2_gpu0_gemm_telemetry_query.csv --label node2_gpu0 \
  --out docs/figures/<run_id>_gpu0_telemetry.png \
  --title "GEMM Telemetry (GPU0): node1 vs node2"
```

### 10a) MAMF Finder (Maximum Achievable Matmul FLOPS)
Scans many matmul shapes to find the TRUE achievable TFLOPS ceiling for each GPU. This is the single most important compute diagnostic: it tells you the real performance bar (not theoretical peak), so you know when to stop optimizing.
```bash
scripts/run_mamf_finder_all_nodes.sh --run-id <run_id> --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --mode quick
scripts/run_mamf_finder_all_nodes.sh --run-id <run_id> --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --mode medium --concurrent
python3 analysis/plot_mamf.py --summary-inputs results/structured/<run_id>_*_mamf_summary.json --output docs/figures/<run_id>_mamf_straggler.png --mode straggler
```

### 10b) All-Reduce Stability Profiling (Network Jitter Detection)
Profiles a single large payload over many iterations to detect per-iteration bandwidth variance. A healthy network should show CV < 2%.
```bash
scripts/run_allreduce_stability.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --payload-gib 2.0 \
  --iters 200 \
  --socket-ifname <iface>
python3 analysis/plot_allreduce_stability.py --input results/structured/<run_id>_allreduce_stability.json --output docs/figures/<run_id>_allreduce_stability.png
```

### 10c) NCCL Algorithm Comparison (Ring vs Tree vs NVLS)
Tests NCCL algorithms explicitly to reveal if auto-selection is optimal:
```bash
scripts/run_nccl_algo_comparison.sh --run-id <run_id> --hosts node1,node2 --algos Ring,Tree,NVLS,auto --ssh-key ~/.ssh/ssh_key.pem --socket-ifname <iface>
python3 analysis/plot_nccl_algo_comparison.py --inputs results/structured/<run_id>_nccl_algo_*.json --output docs/figures/<run_id>_nccl_algo_comparison.png
```

### 10d) Concurrent GPU Straggler Detection
Run all GPUs simultaneously to find the straggler (slowest GPU sets training pace):
```bash
scripts/run_gemm_sanity_all_nodes.sh --run-id <run_id> --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem --concurrent
```

### 10e) All-Reduce Latency Comparison (1x Large vs Many Small)
Compares one large all-reduce vs many smaller all-reduces with equivalent total payload, which highlights communication fragmentation overhead:
```bash
scripts/run_allreduce_latency_comp.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --payload-gib 4.0 \
  --chunks 1000 \
  --socket-ifname <iface>
python3 analysis/plot_allreduce_latency_comp.py --input results/structured/<run_id>_allreduce_latency_comp.json --output docs/figures/<run_id>_allreduce_latency_comp.png
```

### 10f) All-Gather Control-Plane Comparison
Quantifies the overhead of `all_gather_object` versus tensor collectives for control-path synchronization:
```bash
scripts/run_allgather_control_plane.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --iters 2000 \
  --warmup 200 \
  --socket-ifname <iface>
python3 analysis/plot_allgather_control_plane.py --input results/structured/<run_id>_allgather_control_plane.json --output docs/figures/<run_id>_allgather_control_plane.png
```

### 10g) GPUDirect RDMA Validation (IB Perftest + Latency)
Run BW + latency checks with perftest `--use_cuda` (and optional dmabuf) through the health suite:
```bash
scripts/run_cluster_health_suite.sh \
  --run-id <run_id>_health_gdr \
  --hosts node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --extended \
  --gdr \
  --gdr-gpu 0 \
  --gdr-mem-types 0,1 \
  --gdr-use-dmabuf
```
Structured output includes base IB + tagged `ib_gdr` entries in:
`results/structured/<run_id>_<label>_cluster_health_suite_summary.json`.
If perftest CUDA mode is unsupported on a host, or if GDR subtests fail at runtime (for example MR allocation errors), the suite now records warnings and continues with non-GDR/base IB coverage instead of failing the full run.

### 10h) Grace/GB200 C2C + NUMA Probes
CPU<->GPU memcpy benchmark (pageable/pinned/managed host memory):
```bash
scripts/run_c2c_memcpy_bench.sh --run-id <run_id> --label <label> --device 0
python3 analysis/plot_c2c_memcpy.py --input results/structured/<run_id>_<label>_c2c_memcpy.json --out-dir docs/figures --run-id <run_id>_<label>
```

NUMA memory bandwidth probe (CPU NUMA nodes + memory-only NUMA domains):
```bash
scripts/run_numa_mem_bw_all_nodes.sh --run-id <run_id> --hosts node1,node2 --ssh-key ~/.ssh/ssh_key.pem
python3 analysis/plot_numa_mem_bw.py --input results/structured/<run_id>_<label>_numa_mem_bw.json --out docs/figures/<run_id>_<label>_numa_mem_bw.png
```

### 10i) End-To-End Train-Step Benchmark
Distributed tiny-transformer training step benchmark (forward+backward+optimizer), with app clocks captured per rank:
```bash
scripts/run_torchrun_transformer_train_step.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --gpus-per-node 4 \
  --oob-if <iface> \
  --nccl-ib-hca mlx5_0,mlx5_1,mlx5_4,mlx5_5 \
  --steps 30 --warmup-steps 5 --precision bf16 --fsdp 1
python3 analysis/plot_torchrun_train_step.py --input results/structured/<run_id>_<label>_torchrun_train_step.json --out docs/figures/<run_id>_<label>_torchrun_train_step.png
```

### 10j) Checkpoint I/O Benchmark
Checkpoint-like write/read throughput benchmark across nodes:
```bash
scripts/run_checkpoint_io_all_nodes.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --test-dir /tmp \
  --bytes 4G \
  --block-size 4M \
  --files 1 \
  --fsync 1
```
Outputs:
`results/structured/<run_id>_<label>_checkpoint_io.json` and
`results/structured/<run_id>_<label>_checkpoint_io.csv`.

### 10k) FP4 Coverage (DeepGEMM FP8xFP4)
Run FP4 smoke (paired rounds + skew guard) + grouped GEMM benchmark across all hosts:
```bash
scripts/run_fp4_checks_all_nodes.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --runtime host \
  --preset auto \
  --smoke-rounds 3 \
  --smoke-skew-threshold-pct 5 \
  --warmup 5 \
  --iters 30
```
Host bootstrap now installs `host_only` from `orig_parity` image provenance
(Torch/CUDA/NCCL/cuDNN/DeepGEMM) and installs host `nvbandwidth` from the same
image by default. Override source image with `--bootstrap-host-parity-image`.
If host bootstrap cannot satisfy prerequisites, switch to container runtime:
```bash
scripts/repro/build_cluster_perf_image.sh --profile open --tag cfregly/cluster_perf:latest
scripts/run_fp4_checks_all_nodes.sh ... --runtime container --stack-profile new_container
```
Orig-parity open container profile (legacy version matching without `jordannanos` dependency):
```bash
scripts/repro/build_cluster_perf_image.sh --profile orig_parity --tag cfregly/cluster_perf_orig_parity:latest
IMAGE_ID="$(docker image inspect --format '{{.Id}}' cfregly/cluster_perf_orig_parity:latest)"
scripts/run_fp4_checks_all_nodes.sh ... --runtime container --stack-profile orig_parity_container --image "${IMAGE_ID}"
```
Outputs:
`results/structured/<run_id>_<label>_cluster_perf_fp4_platform.json`,
`results/structured/<run_id>_fp4_attestation_consistency.json`,
`results/structured/<run_id>_r<round>_<label>_cluster_perf_fp4_smoke.json`,
`results/structured/<run_id>_r<round>_<label>_cluster_perf_fp4_smoke_clock_lock.json`,
`results/structured/<run_id>_fp4_smoke_skew_guard.json`,
`results/structured/<run_id>_<label>_cluster_perf_grouped_gemm_summary.json`,
`docs/figures/<run_id>_<label>_cluster_perf_grouped_gemm_tflops.png`.

Attestation mode is `balanced` by default. This records semantic patch checks,
runtime provenance metadata, and cross-host consistency without pinning
or mutating host state.

#### Local Attestation Target Verification
The semantic attestation target is in-repo:
`scripts/benchmarks/grouped_gemm_bench.py`.

Static verification check:
```bash
rg -n \
  "use_ue8m0 = arch_major >= 10|disable_ue8m0_cast = not use_ue8m0|m_grouped_fp8_gemm_nt_contiguous|DeepGEMM unsupported|per_token_cast_to_fp8\\(a_bf16, use_ue8m0=use_ue8m0\\)|per_block_cast_to_fp8\\(b_bf16\\[i\\], use_ue8m0=use_ue8m0\\)" \
  scripts/benchmarks/grouped_gemm_bench.py
```

Runtime verification checks:
```bash
scripts/run_cluster_perf_grouped_gemm.sh \
  --runtime host \
  --run-id <run_id> \
  --label <label> \
  --preset auto \
  --require-deepgemm \
  --warmup 2 \
  --iters 5

test -f "docs/figures/<run_id>_<label>_cluster_perf_grouped_gemm_tflops.png"
```

### 10l) Bootstrap Nodes (Reproducibility)
Run node bootstrap directly (code sync + system deps + Python deps):
```bash
scripts/bootstrap_cluster_nodes.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem
```
Outputs:
`results/structured/<run_id>_<label>_bootstrap_status.json`.

### 11) Optional: Screenshot Repro Suite
Runs the commands/benchmarks shown in the case-study screenshots and writes raw logs under `results/raw/` (gitignored):
```bash
RUN_ID=<run_id>_image_suite scripts/repro/run_image_suite.sh --run-id "$RUN_ID"
```

## Layout
```
cluster/
  analysis/               # plotting scripts
  docs/figures/           # generated plots
  env/requirements.txt    # host Python base deps (plotting + analysis)
  results/raw/            # raw logs
  results/structured/     # structured JSON/CSV
  scripts/                # discovery + run helpers
  field-report.md         # clean write-up (structured/figures links; raw links only for root-cause evidence)
```

## Notes
- `results/raw/` is intentionally gitignored; the field report should primarily link to `results/structured/` and `docs/figures/`, with `results/raw/` links allowed only when required to prove root-cause claims.
- Before report sign-off, run: `cluster/scripts/validate_field_report_requirements.sh --report cluster/field-report.md --notes cluster/field-report-notes.md`.

## Current Dependency Disclosure
- Core runtime: NVIDIA GPU + CUDA + NVML + working `nvidia-smi`; benchmark paths require successful clock locking via `scripts/run_with_gpu_clocks.sh`.
- Multi-node orchestration: passwordless SSH/SCP between hosts for all `*_all_nodes.sh` runners.
- Network/system tools used by suite scripts: `nccl-tests`, `iperf3`, RDMA/IB tools (`ibstat`, `rdma`, perftest utilities), and `fio`.
- Python runtime: `env/venv` with repo requirements and runnable `vllm` CLI for host-native vLLM scripts.
- vLLM serving sweep (`scripts/repro/run_vllm_serve_sweep_container.sh`) currently depends on Docker + NVIDIA container runtime and `nvidia-persistenced`.
- vLLM multi-node serving (`scripts/repro/run_vllm_serve_multinode_container.sh`) depends on the same container runtime plus Ray startup across both nodes and synchronized image/runtime contents.
- NVLink/NVSwitch topology artifact generation (`analysis/plot_nvlink_topology.py`) depends on valid `nvidia-smi topo -m` capture under `results/structured/<run_id>_<label>_meta.json`.
- FP4 grouped GEMM checks (`scripts/run_cluster_perf_grouped_gemm.sh`, `scripts/run_fp4_checks_all_nodes.sh`, FP4 path in `scripts/run_cluster_eval_suite.sh`) use an in-repo benchmark target (`scripts/benchmarks/grouped_gemm_bench.py`) with host-native runtime by default; container runtime is optional via `--runtime container --stack-profile new_container` (digest-pinned). See `docs/cluster-perf-stack-profiles.md`.
- DeepGEMM smoke (`analysis/smoke_deepgemm_fp8_fp4.py`) requires importable `deep_gemm` in the selected runtime environment.
