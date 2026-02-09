# Cluster Evaluation Harness: Advanced Runbook

This runbook is the detailed operator reference for discovery, benchmark execution, plotting, and reproducibility artifacts.

Key rule (GPU benchmarks):
- GPU clock locking is **mandatory**. GPU benchmark scripts fail if clock locking cannot be acquired via the repo harness (`lock_gpu_clocks`).
- Practically, this usually means you must configure passwordless sudo for `nvidia-smi` clock locking (so `sudo -n true` succeeds).

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
  --fp4-suite-dir /path/to/cluster_perf_suite \
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
  --suite-dir /path/to/cluster_perf_suite \
  --preset auto \
  --smoke-rounds 3 \
  --smoke-skew-threshold-pct 5 \
  --warmup 5 \
  --iters 30
```
Example suite path: `e.g. /path/to/clustermax` (set via `--suite-dir` or `CLUSTER_PERF_SUITE_DIR`).
Outputs:
`results/structured/<run_id>_<label>_cluster_perf_fp4_platform.json`,
`results/structured/<run_id>_r<round>_<label>_cluster_perf_fp4_smoke.json`,
`results/structured/<run_id>_r<round>_<label>_cluster_perf_fp4_smoke_clock_lock.json`,
`results/structured/<run_id>_fp4_smoke_skew_guard.json`,
`results/structured/<run_id>_<label>_cluster_perf_grouped_gemm_summary.json`,
`docs/figures/<run_id>_<label>_cluster_perf_grouped_gemm_tflops.png`.

#### Local Patch Prerequisite (GB200 DeepGEMM Grouped GEMM)
FP4 grouped-GEMM reproducibility on GB200 requires the local patch:
`code/cluster_perf_patches/deepgemm_gb200_grouped_gemm_ue8m0.patch`.
Without this patch, grouped DeepGEMM can fail with
`Unsupported architecture or scaling factor types`.

Apply steps (exact):
```bash
cd /home/ubuntu/ai-performance-engineering/code/cluster

export SUITE_ROOT=/path/to/cluster_perf_suite
export TARGET="${SUITE_ROOT}/standalone/compute/gemm-bench/grouped_gemm_bench.py"
export PATCH_SRC="code/cluster_perf_patches/deepgemm_gb200_grouped_gemm_ue8m0.patch"
export PATCH_TMP="/tmp/deepgemm_gb200_grouped_gemm_ue8m0.${USER}.patch"

test -f "${TARGET}"
cp "${TARGET}" "${TARGET}.pre_ue8m0.bak"

# Rewrite patch headers to the real suite location.
sed \
  -e "s|^--- code/cluster_perf_suite_snapshot/standalone/compute/gemm-bench/grouped_gemm_bench.py$|--- ${TARGET}|" \
  -e "s|^+++ code/cluster_perf_suite/standalone/compute/gemm-bench/grouped_gemm_bench.py$|+++ ${TARGET}|" \
  "${PATCH_SRC}" > "${PATCH_TMP}"

patch --dry-run -p0 < "${PATCH_TMP}"
patch -p0 < "${PATCH_TMP}"
```

Static verification checks:
```bash
rg -n \
  "use_ue8m0 = arch_major >= 10|disable_ue8m0_cast = not use_ue8m0|m_grouped_fp8_gemm_nt_contiguous|DeepGEMM unsupported|per_token_cast_to_fp8\\(a_bf16, use_ue8m0=use_ue8m0\\)|per_block_cast_to_fp8\\(b_bf16\\[i\\], use_ue8m0=use_ue8m0\\)" \
  "${TARGET}"
```

Runtime verification checks:
```bash
scripts/run_cluster_perf_grouped_gemm.sh \
  --suite-dir "${SUITE_ROOT}" \
  --run-id <run_id> \
  --label <label> \
  --image <image> \
  --preset auto \
  --warmup 2 \
  --iters 5

python3 - <<'PY' "results/structured/<run_id>_<label>_cluster_perf_grouped_gemm_summary.json"
import json, sys
summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
reason = ((summary.get("deepgemm") or {}).get("unsupported_reason") or "")
if "Unsupported architecture or scaling factor types" in reason:
    raise SystemExit("FAIL: legacy DeepGEMM scaling-factor error still present")
print("OK: summary does not contain the legacy scaling-factor error")
PY

test -f "docs/figures/<run_id>_<label>_cluster_perf_grouped_gemm_tflops.png"
```

### 10l) Bootstrap Nodes (Reproducibility)
Run node bootstrap directly (code sync + system deps + Python deps + optional suite sync):
```bash
scripts/bootstrap_cluster_nodes.sh \
  --run-id <run_id> \
  --hosts node1,node2 \
  --labels node1,node2 \
  --ssh-key ~/.ssh/ssh_key.pem \
  --sync-suite-dir /path/to/cluster_perf_suite
```
Example suite path: `e.g. /path/to/clustermax` (set via `--sync-suite-dir` or `CLUSTER_PERF_SUITE_DIR`).
Accepted `--sync-suite-dir` forms are:
- suite root containing `standalone/compute/`
- `<suite>/standalone`
- `<suite>/standalone/compute`
- parent directory containing a suite root
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
  env/requirements.txt    # plotting deps
  results/raw/            # raw logs
  results/structured/     # structured JSON/CSV
  scripts/                # discovery + run helpers
  field-report.md         # clean write-up (no results/raw links)
```

## Notes
- `results/raw/` is intentionally gitignored; the field report should link only to `results/structured/` and `docs/figures/`.

## Current Dependency Disclosure
- Core runtime: NVIDIA GPU + CUDA + NVML + working `nvidia-smi`; benchmark paths require successful clock locking via `scripts/run_with_gpu_clocks.sh`.
- Multi-node orchestration: passwordless SSH/SCP between hosts for all `*_all_nodes.sh` runners.
- Network/system tools used by suite scripts: `nccl-tests`, `iperf3`, RDMA/IB tools (`ibstat`, `rdma`, perftest utilities), and `fio`.
- Python runtime: `env/venv` with repo requirements and runnable `vllm` CLI for host-native vLLM scripts.
- vLLM serving sweep (`scripts/repro/run_vllm_serve_sweep_container.sh`) currently depends on Docker + NVIDIA container runtime and `nvidia-persistenced`.
- FP4 grouped GEMM checks (`scripts/run_cluster_perf_grouped_gemm.sh`, `scripts/run_fp4_checks_all_nodes.sh`, FP4 path in `scripts/run_cluster_eval_suite.sh`) depend on an external suite directory via `--suite-dir` / `CLUSTER_PERF_SUITE_DIR` (e.g. `/path/to/clustermax`) plus a container image via `--image` / `CONTAINER_IMAGE` (e.g. `ghcr.io/jordannanos/cmax-compute:latest`).
- DeepGEMM smoke (`analysis/smoke_deepgemm_fp8_fp4.py`) requires importable `deep_gemm` in the selected runtime environment.
