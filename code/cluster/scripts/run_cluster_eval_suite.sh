#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_cluster_eval_suite.sh --hosts <h1,h2,...> [options]

Runs a reusable "field report" eval suite:
  1) Discovery + TCP sysctl + storage layout + manifest
  2) Optional: quick friction battery (install/pull/import/download/IP/speed) on all nodes
  3) Optional: monitoring expectations snapshot (control-plane/GPU/system signals) on all nodes
  4) Required: hang-triage readiness bundle (py-spy + strace) on all nodes
  5) Required: fast torchrun NCCL connectivity probe (all nodes)
  6) Benchmark A: NCCL all_reduce_perf (single-node + multi-node)
  7) Required: NCCL env sensitivity sweep (CROSS_NIC / QPS / CTAs)
  8) Optional: cluster health suite (iperf/IB/NCCL/torchdist, with optional GDR)
  9) Benchmark B: vLLM online serving sweep (containerized, single-node)
  10) Optional: vLLM multinode serving benchmark (Ray, 2-node; auto-on for multi-node runs)
  11) Benchmark C: BF16 GEMM per-GPU sanity (all nodes)
  12) Optional FP4 checks: DeepGEMM FP8xFP4 smoke + grouped GEMM (all nodes)
  13) Optional high-impact extras (ml-engineering parity):
     MAMF, all-reduce stability, all-reduce latency comp,
     all_gather_object control-plane comparison, NCCL algo comparison
  14) Optional: CPU<->GPU C2C memcpy benchmark (local)
  15) Optional: NUMA memory-bandwidth probe (all nodes)
  16) Optional: end-to-end transformer train-step benchmark (single-node + multi-node)
  17) Optional: checkpoint-like I/O benchmark (all nodes)
  18) Storage: fio (all nodes)
  19) Optional: nvbandwidth bundle (all nodes; auto-on for multi-node runs)
  20) Plots (includes NVLink topology) + manifest refresh + artifact validation

Notes:
  - GPU benchmarks are strict: they FAIL if GPU clock lock cannot be acquired.
  - Preflight is strict: it FAILS if `nvidia-persistenced`, `nvidia-dcgm`, or
    (multi-node) IMEX domain health is not ready.
  - Step execution metadata is written to
    `results/structured/<run_id>_suite_steps.json`.
  - For multi-node runs, explicitly pin OOB and NCCL socket interfaces.

Options:
  --run-id <id>            Base RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>      Comma-separated host list (required)
  --labels <l1,l2,...>     Optional labels (must match host count)
  --ssh-user <user>        SSH user (default: ubuntu)
  --ssh-key <path>         SSH key path (default: $SSH_KEY)
  --oob-if <iface>         Interface for OpenMPI OOB (required for multi-node NCCL)
  --socket-ifname <iface>  NCCL socket bootstrap interface (default: --oob-if)
  --nccl-ib-hca <list>     NCCL_IB_HCA allowlist (optional)
  --nccl-nvls-enable <0|1|2> Export NCCL_NVLS_ENABLE to NCCL ranks (health suite only; default: unset)

  --primary-label <label>  Label for single-node/local steps (default: hostname -s or first --labels entry)

  --connectivity-probe-master-port <port>  Torchrun connectivity probe rendezvous port (default: 29504)
  --connectivity-probe-barrier-iters <n>   Connectivity probe barrier iterations (default: 5)
  --connectivity-probe-payload-bytes <n>   Connectivity probe all-reduce payload bytes (default: 8388608)
  --connectivity-probe-timeout-sec <n>     Connectivity probe distributed timeout in seconds (default: 120)

  --nccl-env-min-bytes <size>   NCCL env sensitivity min bytes (default: 1M)
  --nccl-env-max-bytes <size>   NCCL env sensitivity max bytes (default: 64M)
  --nccl-env-warmup <n>         NCCL env sensitivity warmup iterations (default: 5)
  --nccl-env-iters <n>          NCCL env sensitivity measured iterations (default: 20)

  --run-quick-friction          Run quick friction battery on all nodes (default: on)
  --skip-quick-friction         Skip quick friction battery
  --quick-friction-strict       Fail suite if any quick friction host result is non-ok
  --quick-friction-checks <csv> Quick friction checks (default: uv_torch_install,pip_torch_install,ngc_pull,torch_import,hf_download,ip_owner,speedtest)
  --quick-friction-timeout-sec <n>  Per-check timeout for quick friction (default: 900)
  --quick-friction-torch-version <ver>  Torch version for install checks (default: 2.5.1)
  --quick-friction-torch-index-url <url>  Torch wheel index URL (default: https://download.pytorch.org/whl/cu124)
  --quick-friction-ngc-image <ref>  Container image for pull timing check (default: nvcr.io/nvidia/pytorch:24.05-py3)
  --quick-friction-hf-model <id>    HF model for download timing check (default: openai-community/gpt2)
  --quick-friction-hf-local-dir-base <path>  Base temp dir for HF download check (default: /tmp)

  --run-monitoring-expectations      Run monitoring expectations snapshot on all nodes (default: on)
  --skip-monitoring-expectations     Skip monitoring expectations snapshot
  --monitoring-expectations-strict   Fail suite if any monitoring snapshot host result is non-ok
  --monitoring-checks <csv>          Monitoring checks (default: kubectl_pods,kubectl_top_nodes,kubectl_top_pods,nvidia_dmon,nvidia_nvlink,dcgmi_discovery,dcgmi_dmon,dmesg_tail)
  --monitoring-sample-count <n>      Sample count for dmon checks (default: 20)
  --monitoring-dmesg-lines <n>       dmesg tail lines to capture (default: 400)
  --monitoring-timeout-sec <n>       Per-check timeout for monitoring checks (default: 180)

  --model <hf_model_id>    vLLM model id (default: openai/gpt-oss-120b)
  --tp <n>                 vLLM tensor parallel (default: all visible GPUs)
  --isl <n>                vLLM input seq len (default: 1024)
  --osl <n>                vLLM output seq len (default: 1024)
  --concurrency-range "…"  vLLM concurrencies (default: "32 64 128 256 512")
  --port <port>            vLLM server port (default: 8888)
  --run-vllm-multinode     Force-enable 2-node vLLM serving benchmark via Ray
  --skip-vllm-multinode    Force-disable 2-node vLLM serving benchmark via Ray
  --vllm-multinode-concurrency <n>  Multinode vLLM max concurrency (single-point mode, default: 64)
  --vllm-multinode-concurrency-range "..."
                           Multinode vLLM concurrency sweep values (space/comma-separated).
                           When set, runs one multinode serve pass per value.
  --vllm-multinode-num-prompts <n>  Multinode vLLM prompt count (default: concurrency*10)
  --vllm-multinode-ray-port <port>  Multinode vLLM Ray head port (default: 6379)
  --vllm-multinode-image <image>    Multinode vLLM container image (default: auto by arch)
  --vllm-multinode-ray-timeout <sec>     Timeout waiting for Ray cluster (default: 300)
  --vllm-multinode-server-timeout <sec>  Timeout waiting for vLLM health (default: 1200)
  --vllm-multinode-worker-startup-wait <sec>  Delay before leader launch (default: 10)

  --enable-fp4             Run FP4 checks on all nodes (default: on)
  --disable-fp4            Disable FP4 checks
  --fp4-runtime <mode>     FP4 execution mode: host|container (default: host)
  --fp4-stack-profile <name>
                           FP4 stack profile: old_container|orig_parity_container|new_container|host_only
                           (default: runtime-specific from configs/cluster_perf_stack_profiles.json)
  --fp4-image <image>      FP4 container image for --fp4-runtime container
                           (default: fp4-stack-profile image_ref or $CONTAINER_IMAGE)
  --fp4-preset <name>      Grouped-GEMM preset for FP4 check (default: auto; GB-family uses all)
  --fp4-warmup <n>         Grouped-GEMM warmup for FP4 check (default: 5)
  --fp4-iters <n>          Grouped-GEMM measured iterations for FP4 check (default: 30)
  --fp4-smoke-m <int>      FP4 smoke shape M (default: 4096)
  --fp4-smoke-n <int>      FP4 smoke shape N (default: 4096)
  --fp4-smoke-k <int>      FP4 smoke shape K (default: 4096)
  --fp4-smoke-warmup <n>   FP4 smoke warmup iterations (default: 10)
  --fp4-smoke-iters <n>    FP4 smoke measured iterations (default: 30)
  --fp4-smoke-rounds <n>   FP4 paired smoke rounds per host (default: 3)
  --fp4-smoke-skew-threshold-pct <pct>
                           Fail FP4 smoke guard when max pairwise median gap exceeds this percent (default: 5)

  --bootstrap-nodes                Bootstrap all nodes before checks (default: on)
  --skip-bootstrap-nodes           Skip node bootstrap
  --bootstrap-install-system-packages   Install missing system deps during bootstrap (default: on)
  --bootstrap-skip-system-packages      Skip system package install during bootstrap
  --bootstrap-sync-code            Sync scripts/analysis/env requirements to remote nodes (default: on)
  --bootstrap-skip-sync-code       Skip code sync during bootstrap
  --bootstrap-install-python-deps  Ensure env/venv + Python deps during bootstrap (default: on)
  --bootstrap-skip-python-deps     Skip python dep install during bootstrap
  --bootstrap-host-parity-image <ref>  Source image for host-only parity install
                                        (default: cfregly/cluster_perf_orig_parity:latest)
  --bootstrap-torch-index-url <url>  Legacy fallback torch index (default: https://pypi.ngc.nvidia.com)
  --bootstrap-torch-version <ver>    Expected torch version after bootstrap parity install
                                     (default: 2.10.0a0+a36e1d39eb.nv26.01.42222806)

  --fio-test-dir <path>    fio directory (default: /tmp)
  --fio-runtime <sec>      fio runtime per test (default: 30)
  --run-nvbandwidth        Force-enable nvbandwidth bundle (all nodes)
  --skip-nvbandwidth       Force-disable nvbandwidth bundle
  --nvbandwidth-runtime <host|container>  nvbandwidth runtime (default: host)
  --nvbandwidth-image <image>             nvbandwidth image for runtime=container
                                          (default: cfregly/cluster_perf_orig_parity:latest)
  --nvbandwidth-bin <path>                nvbandwidth executable for runtime=host (default: nvbandwidth)
  --nvbandwidth-quick      Use reduced nvbandwidth testcase subset

  --health-suite <mode>    Optional multi-node diagnostics:
                           off|collectives|base|extended (default: collectives)
  --health-gdr             Enable GPUDirect RDMA checks inside health suite (default: off)
  --health-gdr-gpu <id>    CUDA device id for health-suite GDR checks (default: 0)
  --health-gdr-mem-types <csv>  CUDA mem types for GDR checks (default: 0)
  --health-gdr-use-dmabuf  Also run health-suite GDR checks with --use_cuda_dmabuf

  --enable-mamf            Run MAMF compute-depth scan (default: off)
  --mamf-mode <mode>       MAMF mode: quick|medium|thorough (default: quick)
  --mamf-concurrent        Run MAMF on all GPUs concurrently (straggler focus)

  --enable-allreduce-stability  Run all-reduce jitter/stability profile (default: off)
  --allreduce-payload-gib <f>   Stability payload GiB (default: 2.0)
  --allreduce-iters <n>         Stability iterations (default: 200)
  --allreduce-warmup <n>        Stability warmup iterations (default: 20)

  --enable-allreduce-latency-comp  Run 1x-large vs many-small all-reduce comparison (default: off)
  --allreduce-latency-payload-gib <f>  Latency comparison total payload GiB (default: 4.0)
  --allreduce-latency-chunks <n>       Latency comparison chunk count (default: 1000)
  --allreduce-latency-iters <n>        Latency comparison iterations (default: 5)
  --allreduce-latency-warmup <n>       Latency comparison warmup iterations (default: 1)

  --enable-allgather-control-plane  Run all_gather_object vs tensor collectives benchmark (default: off)
  --allgather-control-iters <n>      Control-plane benchmark iterations (default: 2000)
  --allgather-control-warmup <n>     Control-plane benchmark warmup iterations (default: 200)

  --enable-nccl-algo-comparison  Run NCCL Ring/Tree/NVLS/auto comparison (default: off)
  --nccl-algos <list>            NCCL algorithms for comparison (default: Ring,Tree,NVLS,auto)

  --check-ib-sharp         Optional multi-node check: SHARP userspace + forced NCCL CollNet all-reduce
                           (runs scripts/check_ib_sharp.sh; default: off)
  --ib-sharp-attempt-start-sharp-am  Best-effort: attempt to install/start sharp_am before rerunning the forced CollNet check
  --ib-sharp-am-host <h>   Host to start sharp_am on (default: first host in --hosts)

  --run-c2c                Run CPU<->GPU C2C memcpy benchmark on primary node (default: off)
  --c2c-device <id>        C2C benchmark GPU device (default: 0)
  --c2c-bw-sizes <csv>     C2C bandwidth sizes in bytes (default: 4194304,67108864,1073741824)
  --c2c-lat-sizes <csv>    C2C latency sizes in bytes (default: 4,4096,65536)
  --c2c-bw-iters <n>       C2C bandwidth iterations (default: 20)
  --c2c-lat-iters <n>      C2C latency iterations (default: 20000)
  --c2c-warmup <n>         C2C warmup iterations (default: 5)

  --run-numa-mem-bw        Run NUMA memory-bandwidth probe on all nodes (default: off)
  --numa-bytes <n>         NUMA probe bytes per iteration (default: 1073741824)
  --numa-iters <n>         NUMA probe iterations (default: 10)
  --numa-threads <n>       NUMA probe thread count (default: 16)
  --numa-warmup <n>        NUMA probe warmup iterations (default: 2)
  --numa-nodes <csv>       Optional explicit NUMA nodes to probe (default: auto)
  --numa-cpu-node <n>      Optional CPU NUMA node to bind (default: auto)

  --run-train-step         Run torchrun transformer train-step benchmark (default: off)
  --train-step-single-node Also run a single-node train-step baseline on primary host (default: on when --run-train-step)
  --train-step-multi-node  Run multi-node train-step on --hosts (default: on when --run-train-step)
  --train-master-port <p>  Train-step torchrun rendezvous port (default: 29510)
  --train-steps <n>        Train-step measured steps (default: 30)
  --train-warmup-steps <n> Train-step warmup steps (default: 5)
  --train-batch-size <n>   Train-step per-rank batch size (default: 2)
  --train-seq-len <n>      Train-step sequence length (default: 2048)
  --train-hidden <n>       Train-step hidden size (default: 4096)
  --train-layers <n>       Train-step layers (default: 24)
  --train-heads <n>        Train-step attention heads (default: 32)
  --train-mlp-ratio <n>    Train-step MLP ratio (default: 4)
  --train-precision <bf16|fp16>  Train-step precision (default: bf16)
  --train-fsdp <0|1>       Train-step FSDP FULL_SHARD enable (default: 1)
  --train-lr <float>       Train-step AdamW learning rate (default: 1e-4)

  --run-checkpoint-io      Run checkpoint-like write/read benchmark on all nodes (default: off)
  --checkpoint-test-dir <path>  Checkpoint benchmark directory (default: /tmp)
  --checkpoint-bytes <size>     Bytes per checkpoint file (default: 4G)
  --checkpoint-block-size <size>  Block size (default: 4M)
  --checkpoint-files <n>        Number of files (default: 1)
  --checkpoint-fsync <0|1>      fsync after write (default: 1)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh" ]]; then
  echo "ERROR: missing stack profile helper: ${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh" >&2
  exit 1
fi
# shellcheck source=scripts/cluster_perf_stack_profiles.sh
source "${ROOT_DIR}/scripts/cluster_perf_stack_profiles.sh"

RUN_ID="$(date +%Y-%m-%d)"
HOSTS=""
LABELS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"
OOB_IF="${OOB_IF:-}"
SOCKET_IFNAME=""
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
NCCL_NVLS_ENABLE=""

PRIMARY_LABEL=""

CONNECTIVITY_PROBE_MASTER_PORT="29504"
CONNECTIVITY_PROBE_BARRIER_ITERS="5"
CONNECTIVITY_PROBE_PAYLOAD_BYTES="8388608"
CONNECTIVITY_PROBE_TIMEOUT_SEC="120"

NCCL_ENV_MIN_BYTES="1M"
NCCL_ENV_MAX_BYTES="64M"
NCCL_ENV_WARMUP="5"
NCCL_ENV_ITERS="20"

RUN_QUICK_FRICTION=1
QUICK_FRICTION_STRICT=0
QUICK_FRICTION_CHECKS="uv_torch_install,pip_torch_install,ngc_pull,torch_import,hf_download,ip_owner,speedtest"
QUICK_FRICTION_TIMEOUT_SEC="900"
QUICK_FRICTION_TORCH_VERSION="2.5.1"
QUICK_FRICTION_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
QUICK_FRICTION_NGC_IMAGE="nvcr.io/nvidia/pytorch:24.05-py3"
QUICK_FRICTION_HF_MODEL="openai-community/gpt2"
QUICK_FRICTION_HF_LOCAL_DIR_BASE="/tmp"

RUN_MONITORING_EXPECTATIONS=1
MONITORING_EXPECTATIONS_STRICT=0
MONITORING_CHECKS="kubectl_pods,kubectl_top_nodes,kubectl_top_pods,nvidia_dmon,nvidia_nvlink,dcgmi_discovery,dcgmi_dmon,dmesg_tail"
MONITORING_SAMPLE_COUNT="20"
MONITORING_DMESG_LINES="400"
MONITORING_TIMEOUT_SEC="180"

MODEL="openai/gpt-oss-120b"
TP=""
ISL="1024"
OSL="1024"
CONCURRENCY_RANGE="32 64 128 256 512"
PORT="8888"
RUN_VLLM_MULTINODE_MODE="auto"
RUN_VLLM_MULTINODE=0
VLLM_MULTINODE_CONCURRENCY="64"
VLLM_MULTINODE_CONCURRENCY_RANGE=""
VLLM_MULTINODE_NUM_PROMPTS=""
VLLM_MULTINODE_RAY_PORT="6379"
VLLM_MULTINODE_IMAGE=""
VLLM_MULTINODE_RAY_TIMEOUT="300"
VLLM_MULTINODE_SERVER_TIMEOUT="1200"
VLLM_MULTINODE_WORKER_STARTUP_WAIT="10"
VLLM_MULTINODE_CONCURRENCY_VALUES=()

ENABLE_FP4=1
FP4_RUNTIME="host"
FP4_STACK_PROFILE=""
FP4_IMAGE="${CONTAINER_IMAGE:-}"
FP4_PRESET="auto"
FP4_WARMUP="5"
FP4_ITERS="30"
FP4_SMOKE_M="4096"
FP4_SMOKE_N="4096"
FP4_SMOKE_K="4096"
FP4_SMOKE_WARMUP="10"
FP4_SMOKE_ITERS="30"
FP4_SMOKE_ROUNDS="3"
FP4_SMOKE_SKEW_THRESHOLD_PCT="5"

BOOTSTRAP_NODES=1
BOOTSTRAP_INSTALL_SYSTEM_PACKAGES=1
BOOTSTRAP_SYNC_CODE=1
BOOTSTRAP_INSTALL_PYTHON_DEPS=1
BOOTSTRAP_HOST_PARITY_IMAGE="${BOOTSTRAP_HOST_PARITY_IMAGE:-cfregly/cluster_perf_orig_parity:latest}"
BOOTSTRAP_TORCH_INDEX_URL="https://pypi.ngc.nvidia.com"
BOOTSTRAP_TORCH_VERSION="2.10.0a0+a36e1d39eb.nv26.01.42222806"

FIO_TEST_DIR="/tmp"
FIO_RUNTIME="30"
RUN_NVBANDWIDTH_MODE="auto"
RUN_NVBANDWIDTH=0
NVBANDWIDTH_RUNTIME="host"
NVBANDWIDTH_IMAGE="cfregly/cluster_perf_orig_parity:latest"
NVBANDWIDTH_BIN="nvbandwidth"
NVBANDWIDTH_QUICK=0

HEALTH_SUITE_MODE="collectives"
HEALTH_GDR=0
HEALTH_GDR_GPU="0"
HEALTH_GDR_MEM_TYPES="0"
HEALTH_GDR_USE_DMABUF=0
CHECK_IB_SHARP=0
IB_SHARP_ATTEMPT_START_SHARP_AM=0
IB_SHARP_AM_HOST=""

RUN_C2C=0
C2C_DEVICE="0"
C2C_BW_SIZES="4194304,67108864,1073741824"
C2C_LAT_SIZES="4,4096,65536"
C2C_BW_ITERS="20"
C2C_LAT_ITERS="20000"
C2C_WARMUP="5"

RUN_NUMA_MEM_BW=0
NUMA_BYTES="1073741824"
NUMA_ITERS="10"
NUMA_THREADS="16"
NUMA_WARMUP="2"
NUMA_NODES=""
NUMA_CPU_NODE=""

RUN_TRAIN_STEP=0
TRAIN_STEP_SINGLE_NODE=1
TRAIN_STEP_MULTI_NODE=1
TRAIN_MASTER_PORT="29510"
TRAIN_STEPS="30"
TRAIN_WARMUP_STEPS="5"
TRAIN_BATCH_SIZE="2"
TRAIN_SEQ_LEN="2048"
TRAIN_HIDDEN="4096"
TRAIN_LAYERS="24"
TRAIN_HEADS="32"
TRAIN_MLP_RATIO="4"
TRAIN_PRECISION="bf16"
TRAIN_FSDP="1"
TRAIN_LR="1e-4"

RUN_CHECKPOINT_IO=0
CHECKPOINT_TEST_DIR="/tmp"
CHECKPOINT_BYTES="4G"
CHECKPOINT_BLOCK_SIZE="4M"
CHECKPOINT_FILES="1"
CHECKPOINT_FSYNC="1"

ENABLE_MAMF=0
MAMF_MODE="quick"
MAMF_CONCURRENT=0
ENABLE_ALLREDUCE_STABILITY=0
ALLREDUCE_PAYLOAD_GIB="2.0"
ALLREDUCE_ITERS="200"
ALLREDUCE_WARMUP="20"
ENABLE_ALLREDUCE_LATENCY_COMP=0
ALLREDUCE_LATENCY_PAYLOAD_GIB="4.0"
ALLREDUCE_LATENCY_CHUNKS="1000"
ALLREDUCE_LATENCY_ITERS="5"
ALLREDUCE_LATENCY_WARMUP="1"
ENABLE_ALLGATHER_CONTROL_PLANE=0
ALLGATHER_CONTROL_ITERS="2000"
ALLGATHER_CONTROL_WARMUP="200"
ENABLE_NCCL_ALGO_COMPARISON=0
NCCL_ALGOS="Ring,Tree,NVLS,auto"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --oob-if) OOB_IF="$2"; shift 2 ;;
    --socket-ifname) SOCKET_IFNAME="$2"; shift 2 ;;
    --nccl-ib-hca) NCCL_IB_HCA="$2"; shift 2 ;;
    --nccl-nvls-enable) NCCL_NVLS_ENABLE="$2"; shift 2 ;;

    --primary-label) PRIMARY_LABEL="$2"; shift 2 ;;
    --connectivity-probe-master-port) CONNECTIVITY_PROBE_MASTER_PORT="$2"; shift 2 ;;
    --connectivity-probe-barrier-iters) CONNECTIVITY_PROBE_BARRIER_ITERS="$2"; shift 2 ;;
    --connectivity-probe-payload-bytes) CONNECTIVITY_PROBE_PAYLOAD_BYTES="$2"; shift 2 ;;
    --connectivity-probe-timeout-sec) CONNECTIVITY_PROBE_TIMEOUT_SEC="$2"; shift 2 ;;
    --nccl-env-min-bytes) NCCL_ENV_MIN_BYTES="$2"; shift 2 ;;
    --nccl-env-max-bytes) NCCL_ENV_MAX_BYTES="$2"; shift 2 ;;
    --nccl-env-warmup) NCCL_ENV_WARMUP="$2"; shift 2 ;;
    --nccl-env-iters) NCCL_ENV_ITERS="$2"; shift 2 ;;

    --run-quick-friction) RUN_QUICK_FRICTION=1; shift ;;
    --skip-quick-friction) RUN_QUICK_FRICTION=0; shift ;;
    --quick-friction-strict) QUICK_FRICTION_STRICT=1; shift ;;
    --quick-friction-checks) QUICK_FRICTION_CHECKS="$2"; shift 2 ;;
    --quick-friction-timeout-sec) QUICK_FRICTION_TIMEOUT_SEC="$2"; shift 2 ;;
    --quick-friction-torch-version) QUICK_FRICTION_TORCH_VERSION="$2"; shift 2 ;;
    --quick-friction-torch-index-url) QUICK_FRICTION_TORCH_INDEX_URL="$2"; shift 2 ;;
    --quick-friction-ngc-image) QUICK_FRICTION_NGC_IMAGE="$2"; shift 2 ;;
    --quick-friction-hf-model) QUICK_FRICTION_HF_MODEL="$2"; shift 2 ;;
    --quick-friction-hf-local-dir-base) QUICK_FRICTION_HF_LOCAL_DIR_BASE="$2"; shift 2 ;;

    --run-monitoring-expectations) RUN_MONITORING_EXPECTATIONS=1; shift ;;
    --skip-monitoring-expectations) RUN_MONITORING_EXPECTATIONS=0; shift ;;
    --monitoring-expectations-strict) MONITORING_EXPECTATIONS_STRICT=1; shift ;;
    --monitoring-checks) MONITORING_CHECKS="$2"; shift 2 ;;
    --monitoring-sample-count) MONITORING_SAMPLE_COUNT="$2"; shift 2 ;;
    --monitoring-dmesg-lines) MONITORING_DMESG_LINES="$2"; shift 2 ;;
    --monitoring-timeout-sec) MONITORING_TIMEOUT_SEC="$2"; shift 2 ;;

    --model) MODEL="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --isl) ISL="$2"; shift 2 ;;
    --osl) OSL="$2"; shift 2 ;;
    --concurrency-range) CONCURRENCY_RANGE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --run-vllm-multinode) RUN_VLLM_MULTINODE_MODE="on"; shift ;;
    --skip-vllm-multinode) RUN_VLLM_MULTINODE_MODE="off"; shift ;;
    --vllm-multinode-concurrency) VLLM_MULTINODE_CONCURRENCY="$2"; shift 2 ;;
    --vllm-multinode-concurrency-range) VLLM_MULTINODE_CONCURRENCY_RANGE="$2"; shift 2 ;;
    --vllm-multinode-num-prompts) VLLM_MULTINODE_NUM_PROMPTS="$2"; shift 2 ;;
    --vllm-multinode-ray-port) VLLM_MULTINODE_RAY_PORT="$2"; shift 2 ;;
    --vllm-multinode-image) VLLM_MULTINODE_IMAGE="$2"; shift 2 ;;
    --vllm-multinode-ray-timeout) VLLM_MULTINODE_RAY_TIMEOUT="$2"; shift 2 ;;
    --vllm-multinode-server-timeout) VLLM_MULTINODE_SERVER_TIMEOUT="$2"; shift 2 ;;
    --vllm-multinode-worker-startup-wait) VLLM_MULTINODE_WORKER_STARTUP_WAIT="$2"; shift 2 ;;

    --enable-fp4) ENABLE_FP4=1; shift ;;
    --disable-fp4) ENABLE_FP4=0; shift ;;
    --fp4-runtime) FP4_RUNTIME="$2"; shift 2 ;;
    --fp4-stack-profile) FP4_STACK_PROFILE="$2"; shift 2 ;;
    --fp4-image) FP4_IMAGE="$2"; shift 2 ;;
    --fp4-preset) FP4_PRESET="$2"; shift 2 ;;
    --fp4-warmup) FP4_WARMUP="$2"; shift 2 ;;
    --fp4-iters) FP4_ITERS="$2"; shift 2 ;;
    --fp4-smoke-m) FP4_SMOKE_M="$2"; shift 2 ;;
    --fp4-smoke-n) FP4_SMOKE_N="$2"; shift 2 ;;
    --fp4-smoke-k) FP4_SMOKE_K="$2"; shift 2 ;;
    --fp4-smoke-warmup) FP4_SMOKE_WARMUP="$2"; shift 2 ;;
    --fp4-smoke-iters) FP4_SMOKE_ITERS="$2"; shift 2 ;;
    --fp4-smoke-rounds) FP4_SMOKE_ROUNDS="$2"; shift 2 ;;
    --fp4-smoke-skew-threshold-pct) FP4_SMOKE_SKEW_THRESHOLD_PCT="$2"; shift 2 ;;

    --bootstrap-nodes) BOOTSTRAP_NODES=1; shift ;;
    --skip-bootstrap-nodes) BOOTSTRAP_NODES=0; shift ;;
    --bootstrap-install-system-packages) BOOTSTRAP_INSTALL_SYSTEM_PACKAGES=1; shift ;;
    --bootstrap-skip-system-packages) BOOTSTRAP_INSTALL_SYSTEM_PACKAGES=0; shift ;;
    --bootstrap-sync-code) BOOTSTRAP_SYNC_CODE=1; shift ;;
    --bootstrap-skip-sync-code) BOOTSTRAP_SYNC_CODE=0; shift ;;
    --bootstrap-install-python-deps) BOOTSTRAP_INSTALL_PYTHON_DEPS=1; shift ;;
    --bootstrap-skip-python-deps) BOOTSTRAP_INSTALL_PYTHON_DEPS=0; shift ;;
    --bootstrap-host-parity-image) BOOTSTRAP_HOST_PARITY_IMAGE="$2"; shift 2 ;;
    --bootstrap-torch-index-url) BOOTSTRAP_TORCH_INDEX_URL="$2"; shift 2 ;;
    --bootstrap-torch-version) BOOTSTRAP_TORCH_VERSION="$2"; shift 2 ;;

    --fio-test-dir) FIO_TEST_DIR="$2"; shift 2 ;;
    --fio-runtime) FIO_RUNTIME="$2"; shift 2 ;;
    --run-nvbandwidth) RUN_NVBANDWIDTH_MODE="on"; shift ;;
    --skip-nvbandwidth) RUN_NVBANDWIDTH_MODE="off"; shift ;;
    --nvbandwidth-runtime) NVBANDWIDTH_RUNTIME="$2"; shift 2 ;;
    --nvbandwidth-image) NVBANDWIDTH_IMAGE="$2"; shift 2 ;;
    --nvbandwidth-bin) NVBANDWIDTH_BIN="$2"; shift 2 ;;
    --nvbandwidth-quick) NVBANDWIDTH_QUICK=1; shift ;;

    --health-suite) HEALTH_SUITE_MODE="$2"; shift 2 ;;
    --health-gdr) HEALTH_GDR=1; shift ;;
    --health-gdr-gpu) HEALTH_GDR_GPU="$2"; shift 2 ;;
    --health-gdr-mem-types) HEALTH_GDR_MEM_TYPES="$2"; shift 2 ;;
    --health-gdr-use-dmabuf) HEALTH_GDR_USE_DMABUF=1; shift ;;
    --enable-mamf) ENABLE_MAMF=1; shift ;;
    --mamf-mode) MAMF_MODE="$2"; shift 2 ;;
    --mamf-concurrent) MAMF_CONCURRENT=1; shift ;;
    --enable-allreduce-stability) ENABLE_ALLREDUCE_STABILITY=1; shift ;;
    --allreduce-payload-gib) ALLREDUCE_PAYLOAD_GIB="$2"; shift 2 ;;
    --allreduce-iters) ALLREDUCE_ITERS="$2"; shift 2 ;;
    --allreduce-warmup) ALLREDUCE_WARMUP="$2"; shift 2 ;;
    --enable-allreduce-latency-comp) ENABLE_ALLREDUCE_LATENCY_COMP=1; shift ;;
    --allreduce-latency-payload-gib) ALLREDUCE_LATENCY_PAYLOAD_GIB="$2"; shift 2 ;;
    --allreduce-latency-chunks) ALLREDUCE_LATENCY_CHUNKS="$2"; shift 2 ;;
    --allreduce-latency-iters) ALLREDUCE_LATENCY_ITERS="$2"; shift 2 ;;
    --allreduce-latency-warmup) ALLREDUCE_LATENCY_WARMUP="$2"; shift 2 ;;
    --enable-allgather-control-plane) ENABLE_ALLGATHER_CONTROL_PLANE=1; shift ;;
    --allgather-control-iters) ALLGATHER_CONTROL_ITERS="$2"; shift 2 ;;
    --allgather-control-warmup) ALLGATHER_CONTROL_WARMUP="$2"; shift 2 ;;
    --enable-nccl-algo-comparison) ENABLE_NCCL_ALGO_COMPARISON=1; shift ;;
    --nccl-algos) NCCL_ALGOS="$2"; shift 2 ;;
    --check-ib-sharp) CHECK_IB_SHARP=1; shift ;;
    --ib-sharp-attempt-start-sharp-am) IB_SHARP_ATTEMPT_START_SHARP_AM=1; shift ;;
    --ib-sharp-am-host) IB_SHARP_AM_HOST="$2"; shift 2 ;;

    --run-c2c) RUN_C2C=1; shift ;;
    --c2c-device) C2C_DEVICE="$2"; shift 2 ;;
    --c2c-bw-sizes) C2C_BW_SIZES="$2"; shift 2 ;;
    --c2c-lat-sizes) C2C_LAT_SIZES="$2"; shift 2 ;;
    --c2c-bw-iters) C2C_BW_ITERS="$2"; shift 2 ;;
    --c2c-lat-iters) C2C_LAT_ITERS="$2"; shift 2 ;;
    --c2c-warmup) C2C_WARMUP="$2"; shift 2 ;;

    --run-numa-mem-bw) RUN_NUMA_MEM_BW=1; shift ;;
    --numa-bytes) NUMA_BYTES="$2"; shift 2 ;;
    --numa-iters) NUMA_ITERS="$2"; shift 2 ;;
    --numa-threads) NUMA_THREADS="$2"; shift 2 ;;
    --numa-warmup) NUMA_WARMUP="$2"; shift 2 ;;
    --numa-nodes) NUMA_NODES="$2"; shift 2 ;;
    --numa-cpu-node) NUMA_CPU_NODE="$2"; shift 2 ;;

    --run-train-step) RUN_TRAIN_STEP=1; shift ;;
    --train-step-single-node) TRAIN_STEP_SINGLE_NODE=1; shift ;;
    --train-step-multi-node) TRAIN_STEP_MULTI_NODE=1; shift ;;
    --train-master-port) TRAIN_MASTER_PORT="$2"; shift 2 ;;
    --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
    --train-warmup-steps) TRAIN_WARMUP_STEPS="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --train-seq-len) TRAIN_SEQ_LEN="$2"; shift 2 ;;
    --train-hidden) TRAIN_HIDDEN="$2"; shift 2 ;;
    --train-layers) TRAIN_LAYERS="$2"; shift 2 ;;
    --train-heads) TRAIN_HEADS="$2"; shift 2 ;;
    --train-mlp-ratio) TRAIN_MLP_RATIO="$2"; shift 2 ;;
    --train-precision) TRAIN_PRECISION="$2"; shift 2 ;;
    --train-fsdp) TRAIN_FSDP="$2"; shift 2 ;;
    --train-lr) TRAIN_LR="$2"; shift 2 ;;

    --run-checkpoint-io) RUN_CHECKPOINT_IO=1; shift ;;
    --checkpoint-test-dir) CHECKPOINT_TEST_DIR="$2"; shift 2 ;;
    --checkpoint-bytes) CHECKPOINT_BYTES="$2"; shift 2 ;;
    --checkpoint-block-size) CHECKPOINT_BLOCK_SIZE="$2"; shift 2 ;;
    --checkpoint-files) CHECKPOINT_FILES="$2"; shift 2 ;;
    --checkpoint-fsync) CHECKPOINT_FSYNC="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Defensive defaults: keep bootstrap torch args defined even if unset in caller env.
BOOTSTRAP_HOST_PARITY_IMAGE="${BOOTSTRAP_HOST_PARITY_IMAGE:-cfregly/cluster_perf_orig_parity:latest}"
BOOTSTRAP_TORCH_INDEX_URL="${BOOTSTRAP_TORCH_INDEX_URL:-https://pypi.ngc.nvidia.com}"
BOOTSTRAP_TORCH_VERSION="${BOOTSTRAP_TORCH_VERSION:-2.10.0a0+a36e1d39eb.nv26.01.42222806}"

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

if [[ -n "$NCCL_NVLS_ENABLE" && "$NCCL_NVLS_ENABLE" != "0" && "$NCCL_NVLS_ENABLE" != "1" && "$NCCL_NVLS_ENABLE" != "2" ]]; then
  echo "ERROR: --nccl-nvls-enable must be 0, 1, or 2 (got: ${NCCL_NVLS_ENABLE})" >&2
  exit 2
fi

if [[ "$HEALTH_GDR_USE_DMABUF" -eq 1 && "$HEALTH_GDR" -ne 1 ]]; then
  echo "ERROR: --health-gdr-use-dmabuf requires --health-gdr" >&2
  exit 2
fi

if [[ -n "$HEALTH_GDR_GPU" && ! "$HEALTH_GDR_GPU" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --health-gdr-gpu must be a non-negative integer (got: ${HEALTH_GDR_GPU})" >&2
  exit 2
fi

if [[ -n "$TRAIN_PRECISION" && "$TRAIN_PRECISION" != "bf16" && "$TRAIN_PRECISION" != "fp16" ]]; then
  echo "ERROR: --train-precision must be bf16 or fp16 (got: ${TRAIN_PRECISION})" >&2
  exit 2
fi

if [[ "$RUN_VLLM_MULTINODE_MODE" != "auto" && "$RUN_VLLM_MULTINODE_MODE" != "on" && "$RUN_VLLM_MULTINODE_MODE" != "off" ]]; then
  echo "ERROR: invalid vLLM multinode mode: ${RUN_VLLM_MULTINODE_MODE}" >&2
  exit 2
fi
if [[ "$RUN_NVBANDWIDTH_MODE" != "auto" && "$RUN_NVBANDWIDTH_MODE" != "on" && "$RUN_NVBANDWIDTH_MODE" != "off" ]]; then
  echo "ERROR: invalid nvbandwidth mode: ${RUN_NVBANDWIDTH_MODE}" >&2
  exit 2
fi
if [[ "$NVBANDWIDTH_RUNTIME" != "host" && "$NVBANDWIDTH_RUNTIME" != "container" ]]; then
  echo "ERROR: --nvbandwidth-runtime must be host or container (got: ${NVBANDWIDTH_RUNTIME})" >&2
  exit 2
fi
if [[ "$NVBANDWIDTH_RUNTIME" == "container" && -z "$NVBANDWIDTH_IMAGE" ]]; then
  echo "ERROR: --nvbandwidth-image is required when --nvbandwidth-runtime=container" >&2
  exit 2
fi

if [[ -n "$VLLM_MULTINODE_CONCURRENCY_RANGE" ]]; then
  VLLM_MULTINODE_CONCURRENCY_RANGE="${VLLM_MULTINODE_CONCURRENCY_RANGE//,/ }"
  for c in $VLLM_MULTINODE_CONCURRENCY_RANGE; do
    if ! [[ "$c" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: --vllm-multinode-concurrency-range contains non-positive integer '${c}'" >&2
      exit 2
    fi
    VLLM_MULTINODE_CONCURRENCY_VALUES+=("$c")
  done
  if [[ "${#VLLM_MULTINODE_CONCURRENCY_VALUES[@]}" -eq 0 ]]; then
    echo "ERROR: --vllm-multinode-concurrency-range resolved to an empty list" >&2
    exit 2
  fi
else
  if ! [[ "$VLLM_MULTINODE_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --vllm-multinode-concurrency must be a positive integer (got: ${VLLM_MULTINODE_CONCURRENCY})" >&2
    exit 2
  fi
  VLLM_MULTINODE_CONCURRENCY_VALUES=("$VLLM_MULTINODE_CONCURRENCY")
fi
if [[ -n "$VLLM_MULTINODE_NUM_PROMPTS" && ! "$VLLM_MULTINODE_NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-multinode-num-prompts must be a positive integer (got: ${VLLM_MULTINODE_NUM_PROMPTS})" >&2
  exit 2
fi
if ! [[ "$VLLM_MULTINODE_RAY_PORT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-multinode-ray-port must be a positive integer (got: ${VLLM_MULTINODE_RAY_PORT})" >&2
  exit 2
fi
if ! [[ "$VLLM_MULTINODE_RAY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-multinode-ray-timeout must be a positive integer (got: ${VLLM_MULTINODE_RAY_TIMEOUT})" >&2
  exit 2
fi
if ! [[ "$VLLM_MULTINODE_SERVER_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-multinode-server-timeout must be a positive integer (got: ${VLLM_MULTINODE_SERVER_TIMEOUT})" >&2
  exit 2
fi
if ! [[ "$VLLM_MULTINODE_WORKER_STARTUP_WAIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --vllm-multinode-worker-startup-wait must be a positive integer (got: ${VLLM_MULTINODE_WORKER_STARTUP_WAIT})" >&2
  exit 2
fi

if ! [[ "$CONNECTIVITY_PROBE_MASTER_PORT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --connectivity-probe-master-port must be a positive integer (got: ${CONNECTIVITY_PROBE_MASTER_PORT})" >&2
  exit 2
fi
if ! [[ "$CONNECTIVITY_PROBE_BARRIER_ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --connectivity-probe-barrier-iters must be a positive integer (got: ${CONNECTIVITY_PROBE_BARRIER_ITERS})" >&2
  exit 2
fi
if ! [[ "$CONNECTIVITY_PROBE_PAYLOAD_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --connectivity-probe-payload-bytes must be a positive integer (got: ${CONNECTIVITY_PROBE_PAYLOAD_BYTES})" >&2
  exit 2
fi
if ! [[ "$CONNECTIVITY_PROBE_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --connectivity-probe-timeout-sec must be a positive integer (got: ${CONNECTIVITY_PROBE_TIMEOUT_SEC})" >&2
  exit 2
fi
if ! [[ "$NCCL_ENV_WARMUP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --nccl-env-warmup must be >= 0 (got: ${NCCL_ENV_WARMUP})" >&2
  exit 2
fi
if ! [[ "$NCCL_ENV_ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --nccl-env-iters must be a positive integer (got: ${NCCL_ENV_ITERS})" >&2
  exit 2
fi
if ! [[ "$QUICK_FRICTION_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --quick-friction-timeout-sec must be a positive integer (got: ${QUICK_FRICTION_TIMEOUT_SEC})" >&2
  exit 2
fi
if ! [[ "$MONITORING_SAMPLE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --monitoring-sample-count must be a positive integer (got: ${MONITORING_SAMPLE_COUNT})" >&2
  exit 2
fi
if ! [[ "$MONITORING_DMESG_LINES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --monitoring-dmesg-lines must be a positive integer (got: ${MONITORING_DMESG_LINES})" >&2
  exit 2
fi
if ! [[ "$MONITORING_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --monitoring-timeout-sec must be a positive integer (got: ${MONITORING_TIMEOUT_SEC})" >&2
  exit 2
fi

if [[ "$FP4_RUNTIME" != "host" && "$FP4_RUNTIME" != "container" ]]; then
  echo "ERROR: --fp4-runtime must be host or container (got: ${FP4_RUNTIME})" >&2
  exit 2
fi
if [[ "$ENABLE_FP4" -eq 1 ]]; then
  if [[ -z "$FP4_STACK_PROFILE" ]]; then
    FP4_STACK_PROFILE="$(cluster_perf_default_profile_for_runtime "$ROOT_DIR" "$FP4_RUNTIME")"
  fi
  if ! cluster_perf_profile_exists "$ROOT_DIR" "$FP4_STACK_PROFILE"; then
    echo "ERROR: unknown --fp4-stack-profile: ${FP4_STACK_PROFILE}" >&2
    exit 2
  fi
  if ! cluster_perf_profile_runtime_allowed "$ROOT_DIR" "$FP4_STACK_PROFILE" "$FP4_RUNTIME"; then
    echo "ERROR: --fp4-stack-profile ${FP4_STACK_PROFILE} does not allow --fp4-runtime ${FP4_RUNTIME}" >&2
    exit 2
  fi
  if [[ "$FP4_RUNTIME" == "container" && -z "$FP4_IMAGE" ]]; then
    FP4_IMAGE="$(cluster_perf_profile_image_ref "$ROOT_DIR" "$FP4_STACK_PROFILE")"
  fi
  if [[ "$FP4_RUNTIME" == "container" && -z "$FP4_IMAGE" ]]; then
    echo "ERROR: no FP4 container image resolved for profile=${FP4_STACK_PROFILE}" >&2
    exit 2
  fi
fi

if [[ "$MAMF_MODE" != "quick" && "$MAMF_MODE" != "medium" && "$MAMF_MODE" != "thorough" ]]; then
  echo "ERROR: --mamf-mode must be quick, medium, or thorough (got: ${MAMF_MODE})" >&2
  exit 2
fi

if ! [[ "$ALLGATHER_CONTROL_ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --allgather-control-iters must be a positive integer (got: ${ALLGATHER_CONTROL_ITERS})" >&2
  exit 2
fi
if ! [[ "$ALLGATHER_CONTROL_WARMUP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --allgather-control-warmup must be >= 0 (got: ${ALLGATHER_CONTROL_WARMUP})" >&2
  exit 2
fi

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
IFS=',' read -r -a LABEL_ARR <<<"$LABELS"
if [[ -n "$LABELS" && "${#LABEL_ARR[@]}" -ne "${#HOST_ARR[@]}" ]]; then
  echo "ERROR: --labels count must match --hosts count" >&2
  exit 2
fi

if [[ "$RUN_VLLM_MULTINODE_MODE" == "on" ]]; then
  RUN_VLLM_MULTINODE=1
elif [[ "$RUN_VLLM_MULTINODE_MODE" == "off" ]]; then
  RUN_VLLM_MULTINODE=0
elif [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  RUN_VLLM_MULTINODE=1
else
  RUN_VLLM_MULTINODE=0
fi

if [[ "$RUN_NVBANDWIDTH_MODE" == "on" ]]; then
  RUN_NVBANDWIDTH=1
elif [[ "$RUN_NVBANDWIDTH_MODE" == "off" ]]; then
  RUN_NVBANDWIDTH=0
elif [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  RUN_NVBANDWIDTH=1
else
  RUN_NVBANDWIDTH=0
fi

if [[ "$RUN_VLLM_MULTINODE_MODE" == "on" && "${#HOST_ARR[@]}" -lt 2 ]]; then
  echo "ERROR: --run-vllm-multinode requires at least 2 hosts" >&2
  exit 2
fi

if [[ -z "$PRIMARY_LABEL" ]]; then
  if [[ -n "$LABELS" && "${#LABEL_ARR[@]}" -gt 0 ]]; then
    PRIMARY_LABEL="$(echo "${LABEL_ARR[0]}" | xargs)"
  else
    PRIMARY_LABEL="$(hostname -s)"
  fi
fi

if [[ -z "$SOCKET_IFNAME" ]]; then
  SOCKET_IFNAME="$OOB_IF"
fi

fail=0

SUITE_LOG_DIR="${ROOT_DIR}/results/raw/${RUN_ID}_suite"
SUITE_STEPS_JSON="${ROOT_DIR}/results/structured/${RUN_ID}_suite_steps.json"
mkdir -p "$SUITE_LOG_DIR"
mkdir -p "${ROOT_DIR}/results/structured"
printf "[]\n" >"${SUITE_STEPS_JSON}"

record_suite_step() {
  local json_path="$1"
  local name="$2"
  local start="$3"
  local end="$4"
  local rc="$5"
  local log_path="$6"
  local cmd_str="$7"
  python3 - "$json_path" "$name" "$start" "$end" "$rc" "$log_path" "$cmd_str" <<'PY'
import json
import sys

path, name, start, end, rc, log_path, cmd = sys.argv[1:]
try:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
except Exception:
    payload = []

if not isinstance(payload, list):
    payload = []

payload.append(
    {
        "name": name,
        "start_time": start,
        "end_time": end,
        "exit_code": int(rc),
        "log_path": log_path,
        "command": cmd,
    }
)

with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

run_step() {
  local name="$1"
  shift
  local start end rc log_path cmd_str
  log_path="${SUITE_LOG_DIR}/${name}.log"
  cmd_str="$(printf '%q ' "$@")"
  cmd_str="${cmd_str% }"
  start="$(date -Iseconds)"
  echo "==> [${start}] START ${name}"
  echo "    cmd: ${cmd_str}"
  echo "    log: ${log_path}"
  set +e
  "$@" >"$log_path" 2>&1
  rc=$?
  set -e
  end="$(date -Iseconds)"
  if ! record_suite_step "${SUITE_STEPS_JSON}" "${name}" "${start}" "${end}" "${rc}" "${log_path}" "${cmd_str}"; then
    echo "WARNING: failed to append step metadata for ${name}" >&2
  fi
  echo "<== [${end}] END ${name} rc=${rc}"
  if [[ "$rc" -ne 0 ]]; then
    fail=1
    echo "---- ${name} (last 60 lines) ----" >&2
    tail -n 60 "$log_path" >&2 || true
  fi
  return 0
}

sanitize_label() {
  local raw="$1"
  raw="${raw//./_}"
  raw="${raw//:/_}"
  echo "$raw"
}

label_for_index() {
  local idx="$1"
  local host="${HOST_ARR[$idx]}"
  if [[ -n "$LABELS" ]]; then
    local lbl
    lbl="$(echo "${LABEL_ARR[$idx]}" | xargs)"
    if [[ -n "$lbl" ]]; then
      echo "$lbl"
      return 0
    fi
  fi
  echo "$(sanitize_label "$host")"
}

validate_required_artifacts() {
  local missing=0
  local path=""
  local label=""

  for idx in "${!HOST_ARR[@]}"; do
    label="$(label_for_index "$idx")"
    path="${ROOT_DIR}/results/structured/${RUN_ID}_${label}_fio.json"
    if [[ ! -f "$path" ]]; then
      echo "ERROR: missing required fio artifact: ${path}" >&2
      missing=1
    fi
  done

  if [[ "$RUN_QUICK_FRICTION" -eq 1 ]]; then
    for idx in "${!HOST_ARR[@]}"; do
      label="$(label_for_index "$idx")"
      path="${ROOT_DIR}/results/structured/${RUN_ID}_${label}_quick_friction.json"
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required quick friction artifact: ${path}" >&2
        missing=1
        continue
      fi
      if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("test") != "quick_friction":
    raise SystemExit("unexpected test type")
status = payload.get("status")
if status not in {"ok", "degraded", "error"}:
    raise SystemExit(f"invalid quick friction status: {status}")
checks = payload.get("checks") or []
if not checks:
    raise SystemExit("quick friction checks are empty")
PY
      then
        echo "ERROR: invalid quick friction artifact: ${path}" >&2
        missing=1
      fi
    done
  fi

  if [[ "$RUN_MONITORING_EXPECTATIONS" -eq 1 ]]; then
    for idx in "${!HOST_ARR[@]}"; do
      label="$(label_for_index "$idx")"
      path="${ROOT_DIR}/results/structured/${RUN_ID}_${label}_monitoring_expectations.json"
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required monitoring expectations artifact: ${path}" >&2
        missing=1
        continue
      fi
      if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("test") != "monitoring_expectations":
    raise SystemExit("unexpected test type")
status = payload.get("status")
if status not in {"ok", "degraded", "error"}:
    raise SystemExit(f"invalid monitoring status: {status}")
checks = payload.get("checks") or []
if not checks:
    raise SystemExit("monitoring checks are empty")
categories = payload.get("categories") or {}
if not categories:
    raise SystemExit("monitoring categories are empty")
PY
      then
        echo "ERROR: invalid monitoring expectations artifact: ${path}" >&2
        missing=1
      fi
    done
  fi

  for idx in "${!HOST_ARR[@]}"; do
    label="$(label_for_index "$idx")"
    path="${ROOT_DIR}/results/structured/${RUN_ID}_${label}_hang_triage_readiness.json"
    if [[ ! -f "$path" ]]; then
      echo "ERROR: missing required hang triage artifact: ${path}" >&2
      missing=1
      continue
    fi
    if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
if status != "ok":
    raise SystemExit(f"hang triage readiness status is not ok: {status}")
tools = payload.get("tools") or {}
if not bool((tools.get("py_spy") or {}).get("available")):
    raise SystemExit("py-spy is not available")
if not bool((tools.get("strace") or {}).get("available")):
    raise SystemExit("strace is not available")
PY
    then
      echo "ERROR: invalid hang triage artifact: ${path}" >&2
      missing=1
    fi
  done

  path="${ROOT_DIR}/results/structured/${RUN_ID}_torchrun_connectivity_probe.json"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required connectivity probe artifact: ${path}" >&2
    missing=1
  else
    if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
if status != "ok":
    raise SystemExit(f"connectivity probe status is not ok: {status}")
ranks = payload.get("ranks") or []
if not ranks:
    raise SystemExit("connectivity probe ranks are empty")
bad = [r for r in ranks if (r or {}).get("status") != "ok"]
if bad:
    raise SystemExit(f"connectivity probe has failing ranks: {len(bad)}")
PY
    then
      echo "ERROR: invalid connectivity probe artifact: ${path}" >&2
      missing=1
    fi
  fi

  path="${ROOT_DIR}/results/structured/${RUN_ID}_nccl_env_sensitivity.json"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required NCCL env sensitivity artifact: ${path}" >&2
    missing=1
  else
    if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
if status != "ok":
    raise SystemExit(f"nccl env sensitivity status is not ok: {status}")
if int(payload.get("failure_count", 0)) != 0:
    raise SystemExit(f"nccl env sensitivity failure_count is non-zero: {payload.get('failure_count')}")
profiles = payload.get("profiles") or []
if not profiles:
    raise SystemExit("nccl env sensitivity profiles are empty")
if not any((p or {}).get("profile") == "baseline_auto" and (p or {}).get("status") == "ok" for p in profiles):
    raise SystemExit("nccl env sensitivity missing successful baseline_auto profile")
PY
    then
      echo "ERROR: invalid NCCL env sensitivity artifact: ${path}" >&2
      missing=1
    fi
  fi

  if [[ "$RUN_NVBANDWIDTH" -eq 1 ]]; then
    for idx in "${!HOST_ARR[@]}"; do
      label="$(label_for_index "$idx")"
      for suffix in "_nvbandwidth.json" "_nvbandwidth_sums.csv" "_nvbandwidth_clock_lock.json"; do
        path="${ROOT_DIR}/results/structured/${RUN_ID}_${label}${suffix}"
        if [[ ! -f "$path" ]]; then
          echo "ERROR: missing required nvbandwidth artifact: ${path}" >&2
          missing=1
        fi
      done
      path="${ROOT_DIR}/results/structured/${RUN_ID}_${label}_nvbandwidth.json"
      if [[ -f "$path" ]]; then
        if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
if status != "ok":
    raise SystemExit(f"nvbandwidth status is not ok: {status}")
clock = payload.get("clock_lock") or {}
if not bool(clock.get("all_devices_locked")):
    raise SystemExit("nvbandwidth clock_lock.all_devices_locked is false")
PY
        then
          echo "ERROR: invalid nvbandwidth summary: ${path}" >&2
          missing=1
        fi
      fi
    done
  fi

  if [[ "$RUN_VLLM_MULTINODE" -eq 1 ]]; then
    local leader_label worker_label
    leader_label="$(label_for_index 0)"
    worker_label="$(label_for_index 1)"

    for suffix in "_vllm_multinode_serve.json" "_vllm_multinode_serve.csv" "_vllm_multinode_serve.jsonl"; do
      path="${ROOT_DIR}/results/structured/${RUN_ID}_${leader_label}${suffix}"
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required multinode vLLM artifact: ${path}" >&2
        missing=1
      fi
    done
    for suffix in "_vllm_multinode_leader_clock_lock.json" "_vllm_multinode_worker_clock_lock.json"; do
      if [[ "$suffix" == *_leader_clock_lock.json ]]; then
        path="${ROOT_DIR}/results/structured/${RUN_ID}_${leader_label}${suffix}"
      else
        path="${ROOT_DIR}/results/structured/${RUN_ID}_${worker_label}${suffix}"
      fi
      if [[ ! -f "$path" ]]; then
        echo "ERROR: missing required multinode vLLM lock artifact: ${path}" >&2
        missing=1
      fi
    done

    path="${ROOT_DIR}/results/structured/${RUN_ID}_${leader_label}_vllm_multinode_serve.json"
    if [[ -f "$path" ]]; then
      if ! python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
if status != "ok":
    raise SystemExit(f"multinode vLLM status is not ok: {status}")
rcs = payload.get("return_codes") or {}
if int(rcs.get("leader", 1)) != 0 or int(rcs.get("worker", 1)) != 0:
    raise SystemExit(f"multinode vLLM return codes not clean: {rcs}")
clock = payload.get("clock_lock") or {}
leader = clock.get("leader") or {}
worker = clock.get("worker") or {}
if not bool(leader.get("all_devices_locked")):
    raise SystemExit("multinode vLLM leader clocks are not locked")
if not bool(worker.get("all_devices_locked")):
    raise SystemExit("multinode vLLM worker clocks are not locked")
metrics = payload.get("metrics") or {}
if (metrics.get("total_token_throughput") or 0) <= 0:
    raise SystemExit("multinode vLLM total_token_throughput is not positive")
PY
      then
        echo "ERROR: invalid multinode vLLM summary: ${path}" >&2
        missing=1
      fi
    fi
  fi

  if [[ "${#HOST_ARR[@]}" -gt 1 && "$HEALTH_SUITE_MODE" != "off" && "$HEALTH_GDR" -eq 1 ]]; then
    local -a hs_summaries=()
    shopt -s nullglob
    hs_summaries=( "${ROOT_DIR}/results/structured/${RUN_ID}_health_suite_${HEALTH_SUITE_MODE}_"*"_cluster_health_suite_summary.json" )
    shopt -u nullglob
    if [[ "${#hs_summaries[@]}" -eq 0 ]]; then
      echo "ERROR: health suite summary missing; cannot verify effective GDR." >&2
      missing=1
    else
      if ! python3 - "${hs_summaries[0]}" "${HEALTH_GDR_MEM_TYPES}" "${HEALTH_GDR_USE_DMABUF}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_mem_types = [x.strip() for x in sys.argv[2].split(",") if x.strip()]
expect_dmabuf = bool(int(sys.argv[3]))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
gdr = summary.get("gdr") or {}
if not bool(gdr.get("requested")):
    raise SystemExit(f"GDR requested flag is false in {summary_path}")
if not bool(gdr.get("effective_enabled")):
    reason = gdr.get("disabled_reason")
    raise SystemExit(f"GDR effective_enabled is false in {summary_path}; reason={reason}")
actual_mem_types = [str(x).strip() for x in (gdr.get("mem_types") or []) if str(x).strip()]
if sorted(actual_mem_types) != sorted(expected_mem_types):
    raise SystemExit(
        f"GDR mem_types mismatch in {summary_path}: expected={expected_mem_types} actual={actual_mem_types}"
    )
if expect_dmabuf and not bool(gdr.get("use_dmabuf")):
    raise SystemExit(f"GDR use_dmabuf expected true but false in {summary_path}")
PY
      then
        echo "ERROR: health suite GDR verification failed for ${hs_summaries[0]}" >&2
        missing=1
      fi
    fi
  fi

  if [[ "$missing" -ne 0 ]]; then
    return 1
  fi
  return 0
}

echo "========================================"
echo "Cluster Eval Suite"
echo "========================================"
echo "Date: $(date -Iseconds)"
echo "RUN_ID(base): ${RUN_ID}"
echo "HOSTS: ${HOSTS}"
echo "SUITE_STEPS_JSON: ${SUITE_STEPS_JSON}"
if [[ -n "$LABELS" ]]; then
  echo "LABELS: ${LABELS}"
fi
echo "SSH_USER: ${SSH_USER}"
echo "PRIMARY_LABEL: ${PRIMARY_LABEL}"
echo "connectivity_probe: master_port=${CONNECTIVITY_PROBE_MASTER_PORT} barrier_iters=${CONNECTIVITY_PROBE_BARRIER_ITERS} payload_bytes=${CONNECTIVITY_PROBE_PAYLOAD_BYTES} timeout_sec=${CONNECTIVITY_PROBE_TIMEOUT_SEC}"
echo "nccl_env_sensitivity: min=${NCCL_ENV_MIN_BYTES} max=${NCCL_ENV_MAX_BYTES} warmup=${NCCL_ENV_WARMUP} iters=${NCCL_ENV_ITERS}"
echo "quick_friction: enabled=${RUN_QUICK_FRICTION} strict=${QUICK_FRICTION_STRICT} checks='${QUICK_FRICTION_CHECKS}' timeout_sec=${QUICK_FRICTION_TIMEOUT_SEC} torch=${QUICK_FRICTION_TORCH_VERSION} hf_model=${QUICK_FRICTION_HF_MODEL}"
echo "monitoring_expectations: enabled=${RUN_MONITORING_EXPECTATIONS} strict=${MONITORING_EXPECTATIONS_STRICT} checks='${MONITORING_CHECKS}' sample_count=${MONITORING_SAMPLE_COUNT} dmesg_lines=${MONITORING_DMESG_LINES} timeout_sec=${MONITORING_TIMEOUT_SEC}"
if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  echo "OOB_IF: ${OOB_IF:-<unset>}"
  echo "NCCL_SOCKET_IFNAME: ${SOCKET_IFNAME:-<unset>}"
  echo "NCCL_IB_HCA: ${NCCL_IB_HCA:-<auto>}"
  echo "NCCL_NVLS_ENABLE: ${NCCL_NVLS_ENABLE:-<unset>}"
fi
echo "vLLM: model=${MODEL} tp=${TP:-<auto>} isl=${ISL} osl=${OSL} conc='${CONCURRENCY_RANGE}' port=${PORT}"
if [[ "$RUN_VLLM_MULTINODE" -eq 1 ]]; then
  echo "vLLM(multinode): enabled mode=${RUN_VLLM_MULTINODE_MODE} conc='${VLLM_MULTINODE_CONCURRENCY_VALUES[*]}' prompts=${VLLM_MULTINODE_NUM_PROMPTS:-<auto>} ray_port=${VLLM_MULTINODE_RAY_PORT} ray_timeout_s=${VLLM_MULTINODE_RAY_TIMEOUT} server_timeout_s=${VLLM_MULTINODE_SERVER_TIMEOUT} worker_startup_wait_s=${VLLM_MULTINODE_WORKER_STARTUP_WAIT} image=${VLLM_MULTINODE_IMAGE:-<auto>}"
else
  echo "vLLM(multinode): disabled mode=${RUN_VLLM_MULTINODE_MODE}"
fi
echo "fio: test_dir=${FIO_TEST_DIR} runtime_s=${FIO_RUNTIME}"
if [[ "$RUN_NVBANDWIDTH" -eq 1 ]]; then
  echo "nvbandwidth: enabled mode=${RUN_NVBANDWIDTH_MODE} runtime=${NVBANDWIDTH_RUNTIME} quick=${NVBANDWIDTH_QUICK}"
else
  echo "nvbandwidth: disabled mode=${RUN_NVBANDWIDTH_MODE}"
fi
echo "health_suite: ${HEALTH_SUITE_MODE}"
if [[ "$HEALTH_GDR" -eq 1 ]]; then
  echo "health_suite_gdr: enabled gpu=${HEALTH_GDR_GPU} mem_types=${HEALTH_GDR_MEM_TYPES} dmabuf=${HEALTH_GDR_USE_DMABUF}"
fi
if [[ "$CHECK_IB_SHARP" -eq 1 ]]; then
  echo "ib_sharp_check: enabled (attempt_start_sharp_am=${IB_SHARP_ATTEMPT_START_SHARP_AM}, sharp_am_host=${IB_SHARP_AM_HOST:-<auto>})"
fi
echo "c2c_memcpy: ${RUN_C2C} (device=${C2C_DEVICE})"
echo "numa_mem_bw: ${RUN_NUMA_MEM_BW}"
echo "train_step: ${RUN_TRAIN_STEP} (single_node=${TRAIN_STEP_SINGLE_NODE} multi_node=${TRAIN_STEP_MULTI_NODE})"
echo "checkpoint_io: ${RUN_CHECKPOINT_IO}"
echo "fp4_checks: ${ENABLE_FP4} (runtime=${FP4_RUNTIME} stack_profile=${FP4_STACK_PROFILE:-<auto>} preset=${FP4_PRESET} warmup=${FP4_WARMUP} iters=${FP4_ITERS})"
if [[ "$ENABLE_FP4" -eq 1 ]]; then
  if [[ "$FP4_RUNTIME" == "container" ]]; then
    echo "fp4_smoke: shape=${FP4_SMOKE_M}x${FP4_SMOKE_N}x${FP4_SMOKE_K} warmup=${FP4_SMOKE_WARMUP} iters=${FP4_SMOKE_ITERS} rounds=${FP4_SMOKE_ROUNDS} skew_threshold_pct=${FP4_SMOKE_SKEW_THRESHOLD_PCT} stack_profile=${FP4_STACK_PROFILE} image=${FP4_IMAGE}"
  else
    echo "fp4_smoke: shape=${FP4_SMOKE_M}x${FP4_SMOKE_N}x${FP4_SMOKE_K} warmup=${FP4_SMOKE_WARMUP} iters=${FP4_SMOKE_ITERS} rounds=${FP4_SMOKE_ROUNDS} skew_threshold_pct=${FP4_SMOKE_SKEW_THRESHOLD_PCT} stack_profile=${FP4_STACK_PROFILE}"
  fi
fi
echo "bootstrap_nodes: ${BOOTSTRAP_NODES} (sync_code=${BOOTSTRAP_SYNC_CODE} system_packages=${BOOTSTRAP_INSTALL_SYSTEM_PACKAGES} python_deps=${BOOTSTRAP_INSTALL_PYTHON_DEPS})"
echo "bootstrap_host_parity_image: ${BOOTSTRAP_HOST_PARITY_IMAGE}"
echo "mamf: ${ENABLE_MAMF} (mode=${MAMF_MODE} concurrent=${MAMF_CONCURRENT})"
echo "allreduce_stability: ${ENABLE_ALLREDUCE_STABILITY} (payload_gib=${ALLREDUCE_PAYLOAD_GIB} iters=${ALLREDUCE_ITERS} warmup=${ALLREDUCE_WARMUP})"
echo "allreduce_latency_comp: ${ENABLE_ALLREDUCE_LATENCY_COMP} (payload_gib=${ALLREDUCE_LATENCY_PAYLOAD_GIB} chunks=${ALLREDUCE_LATENCY_CHUNKS} iters=${ALLREDUCE_LATENCY_ITERS} warmup=${ALLREDUCE_LATENCY_WARMUP})"
echo "allgather_control_plane: ${ENABLE_ALLGATHER_CONTROL_PLANE} (iters=${ALLGATHER_CONTROL_ITERS} warmup=${ALLGATHER_CONTROL_WARMUP})"
echo "nccl_algo_comparison: ${ENABLE_NCCL_ALGO_COMPARISON} (algos=${NCCL_ALGOS})"
echo ""

common_args=(--run-id "$RUN_ID" --hosts "$HOSTS")
common_args+=(--ssh-user "$SSH_USER")
if [[ -n "$LABELS" ]]; then
  common_args+=(--labels "$LABELS")
fi
if [[ -n "$SSH_KEY" ]]; then
  common_args+=(--ssh-key "$SSH_KEY")
fi

if [[ "$BOOTSTRAP_NODES" -eq 1 ]]; then
  bootstrap_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --host-parity-image "$BOOTSTRAP_HOST_PARITY_IMAGE"
    --torch-index-url "$BOOTSTRAP_TORCH_INDEX_URL"
    --torch-version "$BOOTSTRAP_TORCH_VERSION"
  )
  if [[ -n "$LABELS" ]]; then
    bootstrap_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    bootstrap_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ "$BOOTSTRAP_INSTALL_SYSTEM_PACKAGES" -eq 0 ]]; then
    bootstrap_args+=(--skip-system-packages)
  fi
  if [[ "$BOOTSTRAP_SYNC_CODE" -eq 0 ]]; then
    bootstrap_args+=(--skip-sync-code)
  fi
  if [[ "$BOOTSTRAP_INSTALL_PYTHON_DEPS" -eq 0 ]]; then
    bootstrap_args+=(--skip-python-deps)
  fi
  run_step "bootstrap_nodes" "${ROOT_DIR}/scripts/bootstrap_cluster_nodes.sh" "${bootstrap_args[@]}"
fi

preflight_args=(--run-id "$RUN_ID" --hosts "$HOSTS" --ssh-user "$SSH_USER")
if [[ -n "$SSH_KEY" ]]; then
  preflight_args+=(--ssh-key "$SSH_KEY")
fi
run_step "preflight_services" "${ROOT_DIR}/scripts/preflight_cluster_services.sh" "${preflight_args[@]}"

run_step "discovery" "${ROOT_DIR}/scripts/collect_discovery_and_tcp_sysctl.sh" "${common_args[@]}"

if [[ "$RUN_QUICK_FRICTION" -eq 1 ]]; then
  quick_friction_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --checks "$QUICK_FRICTION_CHECKS"
    --timeout-sec "$QUICK_FRICTION_TIMEOUT_SEC"
    --torch-version "$QUICK_FRICTION_TORCH_VERSION"
    --torch-index-url "$QUICK_FRICTION_TORCH_INDEX_URL"
    --ngc-image "$QUICK_FRICTION_NGC_IMAGE"
    --hf-model "$QUICK_FRICTION_HF_MODEL"
    --hf-local-dir-base "$QUICK_FRICTION_HF_LOCAL_DIR_BASE"
  )
  if [[ -n "$LABELS" ]]; then
    quick_friction_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    quick_friction_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ "$QUICK_FRICTION_STRICT" -eq 1 ]]; then
    quick_friction_args+=(--strict)
  fi
  run_step "quick_friction_all_nodes" "${ROOT_DIR}/scripts/run_quick_friction_all_nodes.sh" "${quick_friction_args[@]}"
fi

if [[ "$RUN_MONITORING_EXPECTATIONS" -eq 1 ]]; then
  monitoring_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --checks "$MONITORING_CHECKS"
    --sample-count "$MONITORING_SAMPLE_COUNT"
    --dmesg-lines "$MONITORING_DMESG_LINES"
    --timeout-sec "$MONITORING_TIMEOUT_SEC"
  )
  if [[ -n "$LABELS" ]]; then
    monitoring_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    monitoring_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ "$MONITORING_EXPECTATIONS_STRICT" -eq 1 ]]; then
    monitoring_args+=(--strict)
  fi
  run_step "monitoring_expectations_all_nodes" "${ROOT_DIR}/scripts/collect_monitoring_expectations_all_nodes.sh" "${monitoring_args[@]}"
fi

# Required triage readiness bundle (py-spy + strace) for operator-debug path.
triage_args=(--run-id "$RUN_ID" --hosts "$HOSTS" --ssh-user "$SSH_USER")
if [[ -n "$LABELS" ]]; then
  triage_args+=(--labels "$LABELS")
fi
if [[ -n "$SSH_KEY" ]]; then
  triage_args+=(--ssh-key "$SSH_KEY")
fi
run_step "hang_triage_bundle" "${ROOT_DIR}/scripts/collect_hang_triage_bundle.sh" "${triage_args[@]}"

# Required fast distributed connectivity gate before network benchmarks.
connectivity_args=(
  --run-id "$RUN_ID"
  --hosts "$HOSTS"
  --ssh-user "$SSH_USER"
  --gpus-per-node "$(nvidia-smi -L | wc -l | tr -d ' ')"
  --master-port "$CONNECTIVITY_PROBE_MASTER_PORT"
  --barrier-iters "$CONNECTIVITY_PROBE_BARRIER_ITERS"
  --payload-bytes "$CONNECTIVITY_PROBE_PAYLOAD_BYTES"
  --timeout-sec "$CONNECTIVITY_PROBE_TIMEOUT_SEC"
)
if [[ -n "$SSH_KEY" ]]; then
  connectivity_args+=(--ssh-key "$SSH_KEY")
fi
if [[ -n "$OOB_IF" ]]; then
  connectivity_args+=(--oob-if "$OOB_IF")
fi
if [[ -n "$NCCL_IB_HCA" ]]; then
  connectivity_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
fi
run_step "connectivity_probe" "${ROOT_DIR}/scripts/run_torchrun_connectivity_probe.sh" "${connectivity_args[@]}"

# Benchmark A: NCCL all_reduce_perf
run_step "nccl_single_node" "${ROOT_DIR}/scripts/run_nccl_all_reduce.sh" \
  --run-id "${RUN_ID}_node1" \
  --hosts localhost \
  --label "${PRIMARY_LABEL}"

single_nccl_json="${ROOT_DIR}/results/structured/${RUN_ID}_node1_nccl.json"
single_nccl_raw="${ROOT_DIR}/results/raw/${RUN_ID}_node1_${PRIMARY_LABEL}_nccl_all_reduce.log"
if [[ ! -f "$single_nccl_json" && -f "$single_nccl_raw" ]]; then
  single_nccl_gpus="$(nvidia-smi -L | wc -l | tr -d ' ')"
  if [[ "$single_nccl_gpus" =~ ^[1-9][0-9]*$ ]]; then
    run_step "recover_nccl_single_node_json" python3 "${ROOT_DIR}/scripts/parse_nccl_log.py" \
      --input "$single_nccl_raw" \
      --output "$single_nccl_json" \
      --run-id "${RUN_ID}_node1" \
      --hosts localhost \
      --gpus-per-node "$single_nccl_gpus" \
      --command "recovered_from_raw_log:${single_nccl_raw}"
  else
    echo "WARNING: unable to recover single-node NCCL JSON; could not detect GPU count." >&2
  fi
fi

if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  if [[ -z "$OOB_IF" ]]; then
    echo "WARNING: multi-node NCCL requested but --oob-if is not set; skipping multi-node NCCL step." >&2
    fail=1
  else
    nccl_multi_args=(
      --run-id "${RUN_ID}_2nodes"
      --hosts "$HOSTS"
      --label "${PRIMARY_LABEL}node2"
      --oob-if "$OOB_IF"
      --socket-ifname "$SOCKET_IFNAME"
    )
    if [[ -n "$SSH_KEY" ]]; then
      nccl_multi_args+=(--ssh-key "$SSH_KEY")
    fi
    if [[ -n "$NCCL_IB_HCA" ]]; then
      nccl_multi_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
    fi
    run_step "nccl_multi_node" "${ROOT_DIR}/scripts/run_nccl_all_reduce.sh" "${nccl_multi_args[@]}"
  fi
fi

# Required NCCL env sensitivity sweep (CROSS_NIC / QPS / CTAs).
nccl_env_args=(
  --run-id "$RUN_ID"
  --hosts "$HOSTS"
  --gpus-per-node "$(nvidia-smi -L | wc -l | tr -d ' ')"
  --min-bytes "$NCCL_ENV_MIN_BYTES"
  --max-bytes "$NCCL_ENV_MAX_BYTES"
  --warmup "$NCCL_ENV_WARMUP"
  --iters "$NCCL_ENV_ITERS"
)
if [[ -n "$SSH_KEY" ]]; then
  nccl_env_args+=(--ssh-key "$SSH_KEY")
fi
if [[ -n "$OOB_IF" ]]; then
  nccl_env_args+=(--oob-if "$OOB_IF")
fi
if [[ -n "$SOCKET_IFNAME" ]]; then
  nccl_env_args+=(--socket-ifname "$SOCKET_IFNAME")
fi
if [[ -n "$NCCL_IB_HCA" ]]; then
  nccl_env_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
fi
run_step "nccl_env_sensitivity" "${ROOT_DIR}/scripts/run_nccl_env_sensitivity.sh" "${nccl_env_args[@]}"

# Optional multi-node diagnostics
if [[ "${#HOST_ARR[@]}" -gt 1 && "$HEALTH_SUITE_MODE" != "off" ]]; then
  suite_args=("${ROOT_DIR}/scripts/run_cluster_health_suite.sh" --run-id "${RUN_ID}_health_suite_${HEALTH_SUITE_MODE}" --hosts "$HOSTS")
  suite_args+=(--ssh-user "$SSH_USER")
  if [[ -n "$SSH_KEY" ]]; then
    suite_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ -n "$OOB_IF" ]]; then
    suite_args+=(--oob-if "$OOB_IF")
  fi
  if [[ -n "$NCCL_IB_HCA" ]]; then
    suite_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
  fi
  if [[ -n "$NCCL_NVLS_ENABLE" ]]; then
    suite_args+=(--nccl-nvls-enable "$NCCL_NVLS_ENABLE")
  fi
  if [[ "$HEALTH_GDR" -eq 1 ]]; then
    suite_args+=(--gdr --gdr-gpu "$HEALTH_GDR_GPU" --gdr-mem-types "$HEALTH_GDR_MEM_TYPES")
    if [[ "$HEALTH_GDR_USE_DMABUF" -eq 1 ]]; then
      suite_args+=(--gdr-use-dmabuf)
    fi
  fi

  case "$HEALTH_SUITE_MODE" in
    collectives)
      suite_args+=(--skip-iperf3 --skip-ib --skip-torchdist)
      ;;
    base)
      ;;
    extended)
      suite_args+=(--extended)
      ;;
    *)
      echo "ERROR: --health-suite must be one of off|collectives|base|extended (got: ${HEALTH_SUITE_MODE})" >&2
      exit 2
      ;;
  esac

  run_step "health_suite_${HEALTH_SUITE_MODE}" "${suite_args[@]}"
fi

# Optional multi-node SHARP readiness check (separate from the health suite so it can be toggled explicitly).
if [[ "${#HOST_ARR[@]}" -gt 1 && "$CHECK_IB_SHARP" -eq 1 ]]; then
  if [[ -z "$OOB_IF" ]]; then
    echo "WARNING: --check-ib-sharp requested but --oob-if is not set; skipping SHARP check." >&2
    fail=1
  else
    sharp_args=(--run-id "${RUN_ID}_ib_sharp_check" --hosts "$HOSTS" --oob-if "$OOB_IF")
    sharp_args+=(--ssh-user "$SSH_USER")
    if [[ -n "$SSH_KEY" ]]; then
      sharp_args+=(--ssh-key "$SSH_KEY")
    fi
    if [[ -n "$NCCL_IB_HCA" ]]; then
      sharp_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
    fi
    if [[ -n "$IB_SHARP_AM_HOST" ]]; then
      sharp_args+=(--sharp-am-host "$IB_SHARP_AM_HOST")
    fi
    if [[ "$IB_SHARP_ATTEMPT_START_SHARP_AM" -eq 1 ]]; then
      sharp_args+=(--attempt-start-sharp-am)
    fi
    run_step "ib_sharp_check" "${ROOT_DIR}/scripts/check_ib_sharp.sh" "${sharp_args[@]}"
  fi
fi

# Benchmark B: vLLM online serving sweep (local)
vllm_args=(
  --run-id "$RUN_ID"
  --label "$PRIMARY_LABEL"
  --model "$MODEL"
  --isl "$ISL"
  --osl "$OSL"
  --concurrency-range "$CONCURRENCY_RANGE"
  --port "$PORT"
)
if [[ -n "$TP" ]]; then
  vllm_args+=(--tp "$TP")
fi
run_step "vllm_serve_sweep" "${ROOT_DIR}/scripts/repro/run_vllm_serve_sweep_container.sh" "${vllm_args[@]}"

if [[ "$RUN_VLLM_MULTINODE" -eq 1 ]]; then
  if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
    for vllm_multi_conc in "${VLLM_MULTINODE_CONCURRENCY_VALUES[@]}"; do
      vllm_multi_args=(
        --run-id "$RUN_ID"
        --hosts "$HOSTS"
        --ssh-user "$SSH_USER"
        --model "$MODEL"
        --isl "$ISL"
        --osl "$OSL"
        --concurrency "$vllm_multi_conc"
        --port "$PORT"
        --ray-port "$VLLM_MULTINODE_RAY_PORT"
        --ray-ready-timeout "$VLLM_MULTINODE_RAY_TIMEOUT"
        --server-ready-timeout "$VLLM_MULTINODE_SERVER_TIMEOUT"
        --worker-startup-wait "$VLLM_MULTINODE_WORKER_STARTUP_WAIT"
      )
      if [[ -n "$LABELS" ]]; then
        vllm_multi_args+=(--labels "$LABELS")
      fi
      if [[ -n "$SSH_KEY" ]]; then
        vllm_multi_args+=(--ssh-key "$SSH_KEY")
      fi
      if [[ -n "$TP" ]]; then
        vllm_multi_args+=(--tp "$TP")
      fi
      if [[ -n "$VLLM_MULTINODE_NUM_PROMPTS" ]]; then
        vllm_multi_args+=(--num-prompts "$VLLM_MULTINODE_NUM_PROMPTS")
      fi
      if [[ -n "$VLLM_MULTINODE_IMAGE" ]]; then
        vllm_multi_args+=(--image "$VLLM_MULTINODE_IMAGE")
      fi
      if [[ -n "$SOCKET_IFNAME" ]]; then
        vllm_multi_args+=(--socket-ifname "$SOCKET_IFNAME")
      fi
      if [[ -n "$NCCL_IB_HCA" ]]; then
        vllm_multi_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
      fi
      run_step "vllm_serve_multinode_c${vllm_multi_conc}" "${ROOT_DIR}/scripts/repro/run_vllm_serve_multinode_container.sh" "${vllm_multi_args[@]}"
    done
  else
    echo "WARNING: --run-vllm-multinode requested, but only one host was provided; skipping." >&2
    fail=1
  fi
fi

# Benchmark C: GEMM sanity (all nodes)
gemm_args=(--run-id "$RUN_ID" --hosts "$HOSTS")
gemm_args+=(--ssh-user "$SSH_USER")
if [[ -n "$LABELS" ]]; then
  gemm_args+=(--labels "$LABELS")
fi
if [[ -n "$SSH_KEY" ]]; then
  gemm_args+=(--ssh-key "$SSH_KEY")
fi
run_step "gemm_sanity" "${ROOT_DIR}/scripts/run_gemm_sanity_all_nodes.sh" "${gemm_args[@]}"

# Optional FP4 checks: DeepGEMM FP8xFP4 smoke + grouped GEMM benchmark.
if [[ "$ENABLE_FP4" -eq 1 ]]; then
  fp4_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --runtime "$FP4_RUNTIME"
    --stack-profile "$FP4_STACK_PROFILE"
    --preset "$FP4_PRESET"
    --warmup "$FP4_WARMUP"
    --iters "$FP4_ITERS"
    --smoke-m "$FP4_SMOKE_M"
    --smoke-n "$FP4_SMOKE_N"
    --smoke-k "$FP4_SMOKE_K"
    --smoke-warmup "$FP4_SMOKE_WARMUP"
    --smoke-iters "$FP4_SMOKE_ITERS"
    --smoke-rounds "$FP4_SMOKE_ROUNDS"
    --smoke-skew-threshold-pct "$FP4_SMOKE_SKEW_THRESHOLD_PCT"
  )
  if [[ -n "$LABELS" ]]; then
    fp4_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    fp4_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ "$FP4_RUNTIME" == "container" ]]; then
    fp4_args+=(--image "$FP4_IMAGE")
  fi
  run_step "fp4_checks" "${ROOT_DIR}/scripts/run_fp4_checks_all_nodes.sh" "${fp4_args[@]}"
fi

# Optional: MAMF compute-depth scan (all nodes).
if [[ "$ENABLE_MAMF" -eq 1 ]]; then
  mamf_args=(--run-id "$RUN_ID" --hosts "$HOSTS" --ssh-user "$SSH_USER" --mode "$MAMF_MODE")
  if [[ -n "$LABELS" ]]; then
    mamf_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    mamf_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ "$MAMF_CONCURRENT" -eq 1 ]]; then
    mamf_args+=(--concurrent)
  fi
  run_step "mamf_finder" "${ROOT_DIR}/scripts/run_mamf_finder_all_nodes.sh" "${mamf_args[@]}"
fi

# Optional: per-iteration all-reduce jitter profile.
if [[ "$ENABLE_ALLREDUCE_STABILITY" -eq 1 ]]; then
  if [[ "${#HOST_ARR[@]}" -gt 1 && -z "$OOB_IF" && -z "$SOCKET_IFNAME" ]]; then
    echo "WARNING: --enable-allreduce-stability requested for multi-node but no --oob-if/--socket-ifname set; skipping." >&2
    fail=1
  else
    stability_args=(
      --run-id "$RUN_ID"
      --hosts "$HOSTS"
      --ssh-user "$SSH_USER"
      --label "allreduce_stability"
      --payload-gib "$ALLREDUCE_PAYLOAD_GIB"
      --iters "$ALLREDUCE_ITERS"
      --warmup "$ALLREDUCE_WARMUP"
    )
    if [[ -n "$SSH_KEY" ]]; then
      stability_args+=(--ssh-key "$SSH_KEY")
    fi
    if [[ -n "$OOB_IF" ]]; then
      stability_args+=(--oob-if "$OOB_IF")
    fi
    if [[ -n "$SOCKET_IFNAME" ]]; then
      stability_args+=(--socket-ifname "$SOCKET_IFNAME")
    fi
    if [[ -n "$NCCL_IB_HCA" ]]; then
      stability_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
    fi
    run_step "allreduce_stability" "${ROOT_DIR}/scripts/run_allreduce_stability.sh" "${stability_args[@]}"
  fi
fi

# Optional: all-reduce latency comparison (1x large vs many small).
if [[ "$ENABLE_ALLREDUCE_LATENCY_COMP" -eq 1 ]]; then
  if [[ "${#HOST_ARR[@]}" -gt 1 && -z "$OOB_IF" && -z "$SOCKET_IFNAME" ]]; then
    echo "WARNING: --enable-allreduce-latency-comp requested for multi-node but no --oob-if/--socket-ifname set; skipping." >&2
    fail=1
  else
    latency_args=(
      --run-id "$RUN_ID"
      --label "allreduce_latency_comp"
      --hosts "$HOSTS"
      --ssh-user "$SSH_USER"
      --payload-gib "$ALLREDUCE_LATENCY_PAYLOAD_GIB"
      --chunks "$ALLREDUCE_LATENCY_CHUNKS"
      --iters "$ALLREDUCE_LATENCY_ITERS"
      --warmup "$ALLREDUCE_LATENCY_WARMUP"
    )
    if [[ -n "$SSH_KEY" ]]; then
      latency_args+=(--ssh-key "$SSH_KEY")
    fi
    if [[ -n "$OOB_IF" ]]; then
      latency_args+=(--oob-if "$OOB_IF")
    fi
    if [[ -n "$SOCKET_IFNAME" ]]; then
      latency_args+=(--socket-ifname "$SOCKET_IFNAME")
    fi
    if [[ -n "$NCCL_IB_HCA" ]]; then
      latency_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
    fi
    run_step "allreduce_latency_comp" "${ROOT_DIR}/scripts/run_allreduce_latency_comp.sh" "${latency_args[@]}"
  fi
fi

# Optional: control-plane collectives (all_gather_object vs tensor collectives).
if [[ "$ENABLE_ALLGATHER_CONTROL_PLANE" -eq 1 ]]; then
  if [[ "${#HOST_ARR[@]}" -gt 1 && -z "$OOB_IF" && -z "$SOCKET_IFNAME" ]]; then
    echo "WARNING: --enable-allgather-control-plane requested for multi-node but no --oob-if/--socket-ifname set; skipping." >&2
    fail=1
  else
    allgather_args=(
      --run-id "$RUN_ID"
      --label "allgather_control_plane"
      --hosts "$HOSTS"
      --ssh-user "$SSH_USER"
      --iters "$ALLGATHER_CONTROL_ITERS"
      --warmup "$ALLGATHER_CONTROL_WARMUP"
    )
    if [[ -n "$SSH_KEY" ]]; then
      allgather_args+=(--ssh-key "$SSH_KEY")
    fi
    if [[ -n "$OOB_IF" ]]; then
      allgather_args+=(--oob-if "$OOB_IF")
    fi
    if [[ -n "$SOCKET_IFNAME" ]]; then
      allgather_args+=(--socket-ifname "$SOCKET_IFNAME")
    fi
    if [[ -n "$NCCL_IB_HCA" ]]; then
      allgather_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
    fi
    run_step "allgather_control_plane" "${ROOT_DIR}/scripts/run_allgather_control_plane.sh" "${allgather_args[@]}"
  fi
fi

# Optional: NCCL algorithm comparison (Ring/Tree/NVLS/auto).
if [[ "$ENABLE_NCCL_ALGO_COMPARISON" -eq 1 ]]; then
  if [[ "${#HOST_ARR[@]}" -gt 1 && -z "$OOB_IF" ]]; then
    echo "WARNING: --enable-nccl-algo-comparison requested for multi-node but --oob-if is not set; skipping." >&2
    fail=1
  else
    algo_args=(
      --run-id "$RUN_ID"
      --hosts "$HOSTS"
      --algos "$NCCL_ALGOS"
    )
    if [[ -n "$SSH_KEY" ]]; then
      algo_args+=(--ssh-key "$SSH_KEY")
    fi
    if [[ -n "$OOB_IF" ]]; then
      algo_args+=(--oob-if "$OOB_IF")
    fi
    if [[ -n "$SOCKET_IFNAME" ]]; then
      algo_args+=(--socket-ifname "$SOCKET_IFNAME")
    fi
    if [[ -n "$NCCL_IB_HCA" ]]; then
      algo_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
    fi
    run_step "nccl_algo_comparison" "${ROOT_DIR}/scripts/run_nccl_algo_comparison.sh" "${algo_args[@]}"
  fi
fi

# Optional: CPU<->GPU C2C memcpy benchmark (local/primary node)
if [[ "$RUN_C2C" -eq 1 ]]; then
  run_step "c2c_memcpy" "${ROOT_DIR}/scripts/run_c2c_memcpy_bench.sh" \
    --run-id "$RUN_ID" \
    --label "$PRIMARY_LABEL" \
    --device "$C2C_DEVICE" \
    --bw-sizes "$C2C_BW_SIZES" \
    --lat-sizes "$C2C_LAT_SIZES" \
    --bw-iters "$C2C_BW_ITERS" \
    --lat-iters "$C2C_LAT_ITERS" \
    --warmup "$C2C_WARMUP"
fi

# Optional: NUMA memory-bandwidth probe (all nodes)
if [[ "$RUN_NUMA_MEM_BW" -eq 1 ]]; then
  numa_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --bytes "$NUMA_BYTES"
    --iters "$NUMA_ITERS"
    --threads "$NUMA_THREADS"
    --warmup "$NUMA_WARMUP"
  )
  if [[ -n "$LABELS" ]]; then
    numa_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    numa_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ -n "$NUMA_NODES" ]]; then
    numa_args+=(--nodes "$NUMA_NODES")
  fi
  if [[ -n "$NUMA_CPU_NODE" ]]; then
    numa_args+=(--cpu-node "$NUMA_CPU_NODE")
  fi
  run_step "numa_mem_bw" "${ROOT_DIR}/scripts/run_numa_mem_bw_all_nodes.sh" "${numa_args[@]}"
fi

# Optional: end-to-end transformer train-step benchmark.
if [[ "$RUN_TRAIN_STEP" -eq 1 ]]; then
  train_common=(
    --run-id "$RUN_ID"
    --ssh-user "$SSH_USER"
    --gpus-per-node "$(nvidia-smi -L | wc -l | tr -d ' ')"
    --master-port "$TRAIN_MASTER_PORT"
    --steps "$TRAIN_STEPS"
    --warmup-steps "$TRAIN_WARMUP_STEPS"
    --batch-size "$TRAIN_BATCH_SIZE"
    --seq-len "$TRAIN_SEQ_LEN"
    --hidden "$TRAIN_HIDDEN"
    --layers "$TRAIN_LAYERS"
    --heads "$TRAIN_HEADS"
    --mlp-ratio "$TRAIN_MLP_RATIO"
    --precision "$TRAIN_PRECISION"
    --fsdp "$TRAIN_FSDP"
    --lr "$TRAIN_LR"
  )
  if [[ -n "$SSH_KEY" ]]; then
    train_common+=(--ssh-key "$SSH_KEY")
  fi
  if [[ -n "$OOB_IF" ]]; then
    train_common+=(--oob-if "$OOB_IF")
  fi
  if [[ -n "$NCCL_IB_HCA" ]]; then
    train_common+=(--nccl-ib-hca "$NCCL_IB_HCA")
  fi
  if [[ -n "$NCCL_NVLS_ENABLE" ]]; then
    train_common+=(--nccl-nvls-enable "$NCCL_NVLS_ENABLE")
  fi

  if [[ "$TRAIN_STEP_SINGLE_NODE" -eq 1 ]]; then
    run_step "train_step_single_node" "${ROOT_DIR}/scripts/run_torchrun_transformer_train_step.sh" \
      "${train_common[@]}" \
      --hosts "${HOST_ARR[0]}" \
      --label "${PRIMARY_LABEL}_single_node"
  fi

  if [[ "$TRAIN_STEP_MULTI_NODE" -eq 1 ]]; then
    if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
      run_step "train_step_multi_node" "${ROOT_DIR}/scripts/run_torchrun_transformer_train_step.sh" \
        "${train_common[@]}" \
        --hosts "$HOSTS" \
        --label "${PRIMARY_LABEL}_multinode"
    else
      echo "WARNING: --run-train-step requested multi-node run, but only one host provided; skipping multi-node train-step." >&2
    fi
  fi
fi

# Optional: checkpoint-like write/read benchmark (all nodes).
if [[ "$RUN_CHECKPOINT_IO" -eq 1 ]]; then
  ckpt_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --test-dir "$CHECKPOINT_TEST_DIR"
    --bytes "$CHECKPOINT_BYTES"
    --block-size "$CHECKPOINT_BLOCK_SIZE"
    --files "$CHECKPOINT_FILES"
    --fsync "$CHECKPOINT_FSYNC"
    --write 1
    --read 1
  )
  if [[ -n "$LABELS" ]]; then
    ckpt_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    ckpt_args+=(--ssh-key "$SSH_KEY")
  fi
  run_step "checkpoint_io" "${ROOT_DIR}/scripts/run_checkpoint_io_all_nodes.sh" "${ckpt_args[@]}"
fi

# Storage: fio (all nodes)
fio_args=(
  --run-id "$RUN_ID"
  --hosts "$HOSTS"
  --ssh-user "$SSH_USER"
  --test-dir "$FIO_TEST_DIR"
  --runtime "$FIO_RUNTIME"
)
if [[ -n "$LABELS" ]]; then
  fio_args+=(--labels "$LABELS")
fi
if [[ -n "$SSH_KEY" ]]; then
  fio_args+=(--ssh-key "$SSH_KEY")
fi
run_step "fio_all_nodes" "${ROOT_DIR}/scripts/run_fio_all_nodes.sh" "${fio_args[@]}"

# Optional: nvbandwidth bundle (all nodes)
if [[ "$RUN_NVBANDWIDTH" -eq 1 ]]; then
  nvbw_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --runtime "$NVBANDWIDTH_RUNTIME"
  )
  if [[ -n "$LABELS" ]]; then
    nvbw_args+=(--labels "$LABELS")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    nvbw_args+=(--ssh-key "$SSH_KEY")
  fi
  if [[ "$NVBANDWIDTH_RUNTIME" == "container" ]]; then
    nvbw_args+=(--image "$NVBANDWIDTH_IMAGE")
  else
    nvbw_args+=(--nvbw-bin "$NVBANDWIDTH_BIN")
  fi
  if [[ "$NVBANDWIDTH_QUICK" -eq 1 ]]; then
    nvbw_args+=(--quick)
  fi
  run_step "nvbandwidth_all_nodes" "${ROOT_DIR}/scripts/run_nvbandwidth_bundle_all_nodes.sh" "${nvbw_args[@]}"
fi

# Plotting (best-effort)
if [[ -f "${ROOT_DIR}/results/structured/${RUN_ID}_node1_nccl.json" ]]; then
  run_step "plot_nccl_single_node" python3 "${ROOT_DIR}/analysis/plot_nccl.py" \
    --input "${ROOT_DIR}/results/structured/${RUN_ID}_node1_nccl.json" \
    --out-dir "${ROOT_DIR}/docs/figures" \
    --run-id "${RUN_ID}_node1"
else
  echo "WARNING: skipping plot_nccl_single_node; missing ${ROOT_DIR}/results/structured/${RUN_ID}_node1_nccl.json" >&2
fi

if [[ "${#HOST_ARR[@]}" -gt 1 && -f "${ROOT_DIR}/results/structured/${RUN_ID}_2nodes_nccl.json" ]]; then
  nccl_plot_multi_args=(
    --input "${ROOT_DIR}/results/structured/${RUN_ID}_2nodes_nccl.json"
    --out-dir "${ROOT_DIR}/docs/figures"
    --run-id "${RUN_ID}_2nodes"
  )
  if [[ -f "${ROOT_DIR}/results/structured/${RUN_ID}_node1_nccl.json" ]]; then
    nccl_plot_multi_args+=(--baseline-input "${ROOT_DIR}/results/structured/${RUN_ID}_node1_nccl.json")
  fi
  run_step "plot_nccl_multi_node" python3 "${ROOT_DIR}/analysis/plot_nccl.py" "${nccl_plot_multi_args[@]}"
fi

if [[ -f "${ROOT_DIR}/results/structured/${RUN_ID}_nccl_env_sensitivity.json" ]]; then
  run_step "plot_nccl_env_sensitivity" python3 "${ROOT_DIR}/analysis/plot_nccl_env_sensitivity.py" \
    --input "${ROOT_DIR}/results/structured/${RUN_ID}_nccl_env_sensitivity.json" \
    --output "${ROOT_DIR}/docs/figures/${RUN_ID}_nccl_env_sensitivity.png" \
    --title "NCCL Env Sensitivity ${RUN_ID}"
fi

if [[ -f "${ROOT_DIR}/results/structured/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_sweep.csv" ]]; then
  run_step "plot_vllm_serve" python3 "${ROOT_DIR}/analysis/plot_vllm_serve_sweep.py" \
    --input "${ROOT_DIR}/results/structured/${RUN_ID}_${PRIMARY_LABEL}_vllm_serve_sweep.csv" \
    --out-dir "${ROOT_DIR}/docs/figures" \
    --run-id "${RUN_ID}_${PRIMARY_LABEL}"
fi

if [[ "$RUN_VLLM_MULTINODE" -eq 1 ]]; then
  vllm_multi_label="$(label_for_index 0)"
  if [[ -f "${ROOT_DIR}/results/structured/${RUN_ID}_${vllm_multi_label}_vllm_multinode_serve.csv" ]]; then
    run_step "plot_vllm_serve_multinode" python3 "${ROOT_DIR}/analysis/plot_vllm_serve_sweep.py" \
      --input "${ROOT_DIR}/results/structured/${RUN_ID}_${vllm_multi_label}_vllm_multinode_serve.csv" \
      --out-dir "${ROOT_DIR}/docs/figures" \
      --run-id "${RUN_ID}_${vllm_multi_label}_multinode"
  fi
fi

if [[ "${#HOST_ARR[@]}" -gt 1 && "$HEALTH_SUITE_MODE" != "off" ]]; then
  shopt -s nullglob
  suite_summaries=( "${ROOT_DIR}/results/structured/${RUN_ID}_health_suite_${HEALTH_SUITE_MODE}_"*"_cluster_health_suite_summary.json" )
  shopt -u nullglob
  if [[ "${#suite_summaries[@]}" -gt 0 ]]; then
    run_step "plot_iperf3_oob" python3 "${ROOT_DIR}/analysis/plot_iperf3.py" \
      --summary "${suite_summaries[0]}" \
      --out "${ROOT_DIR}/docs/figures/${RUN_ID}_iperf3_oob_tcp.png" \
      --out-json "${ROOT_DIR}/results/structured/${RUN_ID}_iperf3_oob_tcp.json" \
      --skip-if-missing
  fi
fi

shopt -s nullglob
gemm_inputs=( "${ROOT_DIR}/results/structured/${RUN_ID}_"*_gemm_gpu_sanity.csv )
shopt -u nullglob
if [[ "${#gemm_inputs[@]}" -gt 0 ]]; then
  run_step "plot_gemm_sanity" python3 "${ROOT_DIR}/analysis/plot_gemm_bar.py" \
    --inputs "${gemm_inputs[@]}" \
    --output "${ROOT_DIR}/docs/figures/${RUN_ID}_gemm_gpu_sanity.png" \
    --filter-m 16384
fi

if [[ "$ENABLE_MAMF" -eq 1 ]]; then
  shopt -s nullglob
  mamf_summaries=( "${ROOT_DIR}/results/structured/${RUN_ID}_"*_mamf_summary.json )
  shopt -u nullglob
  if [[ "${#mamf_summaries[@]}" -gt 0 ]]; then
    run_step "plot_mamf" python3 "${ROOT_DIR}/analysis/plot_mamf.py" \
      --summary-inputs "${mamf_summaries[@]}" \
      --output "${ROOT_DIR}/docs/figures/${RUN_ID}_mamf_straggler.png" \
      --mode straggler \
      --title "MAMF Straggler View ${RUN_ID}"
  fi
fi

if [[ -f "${ROOT_DIR}/results/structured/${RUN_ID}_allreduce_stability.json" ]]; then
  run_step "plot_allreduce_stability" python3 "${ROOT_DIR}/analysis/plot_allreduce_stability.py" \
    --input "${ROOT_DIR}/results/structured/${RUN_ID}_allreduce_stability.json" \
    --output "${ROOT_DIR}/docs/figures/${RUN_ID}_allreduce_stability.png"
fi

if [[ -f "${ROOT_DIR}/results/structured/${RUN_ID}_allreduce_latency_comp.json" ]]; then
  run_step "plot_allreduce_latency_comp" python3 "${ROOT_DIR}/analysis/plot_allreduce_latency_comp.py" \
    --input "${ROOT_DIR}/results/structured/${RUN_ID}_allreduce_latency_comp.json" \
    --output "${ROOT_DIR}/docs/figures/${RUN_ID}_allreduce_latency_comp.png"
fi

if [[ -f "${ROOT_DIR}/results/structured/${RUN_ID}_allgather_control_plane.json" ]]; then
  run_step "plot_allgather_control_plane" python3 "${ROOT_DIR}/analysis/plot_allgather_control_plane.py" \
    --input "${ROOT_DIR}/results/structured/${RUN_ID}_allgather_control_plane.json" \
    --output "${ROOT_DIR}/docs/figures/${RUN_ID}_allgather_control_plane.png"
fi

if [[ "$ENABLE_NCCL_ALGO_COMPARISON" -eq 1 ]]; then
  shopt -s nullglob
  nccl_algo_inputs=( "${ROOT_DIR}/results/structured/${RUN_ID}_nccl_algo_"*.json )
  shopt -u nullglob
  if [[ "${#nccl_algo_inputs[@]}" -gt 0 ]]; then
    run_step "plot_nccl_algo_comparison" python3 "${ROOT_DIR}/analysis/plot_nccl_algo_comparison.py" \
      --inputs "${nccl_algo_inputs[@]}" \
      --output "${ROOT_DIR}/docs/figures/${RUN_ID}_nccl_algo_comparison.png" \
      --title "NCCL Algorithm Comparison ${RUN_ID}"
  fi
fi

shopt -s nullglob
fio_inputs=( "${ROOT_DIR}/results/structured/${RUN_ID}_"*_fio.json )
shopt -u nullglob
for fio_in in "${fio_inputs[@]}"; do
  fio_base="$(basename "$fio_in" .json)"
  run_step "plot_fio_${fio_base}" python3 "${ROOT_DIR}/analysis/plot_fio.py" \
    --input "$fio_in" \
    --out "${ROOT_DIR}/docs/figures/${fio_base}.png"
done

if [[ "$RUN_NVBANDWIDTH" -eq 1 ]]; then
  shopt -s nullglob
  nvbw_sums_inputs=( "${ROOT_DIR}/results/structured/${RUN_ID}_"*_nvbandwidth_sums.csv )
  shopt -u nullglob
  for nvbw_csv in "${nvbw_sums_inputs[@]}"; do
    nvbw_base="$(basename "$nvbw_csv" _sums.csv)"
    run_step "plot_nvbandwidth_${nvbw_base}" python3 "${ROOT_DIR}/analysis/plot_nvbandwidth_sums.py" \
      --input "$nvbw_csv" \
      --out "${ROOT_DIR}/docs/figures/${nvbw_base}_sums.png" \
      --title "nvbandwidth SUM metrics ${nvbw_base}"
  done
fi

if [[ -f "${ROOT_DIR}/results/structured/${RUN_ID}_${PRIMARY_LABEL}_c2c_memcpy.json" ]]; then
  run_step "plot_c2c_memcpy" python3 "${ROOT_DIR}/analysis/plot_c2c_memcpy.py" \
    --input "${ROOT_DIR}/results/structured/${RUN_ID}_${PRIMARY_LABEL}_c2c_memcpy.json" \
    --out-dir "${ROOT_DIR}/docs/figures" \
    --run-id "${RUN_ID}_${PRIMARY_LABEL}"
fi

if [[ "$RUN_NUMA_MEM_BW" -eq 1 ]]; then
  shopt -s nullglob
  numa_inputs=( "${ROOT_DIR}/results/structured/${RUN_ID}_"*_numa_mem_bw.json )
  shopt -u nullglob
  for inp in "${numa_inputs[@]}"; do
    base="$(basename "$inp" .json)"
    run_step "plot_numa_mem_bw_${base}" python3 "${ROOT_DIR}/analysis/plot_numa_mem_bw.py" \
      --input "$inp" \
      --out "${ROOT_DIR}/docs/figures/${base}.png" \
      --title "NUMA memcpy bandwidth: ${base}"
  done
fi

if [[ -f "${ROOT_DIR}/results/structured/${RUN_ID}_${PRIMARY_LABEL}_single_node_torchrun_train_step.json" ]]; then
  run_step "plot_train_step_single" python3 "${ROOT_DIR}/analysis/plot_torchrun_train_step.py" \
    --input "${ROOT_DIR}/results/structured/${RUN_ID}_${PRIMARY_LABEL}_single_node_torchrun_train_step.json" \
    --out "${ROOT_DIR}/docs/figures/${RUN_ID}_${PRIMARY_LABEL}_single_node_torchrun_train_step.png"
fi

if [[ -f "${ROOT_DIR}/results/structured/${RUN_ID}_${PRIMARY_LABEL}_multinode_torchrun_train_step.json" ]]; then
  run_step "plot_train_step_multi" python3 "${ROOT_DIR}/analysis/plot_torchrun_train_step.py" \
    --input "${ROOT_DIR}/results/structured/${RUN_ID}_${PRIMARY_LABEL}_multinode_torchrun_train_step.json" \
    --out "${ROOT_DIR}/docs/figures/${RUN_ID}_${PRIMARY_LABEL}_multinode_torchrun_train_step.png"
fi

shopt -s nullglob
meta_inputs=( "${ROOT_DIR}/results/structured/${RUN_ID}_"*_meta.json )
shopt -u nullglob
for meta_in in "${meta_inputs[@]}"; do
  if [[ "$meta_in" == *_cluster_meta.json ]]; then
    continue
  fi
  meta_base="$(basename "$meta_in" .json)"
  run_step "plot_nvlink_topology_${meta_base}" python3 "${ROOT_DIR}/analysis/plot_nvlink_topology.py" \
    --meta "$meta_in" \
    --fig-out "${ROOT_DIR}/docs/figures/${meta_base}_nvlink_topology.png" \
    --summary-out "${ROOT_DIR}/results/structured/${meta_base}_nvlink_topology.json"
done

dashboard_labels=("${PRIMARY_LABEL}")
if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  if [[ -n "$LABELS" && "${#LABEL_ARR[@]}" -gt 1 ]]; then
    dashboard_labels+=("$(echo "${LABEL_ARR[1]}" | xargs)")
  else
    dashboard_labels+=("$(echo "${HOST_ARR[1]}" | xargs)")
  fi
fi
dashboard_node_labels_csv="$(IFS=','; echo "${dashboard_labels[*]}")"
run_step "plot_cluster_story_dashboard" python3 "${ROOT_DIR}/analysis/plot_cluster_story_dashboard.py" \
  --run-id "${RUN_ID}" \
  --structured-dir "${ROOT_DIR}/results/structured" \
  --node-labels "${dashboard_node_labels_csv}" \
  --fig-out "${ROOT_DIR}/docs/figures/${RUN_ID}_cluster_story_dashboard.png" \
  --summary-out "${ROOT_DIR}/results/structured/${RUN_ID}_node_parity_summary.json"

run_step "validate_required_artifacts" validate_required_artifacts

manifest_args=(--run-id "$RUN_ID" --hosts "$HOSTS" --include-figures)
if [[ -n "$LABELS" ]]; then
  manifest_args+=(--labels "$LABELS")
fi
run_step "manifest_refresh" python3 "${ROOT_DIR}/scripts/write_manifest.py" "${manifest_args[@]}"

echo ""
echo "========================================"
echo "Suite Complete"
echo "========================================"
echo "Manifest: results/structured/${RUN_ID}_manifest.json"
echo "Field report template: docs/field-report-template.md"

if [[ "$fail" -ne 0 ]]; then
  echo "STATUS: FAILED (one or more steps failed)" >&2
  exit 1
fi
echo "STATUS: OK"
