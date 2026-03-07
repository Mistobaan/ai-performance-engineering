#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_fio_bench.sh [options]

Runs a small fio suite (sequential + random) and writes:
  - raw fio JSON logs under results/raw/
  - a structured summary JSON + CSV under results/structured/

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --label <label>        Label for output paths (default: hostname)
  --test-dir <path>      Directory to run fio in (default: /tmp)
  --file-size <size>     Size per job (default: 8G)
  --runtime <sec>        Runtime per test (default: 60)
  --repeats <n>          Number of repetitions per fio pattern (default: 1)
  --numjobs <n>          Number of jobs (default: 4)
  --iodepth <n>          iodepth for random tests (default: 64)
  --bs-seq <size>        Block size for sequential tests (default: 1M)
  --bs-rand <size>       Block size for random tests (default: 4K)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
LABEL="${LABEL:-$(hostname)}"
TEST_DIR="${TEST_DIR:-/tmp}"
FILE_SIZE="${FILE_SIZE:-8G}"
RUNTIME="${RUNTIME:-60}"
REPEATS="${REPEATS:-1}"
NUMJOBS="${NUMJOBS:-4}"
IODEPTH="${IODEPTH:-64}"
BS_SEQ="${BS_SEQ:-1M}"
BS_RAND="${BS_RAND:-4K}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --test-dir) TEST_DIR="$2"; shift 2 ;;
    --file-size) FILE_SIZE="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --numjobs) NUMJOBS="$2"; shift 2 ;;
    --iodepth) IODEPTH="$2"; shift 2 ;;
    --bs-seq) BS_SEQ="$2"; shift 2 ;;
    --bs-rand) BS_RAND="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v fio >/dev/null 2>&1; then
  echo "ERROR: fio not found in PATH." >&2
  exit 1
fi
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --repeats must be a positive integer (got: ${REPEATS})." >&2
  exit 2
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"

OUT_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}/${RUN_ID}_${LABEL}_fio"
OUT_STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_RAW_DIR" "$OUT_STRUCT_DIR"

WORK_DIR="${TEST_DIR%/}/aisp_fio_${RUN_ID}_${LABEL}"
mkdir -p "$WORK_DIR"

echo "========================================"
echo "fio Storage Benchmark"
echo "========================================"
echo "Date: $(date -Is)"
echo "Host: $(hostname)"
echo "RUN_ID: $RUN_ID"
echo "Label: $LABEL"
echo "Test dir: $WORK_DIR"
echo "File size: $FILE_SIZE"
echo "Runtime: ${RUNTIME}s"
echo "Repeats: ${REPEATS}"
echo "Numjobs: $NUMJOBS"
echo "Rand iodepth: $IODEPTH"
echo "Seq BS: $BS_SEQ"
echo "Rand BS: $BS_RAND"
echo ""
df -hT "$WORK_DIR" || true
echo ""

run_fio() {
  local name="$1"
  local rw="$2"
  local bs="$3"
  local extra=("${@:4}")
  local out_json="${OUT_RAW_DIR}/${name}.json"

  echo "== fio ${name} (${rw}, bs=${bs}) =="
  fio \
    --name="$name" \
    --directory="$WORK_DIR" \
    --ioengine=libaio \
    --direct=1 \
    --rw="$rw" \
    --bs="$bs" \
    --size="$FILE_SIZE" \
    --numjobs="$NUMJOBS" \
    --runtime="$RUNTIME" \
    --time_based \
    --group_reporting \
    --unlink=1 \
    --output-format=json \
    --output="$out_json" \
    "${extra[@]}"
  echo "Wrote $out_json"
}

for rep in $(seq 1 "$REPEATS"); do
  echo "========================================"
  echo "fio repetition ${rep}/${REPEATS}"
  echo "========================================"
  run_fio "seq_write_r${rep}" "write" "$BS_SEQ"
  run_fio "seq_read_r${rep}" "read" "$BS_SEQ"
  run_fio "rand_read_r${rep}" "randread" "$BS_RAND" --iodepth="$IODEPTH"
  run_fio "rand_write_r${rep}" "randwrite" "$BS_RAND" --iodepth="$IODEPTH"
done

SUMMARY_JSON="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_fio.json"
SUMMARY_CSV="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_fio.csv"
STABILITY_JSON="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_fio_stability.json"

python3 - <<'PY' "$SUMMARY_JSON" "$SUMMARY_CSV" "$STABILITY_JSON" "$RUN_ID" "$LABEL" "$WORK_DIR" "$FILE_SIZE" "$RUNTIME" "$REPEATS" "$NUMJOBS" "$IODEPTH" "$BS_SEQ" "$BS_RAND" "$OUT_RAW_DIR"
import csv
import json
import math
from statistics import mean, median, pstdev
import sys
from pathlib import Path

summary_json = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
stability_json = Path(sys.argv[3])
run_id = sys.argv[4]
label = sys.argv[5]
work_dir = sys.argv[6]
file_size = sys.argv[7]
runtime_s = int(sys.argv[8])
repeats = int(sys.argv[9])
numjobs = int(sys.argv[10])
iodepth = int(sys.argv[11])
bs_seq = sys.argv[12]
bs_rand = sys.argv[13]
raw_dir = Path(sys.argv[14])

def load(name: str, rep: int):
    return json.loads((raw_dir / f"{name}_r{rep}.json").read_text())

def job_metrics(d, op: str):
    # Aggregate bandwidth/iops across jobs via group_reporting (job 0 contains group).
    jobs = d.get("jobs") or []
    if not jobs:
        return {}
    j = jobs[0]
    sect = j.get(op, {}) if isinstance(j, dict) else {}
    bw_bytes = float(sect.get("bw_bytes", 0.0) or 0.0)
    iops = float(sect.get("iops", 0.0) or 0.0)
    lat_ns = sect.get("lat_ns", {}) if isinstance(sect, dict) else {}
    p99_ns = None
    if isinstance(lat_ns, dict):
        pct = lat_ns.get("percentile", {})
        if isinstance(pct, dict):
            # fio uses string keys like "99.000000"
            for k in ("99.000000", "99.0"):
                if k in pct:
                    p99_ns = float(pct[k])
                    break
    return {
        "bw_mb_s": bw_bytes / 1024 / 1024,
        "iops": iops,
        "p99_lat_ms": (p99_ns / 1e6) if p99_ns is not None else None,
    }

def stats(vals):
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return {"count": 0, "mean": None, "median": None, "stdev": None, "cv_pct": None, "min": None, "max": None}
    m = mean(xs)
    sd = pstdev(xs) if len(xs) > 1 else 0.0
    cv = (sd / m * 100.0) if m != 0 else None
    return {
        "count": len(xs),
        "mean": m,
        "median": median(xs),
        "stdev": sd,
        "cv_pct": cv,
        "min": min(xs),
        "max": max(xs),
    }

def aggregate_case(case_name: str, op_name: str):
    per_rep = []
    for rep in range(1, repeats + 1):
        d = load(case_name, rep)
        per_rep.append(job_metrics(d, op_name))
    bw_s = stats([r.get("bw_mb_s") for r in per_rep])
    iops_s = stats([r.get("iops") for r in per_rep])
    p99_s = stats([r.get("p99_lat_ms") for r in per_rep])
    return {
        "bw_mb_s": bw_s["median"],
        "iops": iops_s["median"],
        "p99_lat_ms": p99_s["median"],
        "repeat_statistics": {
            "bw_mb_s": bw_s,
            "iops": iops_s,
            "p99_lat_ms": p99_s,
        },
        "per_repeat": per_rep,
    }

seq_w = aggregate_case("seq_write", "write")
seq_r = aggregate_case("seq_read", "read")
rr = aggregate_case("rand_read", "read")
rw = aggregate_case("rand_write", "write")

payload = {
    "run_id": run_id,
    "label": label,
    "work_dir": work_dir,
    "file_size": file_size,
    "runtime_s": runtime_s,
    "repeats": repeats,
    "numjobs": numjobs,
    "iodepth": iodepth,
    "bs_seq": bs_seq,
    "bs_rand": bs_rand,
    "results": {
        "seq_write": seq_w,
        "seq_read": seq_r,
        "rand_read": rr,
        "rand_write": rw,
    },
    "raw": {
        "seq_write_jsons": [str(raw_dir / f"seq_write_r{rep}.json") for rep in range(1, repeats + 1)],
        "seq_read_jsons": [str(raw_dir / f"seq_read_r{rep}.json") for rep in range(1, repeats + 1)],
        "rand_read_jsons": [str(raw_dir / f"rand_read_r{rep}.json") for rep in range(1, repeats + 1)],
        "rand_write_jsons": [str(raw_dir / f"rand_write_r{rep}.json") for rep in range(1, repeats + 1)],
    },
}

stability = {
    "run_id": run_id,
    "label": label,
    "repeats": repeats,
    "summary": {
        "seq_read_bw_cv_pct": payload["results"]["seq_read"]["repeat_statistics"]["bw_mb_s"]["cv_pct"],
        "seq_write_bw_cv_pct": payload["results"]["seq_write"]["repeat_statistics"]["bw_mb_s"]["cv_pct"],
        "rand_read_iops_cv_pct": payload["results"]["rand_read"]["repeat_statistics"]["iops"]["cv_pct"],
        "rand_write_iops_cv_pct": payload["results"]["rand_write"]["repeat_statistics"]["iops"]["cv_pct"],
    },
    "results": payload["results"],
}

summary_json.parent.mkdir(parents=True, exist_ok=True)
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
stability_json.write_text(json.dumps(stability, indent=2, sort_keys=True))

cols = [
    "run_id",
    "label",
    "file_size",
    "runtime_s",
    "repeats",
    "numjobs",
    "iodepth",
    "seq_write_bw_mb_s",
    "seq_read_bw_mb_s",
    "rand_read_iops",
    "rand_write_iops",
    "rand_read_p99_lat_ms",
    "rand_write_p99_lat_ms",
]

row = [
    run_id,
    label,
    file_size,
    runtime_s,
    repeats,
    numjobs,
    iodepth,
    payload["results"]["seq_write"].get("bw_mb_s", ""),
    payload["results"]["seq_read"].get("bw_mb_s", ""),
    payload["results"]["rand_read"].get("iops", ""),
    payload["results"]["rand_write"].get("iops", ""),
    payload["results"]["rand_read"].get("p99_lat_ms", ""),
    payload["results"]["rand_write"].get("p99_lat_ms", ""),
]

write_header = True
if summary_csv.exists():
    try:
        if summary_csv.read_text(encoding="utf-8").strip():
            write_header = False
    except Exception:
        pass

with summary_csv.open("a", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(cols)
    w.writerow(row)

print(summary_json)
print(summary_csv)
print(stability_json)
PY

echo "Wrote ${SUMMARY_JSON}"
echo "Wrote ${SUMMARY_CSV}"
echo "Wrote ${STABILITY_JSON}"
