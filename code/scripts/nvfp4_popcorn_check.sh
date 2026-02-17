#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/nvfp4_popcorn_check.sh <submission.py> [test|benchmark]

Runs Popcorn against nvfp4_group_gemm on B200 without creating a real leaderboard
submission flow in this repo, and stores raw + parsed artifacts under:
  artifacts/popcorn_checks/<timestamp>/
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

submission_file="$1"
mode="${2:-benchmark}"

if [[ ! -f "${submission_file}" ]]; then
  echo "Submission file not found: ${submission_file}" >&2
  exit 1
fi

if [[ "${mode}" != "test" && "${mode}" != "benchmark" ]]; then
  echo "Mode must be 'test' or 'benchmark', got: ${mode}" >&2
  exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="artifacts/popcorn_checks/${timestamp}"
mkdir -p "${out_dir}"
raw_log="${out_dir}/popcorn_${mode}.log"

cmd=(
  popcorn submit
  --no-tui
  --leaderboard nvfp4_group_gemm
  --gpu B200
  --mode "${mode}"
  "${submission_file}"
)

echo "Running: ${cmd[*]}"
"${cmd[@]}" | tee "${raw_log}"

if [[ "${mode}" == "benchmark" ]]; then
  parsed_json="${out_dir}/benchmark_parsed.json"
  python - "${raw_log}" "${parsed_json}" "${submission_file}" <<'PY'
import ast
import json
import re
import sys
from pathlib import Path

raw_log = Path(sys.argv[1])
out_json = Path(sys.argv[2])
submission = sys.argv[3]
text = raw_log.read_text()

def to_us(value: float, unit: str) -> float:
    norm = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    if norm == "ms":
        return float(value) * 1000.0
    if norm == "us":
        return float(value)
    raise ValueError(f"Unsupported time unit: {unit!r}")

match = re.search(r"Response:\s*(\[[^\]]+\])", text)
if match:
    values = ast.literal_eval(match.group(1))
    if not isinstance(values, list):
        values = [values]

    groups_by_case = [8, 8, 2, 2]
    cases = []
    for idx, total_us in enumerate(values):
        group_count = groups_by_case[idx] if idx < len(groups_by_case) else None
        per_group = (float(total_us) / group_count) if group_count else None
        cases.append(
            {
                "case_index": idx,
                "total_us": float(total_us),
                "groups": group_count,
                "per_group_us": per_group,
            }
        )

    payload = {
        "ok": True,
        "format": "response_list",
        "submission_file": submission,
        "raw_log": str(raw_log),
        "response_values": [float(v) for v in values],
        "cases": cases,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    sys.exit(0)

bench_block_re = re.compile(
    r"g:\s*(?P<g>\d+);.*?\n"
    r"\s*⏱\s*(?P<mean>[0-9.]+)\s*±\s*(?P<std>[0-9.]+)\s*(?P<mean_unit>ms|us|µs)\n"
    r"\s*⚡\s*(?P<fast>[0-9.]+)\s*(?P<fast_unit>ms|us|µs)\s*🐌\s*(?P<slow>[0-9.]+)\s*(?P<slow_unit>ms|us|µs)",
    flags=re.DOTALL,
)

cases = []
for idx, block in enumerate(bench_block_re.finditer(text)):
    group_count = int(block.group("g"))
    mean_total_us = to_us(float(block.group("mean")), block.group("mean_unit"))
    std_total_us = to_us(float(block.group("std")), block.group("mean_unit"))
    fast_total_us = to_us(float(block.group("fast")), block.group("fast_unit"))
    slow_total_us = to_us(float(block.group("slow")), block.group("slow_unit"))
    cases.append(
        {
            "case_index": idx,
            "mean_total_us": mean_total_us,
            "std_total_us": std_total_us,
            "fast_total_us": fast_total_us,
            "slow_total_us": slow_total_us,
            "groups": group_count,
            "per_group_us": (mean_total_us / group_count),
        }
    )

payload = {
    "ok": len(cases) > 0,
    "format": "benchmark_blocks" if len(cases) > 0 else "unknown",
    "submission_file": submission,
    "raw_log": str(raw_log),
    "cases": cases,
}
if not cases:
    payload["error"] = "Could not parse benchmark blocks from popcorn output"
out_json.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY
  echo "Parsed benchmark artifact: ${parsed_json}"
fi

echo "Raw log: ${raw_log}"
