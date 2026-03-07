#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
LABEL="${LABEL:-$(hostname)}"
if [[ -z "${LABEL}" || "${LABEL}" == "localhost" ]]; then
  LABEL="$(hostname)"
fi
DURATION="${DURATION:-600}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_RAW="${CLUSTER_RAW_DIR_EFFECTIVE}"

mkdir -p "$OUT_RAW"

QUERY_OUT="${OUT_RAW}/${RUN_ID}_${LABEL}_gpu_telemetry_query.csv"
PMON_OUT="${OUT_RAW}/${RUN_ID}_${LABEL}_gpu_telemetry_pmon.log"

# Avoid clobbering prior logs (and avoid deletes); if the target exists, suffix it.
if [[ -e "$QUERY_OUT" ]]; then
  QUERY_OUT="${OUT_RAW}/${RUN_ID}_${LABEL}_gpu_telemetry_query_$(date +%s).csv"
fi
if [[ -e "$PMON_OUT" ]]; then
  PMON_OUT="${OUT_RAW}/${RUN_ID}_${LABEL}_gpu_telemetry_pmon_$(date +%s).log"
fi

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,clocks_event_reasons.sw_power_cap \
  --format=csv > "$QUERY_OUT"

for i in $(seq 1 "$DURATION"); do
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,clocks_event_reasons.sw_power_cap \
    --format=csv,noheader >> "$QUERY_OUT"
  nvidia-smi pmon -c 1 >> "$PMON_OUT"
  sleep 1
done

echo "Wrote $QUERY_OUT and $PMON_OUT"
