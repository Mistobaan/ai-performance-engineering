#!/usr/bin/env bash
set -euo pipefail

cluster_run_dir_for_root() {
  local root_dir="$1"
  local run_id="${2:-${RUN_ID:-$(date +%Y-%m-%d)}}"
  printf '%s\n' "${root_dir}/runs/${run_id}"
}

cluster_structured_dir_for_root() {
  local root_dir="$1"
  local run_id="${2:-${RUN_ID:-$(date +%Y-%m-%d)}}"
  printf '%s\n' "${CLUSTER_RESULTS_STRUCTURED_DIR:-$(cluster_run_dir_for_root "${root_dir}" "${run_id}")/structured}"
}

cluster_raw_dir_for_root() {
  local root_dir="$1"
  local run_id="${2:-${RUN_ID:-$(date +%Y-%m-%d)}}"
  printf '%s\n' "${CLUSTER_RESULTS_RAW_DIR:-$(cluster_run_dir_for_root "${root_dir}" "${run_id}")/raw}"
}

cluster_figures_dir_for_root() {
  local root_dir="$1"
  local run_id="${2:-${RUN_ID:-$(date +%Y-%m-%d)}}"
  printf '%s\n' "${CLUSTER_FIGURES_DIR:-$(cluster_run_dir_for_root "${root_dir}" "${run_id}")/figures}"
}

cluster_reports_dir_for_root() {
  local root_dir="$1"
  local run_id="${2:-${RUN_ID:-$(date +%Y-%m-%d)}}"
  printf '%s\n' "${CLUSTER_REPORTS_DIR:-$(cluster_run_dir_for_root "${root_dir}" "${run_id}")/reports}"
}

resolve_cluster_artifact_dirs() {
  local root_dir="$1"
  local run_id="${2:-${RUN_ID:-$(date +%Y-%m-%d)}}"
  CLUSTER_RUN_DIR_EFFECTIVE="${CLUSTER_RUN_DIR:-$(cluster_run_dir_for_root "${root_dir}" "${run_id}")}"
  CLUSTER_STRUCTURED_DIR_EFFECTIVE="${CLUSTER_RESULTS_STRUCTURED_DIR:-${CLUSTER_RUN_DIR_EFFECTIVE}/structured}"
  CLUSTER_RAW_DIR_EFFECTIVE="${CLUSTER_RESULTS_RAW_DIR:-${CLUSTER_RUN_DIR_EFFECTIVE}/raw}"
  CLUSTER_FIGURES_DIR_EFFECTIVE="${CLUSTER_FIGURES_DIR:-${CLUSTER_RUN_DIR_EFFECTIVE}/figures}"
  CLUSTER_REPORTS_DIR_EFFECTIVE="${CLUSTER_REPORTS_DIR:-${CLUSTER_RUN_DIR_EFFECTIVE}/reports}"
  export CLUSTER_RUN_DIR="${CLUSTER_RUN_DIR_EFFECTIVE}"
  export CLUSTER_RESULTS_STRUCTURED_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
  export CLUSTER_RESULTS_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}"
  export CLUSTER_FIGURES_DIR="${CLUSTER_FIGURES_DIR_EFFECTIVE}"
  export CLUSTER_REPORTS_DIR="${CLUSTER_REPORTS_DIR_EFFECTIVE}"
}

cluster_artifact_env_prefix_for_root() {
  local root_dir="$1"
  local run_id="${2:-${RUN_ID:-$(date +%Y-%m-%d)}}"
  local run_dir structured_dir raw_dir figures_dir reports_dir
  run_dir="$(cluster_run_dir_for_root "${root_dir}" "${run_id}")"
  structured_dir="${run_dir}/structured"
  raw_dir="${run_dir}/raw"
  figures_dir="${run_dir}/figures"
  reports_dir="${run_dir}/reports"
  printf 'CLUSTER_RUN_DIR=%q CLUSTER_RESULTS_STRUCTURED_DIR=%q CLUSTER_RESULTS_RAW_DIR=%q CLUSTER_FIGURES_DIR=%q CLUSTER_REPORTS_DIR=%q' \
    "${run_dir}" \
    "${structured_dir}" \
    "${raw_dir}" \
    "${figures_dir}" \
    "${reports_dir}"
}
