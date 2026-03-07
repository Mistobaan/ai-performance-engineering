#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  log_activity.sh cmd [--run-id <id>] [--label <label>] [--note <note>] -- <command...>
  log_activity.sh decision [--run-id <id>] --note <note>

Notes:
- Appends a markdown entry to field-report.md (Activity Log table section).
- Writes a raw log to runs/<RUN_ID>/raw/activity/<RUN_ID>_activity.log.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

MODE="$1"
shift

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIELD_REPORT="${ROOT_DIR}/field-report.md"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="$(date +%Y-%m-%d)"
LABEL=""
NOTE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --note)
      NOTE="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}/activity"
RAW_LOG="${RAW_DIR}/${RUN_ID}_activity.log"
mkdir -p "$RAW_DIR"

timestamp="$(date -Iseconds)"

escape_md_cell() {
  local s="$1"
  s="${s//$'\n'/ }"
  s="${s//|/\\|}"
  printf '%s' "$s"
}

append_entry() {
  local date_cell="$1"
  local update_cell="$2"
  local row
  row="| ${date_cell} | $(escape_md_cell "$update_cell") |"

  if ! grep -q "ACTIVITY_LOG_START" "$FIELD_REPORT" || ! grep -q "ACTIVITY_LOG_END" "$FIELD_REPORT"; then
    echo "Activity log markers not found in ${FIELD_REPORT}. Appending at end." >&2
    printf "\n## Activity Log\n| Date | Update |\n| --- | --- |\n%s\n" "$row" >> "$FIELD_REPORT"
    return
  fi

  if ! awk '/ACTIVITY_LOG_START/{f=1;next}/ACTIVITY_LOG_END/{f=0}f' "$FIELD_REPORT" | grep -q '^| Date | Update |'; then
    local tmp_with_header
    tmp_with_header="$(mktemp)"
    awk '
      /ACTIVITY_LOG_START/ {
        print
        print "| Date | Update |"
        print "| --- | --- |"
        next
      }
      { print }
    ' "$FIELD_REPORT" > "$tmp_with_header"
    mv "$tmp_with_header" "$FIELD_REPORT"
  fi

  local tmp
  tmp="$(mktemp)"
  awk -v row="$row" '
    /ACTIVITY_LOG_END/ { print row; print; next }
    { print }
  ' "$FIELD_REPORT" > "$tmp"
  mv "$tmp" "$FIELD_REPORT"
}

case "$MODE" in
  cmd)
    if [[ $# -lt 1 ]]; then
      echo "cmd mode requires a command after --" >&2
      usage >&2
      exit 1
    fi
    cmd_str="$*"
    {
      echo "== ${timestamp} [cmd] label=${LABEL:-n/a} note=${NOTE:-} =="
      echo "\$ ${cmd_str}"
    } >> "$RAW_LOG"
    set +e
    "$@" 2>&1 | tee -a "$RAW_LOG"
    rc=${PIPESTATUS[0]}
    set -e
    echo "rc=${rc}" >> "$RAW_LOG"
    entry_date="${timestamp%%T*}"
    entry="[cmd] label=${LABEL:-n/a} rc=${rc} cmd=\`$cmd_str\`${NOTE:+ note=\"$NOTE\"}"
    append_entry "$entry_date" "$entry"
    exit "$rc"
    ;;
  decision)
    if [[ -z "$NOTE" ]]; then
      echo "decision mode requires --note" >&2
      usage >&2
      exit 1
    fi
    echo "== ${timestamp} [decision] note=${NOTE} ==" >> "$RAW_LOG"
    entry_date="${timestamp%%T*}"
    entry="[decision] note=\"$NOTE\""
    append_entry "$entry_date" "$entry"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 1
    ;;
esac
