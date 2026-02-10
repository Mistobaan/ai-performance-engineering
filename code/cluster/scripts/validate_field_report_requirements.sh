#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT="${ROOT_DIR}/cluster/field-report.md"
NOTES="${ROOT_DIR}/cluster/field-report-notes.md"
CANONICAL_RUN_ID=""

usage() {
  cat <<'USAGE'
Usage: cluster/scripts/validate_field_report_requirements.sh [options]

Options:
  --report <path>          Path to field-report.md (default: cluster/field-report.md)
  --notes <path>           Path to field-report-notes.md (default: cluster/field-report-notes.md)
  --canonical-run-id <id>  Expected canonical run id (optional)
  -h, --help               Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      REPORT="$2"
      shift 2
      ;;
    --notes)
      NOTES="$2"
      shift 2
      ;;
    --canonical-run-id)
      CANONICAL_RUN_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

# Resolve relative paths against repo root.
if [[ "$REPORT" != /* ]]; then
  REPORT="${ROOT_DIR}/${REPORT}"
fi
if [[ "$NOTES" != /* ]]; then
  NOTES="${ROOT_DIR}/${NOTES}"
fi

failures=0
warns=0

pass() {
  echo "PASS: $1"
}

warn() {
  echo "WARN: $1"
  warns=$((warns + 1))
}

fail() {
  echo "FAIL: $1"
  failures=$((failures + 1))
}

require_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    pass "file exists: ${file}"
  else
    fail "missing file: ${file}"
  fi
}

has_header() {
  local file="$1"
  local header="$2"
  rg -n --fixed-strings "${header}" "${file}" >/dev/null 2>&1
}

section_body() {
  local file="$1"
  local header="$2"
  awk -v header="$header" '
    $0 == header {in_section=1; next}
    in_section && /^## / {exit}
    in_section {print}
  ' "$file"
}

section_has_table() {
  local file="$1"
  local header="$2"
  section_body "$file" "$header" | rg -q '^\| .*\|'
}

section_has_visual() {
  local file="$1"
  local header="$2"
  section_body "$file" "$header" | rg -q '<img '
}

require_contains() {
  local file="$1"
  local needle="$2"
  local desc="$3"
  if rg -F -- "$needle" "$file" >/dev/null 2>&1; then
    pass "$desc"
  else
    fail "$desc"
  fi
}

forbid_contains() {
  local file="$1"
  local needle="$2"
  local desc="$3"
  if rg -F -- "$needle" "$file" >/dev/null 2>&1; then
    fail "$desc"
  else
    pass "$desc"
  fi
}

require_header() {
  local header="$1"
  if has_header "$REPORT" "$header"; then
    pass "header present: ${header}"
  else
    fail "missing header: ${header}"
  fi
}

check_table_forward_section() {
  local header="$1"
  if ! has_header "$REPORT" "$header"; then
    fail "table-forward check cannot run; missing header: ${header}"
    return
  fi
  if section_has_table "$REPORT" "$header"; then
    pass "table-forward content present: ${header}"
  else
    fail "section is not table-forward: ${header}"
  fi
}

require_file "$REPORT"
require_file "$NOTES"

if [[ $failures -gt 0 ]]; then
  echo "Validation aborted due to missing input files."
  exit 1
fi

required_headers=(
  "## Table of Contents"
  "## TL;DR"
  "## Scope + Canonical Artifacts"
  "## Cluster Story (First Contact)"
  "## Weird / New / Interesting (with Normal Baseline)"
  "## Benchmark A (Networking Story)"
  "## Benchmark B (Inference Story)"
  "## Required Issues (Explicit)"
  "## Root Cause + Fix Mapping"
  "## Report Completeness Delta (vs prior condensed revision)"
  "## Gaps, Risks, and Smell Checks"
  "## Implications for Small AI Teams"
  "## Stakeholder Recommendations (Prioritized)"
  "## Repro Steps"
  "## Reproducibility Package"
  "## Appendix (Coverage vs Case-Study Goals)"
)

for header in "${required_headers[@]}"; do
  require_header "$header"
done

required_subheaders=(
  "### Baseline vs Weird Log"
  "### Deep-Dive Findings"
)

for subheader in "${required_subheaders[@]}"; do
  require_contains "$REPORT" "$subheader" "subheader present: ${subheader}"
done

table_forward_headers=(
  "## TL;DR"
  "## Scope + Canonical Artifacts"
  "## Weird / New / Interesting (with Normal Baseline)"
  "## Benchmark A (Networking Story)"
  "## Benchmark B (Inference Story)"
  "## Required Issues (Explicit)"
  "## Gaps, Risks, and Smell Checks"
  "## Stakeholder Recommendations (Prioritized)"
  "## Reproducibility Package"
  "## Activity Log"
)

for header in "${table_forward_headers[@]}"; do
  check_table_forward_section "$header"
done

visual_sections=(
  "## Benchmark A (Networking Story)"
  "## Benchmark B (Inference Story)"
  "## Weird / New / Interesting (with Normal Baseline)"
)

for header in "${visual_sections[@]}"; do
  if has_header "$REPORT" "$header"; then
    if section_has_visual "$REPORT" "$header"; then
      pass "visual present in section: ${header}"
    else
      fail "missing visual in section: ${header}"
    fi
  fi
done

# Required issue lines must appear verbatim.
required_issues=(
  "Missing node2 fio artifact in canonical package (node2_fio.json absent)."
  "No multinode vLLM artifact in canonical package."
  "No nvbandwidth bundle in canonical package."
  "Health suite had GDR requested, but effective GDR was false due non-CUDA IB local checks."
  "Tail latency knee is severe at high concurrency (throughput up, TTFT/p99 TTFT much worse)."
)
for issue in "${required_issues[@]}"; do
  require_contains "$REPORT" "$issue" "required issue line present: ${issue}"
done

# Core case-study requirement anchors.
require_contains "$REPORT" "## Cluster Story (First Contact)" "cluster story section present"
require_contains "$REPORT" "## Weird / New / Interesting (with Normal Baseline)" "weird/new/interesting section present"
require_contains "$REPORT" "## Benchmark A (Networking Story)" "benchmark A section present"
require_contains "$REPORT" "## Benchmark B (Inference Story)" "benchmark B section present"
require_contains "$REPORT" "## Repro Steps" "repro steps section present"
require_contains "$REPORT" "## Reproducibility Package" "reproducibility package section present"

forbid_contains "$REPORT" "## Normal vs Weird Log" "legacy split header absent: ## Normal vs Weird Log"
forbid_contains "$REPORT" "## Weird / New / Interesting Findings" "legacy split header absent: ## Weird / New / Interesting Findings"

if ! rg -n 'results/structured/.*\.(json|csv|jsonl|txt)' "$REPORT" >/dev/null 2>&1; then
  fail "no structured artifact links found in report"
else
  pass "structured artifact links present"
fi

# Canonical run id sync between report and notes.
report_run_id="$(sed -n 's/.*Canonical run: `\([^`]*\)`.*/\1/p' "$REPORT" | head -n1)"
notes_run_id="$(sed -n 's/.*Canonical run: `\([^`]*\)`.*/\1/p' "$NOTES" | head -n1)"

if [[ -z "$report_run_id" ]]; then
  fail "could not parse canonical run id from report"
else
  pass "parsed report canonical run id: ${report_run_id}"
fi

if [[ -z "$notes_run_id" ]]; then
  fail "could not parse canonical run id from notes"
else
  pass "parsed notes canonical run id: ${notes_run_id}"
fi

if [[ -n "$report_run_id" && -n "$notes_run_id" ]]; then
  if [[ "$report_run_id" == "$notes_run_id" ]]; then
    pass "report and notes canonical run ids match"
  else
    fail "report/notes canonical run id mismatch (${report_run_id} vs ${notes_run_id})"
  fi
fi

if [[ -n "$CANONICAL_RUN_ID" ]]; then
  if [[ "$report_run_id" == "$CANONICAL_RUN_ID" ]]; then
    pass "report canonical run id matches expected (${CANONICAL_RUN_ID})"
  else
    fail "report canonical run id does not match expected (${CANONICAL_RUN_ID})"
  fi
fi

if rg -n 'TODO|TBD|FIXME' "$REPORT" >/dev/null 2>&1; then
  warn "report contains TODO/TBD/FIXME placeholders"
else
  pass "no TODO/TBD/FIXME placeholders found"
fi

echo "---"
echo "Validation summary: failures=${failures}, warnings=${warns}"

if [[ $failures -gt 0 ]]; then
  exit 1
fi
