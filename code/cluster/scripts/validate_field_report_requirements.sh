#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT="${ROOT_DIR}/cluster/field-report.md"
NOTES="${ROOT_DIR}/cluster/field-report-notes.md"
TEMPLATE="${ROOT_DIR}/cluster/docs/field-report-template.md"
RUNBOOK="${ROOT_DIR}/cluster/docs/advanced-runbook.md"
CANONICAL_RUN_ID=""
ALLOW_RUN_IDS=()

usage() {
  cat <<'USAGE'
Usage: cluster/scripts/validate_field_report_requirements.sh [options]

Options:
  --repo-root <path>        Override repo root (default: auto-detect from script path)
  --report <path>          Path to field-report.md (default: cluster/field-report.md)
  --notes <path>           Path to field-report-notes.md (default: cluster/field-report-notes.md)
  --template <path>        Path to field-report-template.md (default: cluster/docs/field-report-template.md)
  --runbook <path>         Path to advanced-runbook.md (default: cluster/docs/advanced-runbook.md)
  --canonical-run-id <id>  Expected canonical run id (optional)
  --allow-run-id <id>      Additional run id to keep during stale-artifact cleanup checks (repeatable)
  -h, --help               Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      ROOT_DIR="$2"
      REPORT="${ROOT_DIR}/cluster/field-report.md"
      NOTES="${ROOT_DIR}/cluster/field-report-notes.md"
      TEMPLATE="${ROOT_DIR}/cluster/docs/field-report-template.md"
      RUNBOOK="${ROOT_DIR}/cluster/docs/advanced-runbook.md"
      shift 2
      ;;
    --report)
      REPORT="$2"
      shift 2
      ;;
    --notes)
      NOTES="$2"
      shift 2
      ;;
    --template)
      TEMPLATE="$2"
      shift 2
      ;;
    --runbook)
      RUNBOOK="$2"
      shift 2
      ;;
    --canonical-run-id)
      CANONICAL_RUN_ID="$2"
      shift 2
      ;;
    --allow-run-id)
      ALLOW_RUN_IDS+=("$2")
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
if [[ "$TEMPLATE" != /* ]]; then
  TEMPLATE="${ROOT_DIR}/${TEMPLATE}"
fi
if [[ "$RUNBOOK" != /* ]]; then
  RUNBOOK="${ROOT_DIR}/${RUNBOOK}"
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
require_file "$TEMPLATE"
require_file "$RUNBOOK"

report_base="$(basename "$REPORT")"
if [[ "$report_base" == *_environment_report.md ]]; then
  fail "environment-only report is not allowed as canonical field report (${REPORT})"
fi

if [[ $failures -gt 0 ]]; then
  echo "Validation aborted due to missing input files."
  exit 1
fi

required_headers=(
  "## Table of Contents"
  "## TL;DR"
  "## Scope + Canonical Artifacts"
  "## Required Reliability Gates (Canonical Run)"
  "## Operator Friction + Monitoring Expectations (New Checks)"
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
  "## Required Reliability Gates (Canonical Run)"
  "## Operator Friction + Monitoring Expectations (New Checks)"
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
  "## Required Reliability Gates (Canonical Run)"
  "## Operator Friction + Monitoring Expectations (New Checks)"
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
require_contains "$REPORT" "_quick_friction.json" "quick friction artifact links present in report"
require_contains "$REPORT" "_monitoring_expectations.json" "monitoring expectations artifact links present in report"
require_contains "$REPORT" "_operator_checks_dashboard.png" "operator checks dashboard visual referenced in report"
require_contains "$REPORT" "_operator_checks_dashboard.json" "operator checks dashboard summary referenced in report"
require_contains "$NOTES" "_quick_friction.json" "quick friction artifact links present in notes"
require_contains "$NOTES" "_monitoring_expectations.json" "monitoring expectations artifact links present in notes"
require_contains "$TEMPLATE" "## Operator Friction + Monitoring Expectations (Required)" "template includes required operator section"
require_contains "$TEMPLATE" "Quick friction check (required)" "template marks quick friction as required"
require_contains "$TEMPLATE" "Monitoring expectations alignment (required)" "template marks monitoring expectations as required"
require_contains "$TEMPLATE" "operator_checks_dashboard" "template references operator dashboard artifact"
require_contains "$RUNBOOK" "### 4f) Quick Friction Checks (Required For Canonical Runs)" "runbook marks quick friction as required for canonical runs"
require_contains "$RUNBOOK" "### 4g) Monitoring Expectations Snapshot (Required For Canonical Runs)" "runbook marks monitoring expectations as required for canonical runs"
require_contains "$RUNBOOK" "operator_checks_dashboard" "runbook references operator dashboard artifact"

forbid_contains "$REPORT" "## Normal vs Weird Log" "legacy split header absent: ## Normal vs Weird Log"
forbid_contains "$REPORT" "## Weird / New / Interesting Findings" "legacy split header absent: ## Weird / New / Interesting Findings"

if ! rg -n '(published/current/structured|cluster/published/.*/structured|results/structured|cluster/runs/.*/structured|runs/.*/structured)/.*\.(json|csv|jsonl|txt)' "$REPORT" >/dev/null 2>&1; then
  fail "no structured artifact links found in report"
else
  pass "structured artifact links present"
fi

# Validate local markdown links (report+notes+template+runbook) point to existing artifacts.
link_check_out="$(python3 - "$REPORT" "$NOTES" "$TEMPLATE" "$RUNBOOK" <<'PY'
import re
import sys
from pathlib import Path

files = [Path(p) for p in sys.argv[1:]]

def is_placeholder(target: str) -> bool:
    return any(x in target for x in ("<RUN_ID>", "<run_id>", "<hosts>", "<label>", "${", "{RUN_ID}", "<"))

def iter_links(text: str):
    # Markdown: [text](target)
    for m in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
        yield m.group(1).strip()
    # HTML: href="..." and src="..."
    for m in re.finditer(r'(?:href|src)=\"([^\"]+)\"', text):
        yield m.group(1).strip()

missing = []
for path in files:
    base = path.parent
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in iter_links(text):
        if not raw or raw.startswith("#") or "://" in raw or raw.startswith("mailto:"):
            continue
        # Strip anchors and any optional markdown title.
        target = raw.split()[0].split("#", 1)[0]
        if not target or is_placeholder(target):
            continue
        tpath = Path(target)
        resolved = tpath if tpath.is_absolute() else (base / tpath)
        if not resolved.exists():
            missing.append((str(path), target, str(resolved)))

if missing:
    for doc, target, resolved in missing:
        print(f"MISSING_LINK {doc} target={target} resolved={resolved}")
    sys.exit(2)

print("LINKS ok")
PY
)" || link_rc=$?
link_rc="${link_rc:-0}"
if [[ "$link_rc" -eq 0 ]]; then
  pass "local link check passed (${link_check_out})"
else
  while IFS= read -r line; do
    [[ -n "$line" ]] && fail "local link missing: ${line}"
  done <<< "$link_check_out"
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

# Stale artifact hygiene:
# For any run id discovered via *_manifest.json (except canonical and allowlisted),
# fail if artifacts remain while that run id is not linked by report/notes markdown links.
effective_canonical_run_id="$report_run_id"
if [[ -n "$CANONICAL_RUN_ID" ]]; then
  effective_canonical_run_id="$CANONICAL_RUN_ID"
fi

if [[ -z "$effective_canonical_run_id" ]]; then
  fail "cannot run stale-artifact hygiene check without canonical run id"
else
  stale_output="$(python3 - "$ROOT_DIR" "$REPORT" "$NOTES" "$effective_canonical_run_id" "${ALLOW_RUN_IDS[@]}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
report = pathlib.Path(sys.argv[2]).read_text()
notes = pathlib.Path(sys.argv[3]).read_text()
canonical = sys.argv[4]
allow = set(sys.argv[5:])

link_targets = []
for text in (report, notes):
    for m in re.finditer(r'\[[^\]]+\]\(([^)]+)\)', text):
        t = m.group(1).strip()
        if not t or t.startswith('#') or '://' in t or t.startswith('mailto:'):
            continue
        t = t.split()[0]
        t = t.split('#', 1)[0]
        if t:
            link_targets.append(t)

manifest_runs = []
flat_structured = root / "cluster" / "results" / "structured"
if flat_structured.exists():
    for p in flat_structured.glob("*_manifest.json"):
        name = p.name
        if re.match(r"20\d{2}-\d{2}-\d{2}_", name):
            manifest_runs.append(name[:-len("_manifest.json")])

runs_root = root / "cluster" / "runs"
if runs_root.exists():
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        if re.match(r"20\d{2}-\d{2}-\d{2}_", p.name) and (p / "manifest.json").exists():
            manifest_runs.append(p.name)

stale = []
for run_id in sorted(set(manifest_runs)):
    if run_id == canonical or run_id in allow:
        continue
    linked = any(run_id in target for target in link_targets)
    if linked:
        continue

    structured = list((root / "cluster" / "results" / "structured").glob(f"{run_id}*"))
    raw = list((root / "cluster" / "results" / "raw").glob(f"{run_id}*"))
    figures = list((root / "cluster" / "docs" / "figures").glob(f"{run_id}*"))
    run_dir = root / "cluster" / "runs" / run_id
    run_dir_present = run_dir.exists()
    total = len(structured) + len(raw) + len(figures) + (1 if run_dir_present else 0)
    if total > 0:
        stale.append((run_id, len(structured), len(raw), len(figures), int(run_dir_present), total))

if stale:
    for run_id, s_cnt, r_cnt, f_cnt, run_dir_present, total in stale:
        print(
            f"STALE {run_id} structured={s_cnt} raw={r_cnt} figures={f_cnt} "
            f"run_dir={run_dir_present} total={total}"
        )
    sys.exit(2)

print("STALE none")
PY
)" || stale_rc=$?
  stale_rc="${stale_rc:-0}"
  if [[ "$stale_rc" -eq 0 ]]; then
    pass "stale-artifact hygiene check passed (${stale_output})"
  else
    while IFS= read -r line; do
      [[ -n "$line" ]] && fail "stale artifacts detected: ${line}"
    done <<< "$stale_output"
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
