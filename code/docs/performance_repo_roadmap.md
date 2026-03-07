# Performance Repo Roadmap

## Purpose
This roadmap turns the repository into a performance-first system with a short chain from code change to believable speedup claim.

The target state is:
- canonical benchmarks are clearly defined and run the same way every time
- every material speedup has artifacts, verification, and profiler evidence
- regressions are visible historically instead of being discovered ad hoc
- anti-patterns are blocked before they become “wins”
- chapter and lab docs show measured results, not just code listings

## Success Criteria
| Goal | Definition of done |
| --- | --- |
| Canonical suite | One tier-1 benchmark suite exists, is versioned, and is runnable from the CLI and CI with stable artifact paths. |
| Historical evidence | Nightly or release runs write trend and regression artifacts that can be diffed and rendered. |
| Anti-pattern enforcement | The linter and harness fail on the main benchmark-distorting patterns, not just sync misuse. |
| Shared scaffolding | The highest-traffic benchmark families use shared bases instead of chapter-local boilerplate. |
| Evidence-first docs | Top chapter and lab READMEs show measured baseline vs optimized deltas plus profiler-backed explanations. |

## Order of Execution
1. Define and codify a tier-1 canonical benchmark suite.
2. Add artifacted historical trend tracking and regression summaries.
3. Add more performance anti-pattern checks to the linter and harness.
4. Collapse duplicated benchmark scaffolding into shared bases.
5. Rewrite the top chapter and lab READMEs to be evidence-first and profiler-backed.

## Phase 1: Tier-1 Canonical Benchmark Suite
### Objective
Create one small, stable, high-signal benchmark set that becomes the repo’s source of truth for performance claims.

### Scope
- include representative kernels and end-to-end paths
- cover at least GEMM, attention, decode/inference, communication, and one system-level workload
- keep runtime short enough for nightly execution and release gating

### Files to Add
- `core/benchmark/suites/tier1.py`
- `core/benchmark/suites/__init__.py`
- `configs/benchmark_suites/tier1.yaml`
- `core/scripts/benchmarks/run_tier1_suite.py`
- `tests/test_tier1_suite.py`

### Existing Files to Update
- [core/harness/run_benchmarks.py](/home/cfregly/ai-performance-engineering/code/core/harness/run_benchmarks.py)
- [cli/aisp.py](/home/cfregly/ai-performance-engineering/code/cli/aisp.py)
- [core/benchmark/run_manifest.py](/home/cfregly/ai-performance-engineering/code/core/benchmark/run_manifest.py)
- [README.md](/home/cfregly/ai-performance-engineering/code/README.md)

### Proposed Tier-1 Targets
| Category | Candidate targets | Why |
| --- | --- | --- |
| GEMM | `labs/block_scaling:block_scaling`, `labs/blackwell_matmul:*` | Kernel-level math path with clear TFLOP/s story. |
| Attention | `labs/flashattention4:*`, `labs/flashinfer_attention:*` | High-value inference primitive with multiple implementations. |
| Decode | `labs/persistent_decode:*`, `labs/speculative_decode:*` | Real serving-path latency and throughput story. |
| KV/Memory | `labs/kv_optimization:*`, `labs/kv_cache_compression:*` | Memory and cache tradeoff coverage. |
| Communication | selected `ch04` / distributed targets | Multi-GPU performance sanity. |
| End-to-end | one `real_world_models` or `fullstack_cluster` target | Prevent overfitting to microbenchmarks only. |

### CI Changes
- add `.github/workflows/tier1-nightly.yml`
- add a release-only tier-1 job to `.github/workflows/benchmark-validation.yml` or a sibling release workflow
- require tier-1 suite manifest output for release tags

### Acceptance Criteria
- `python -m cli.aisp bench run-tier1` exists
- suite definition is versioned in `configs/benchmark_suites/tier1.yaml`
- every run writes a machine-readable suite summary plus per-target artifacts
- README documents how to run the suite and where artifacts land

## Phase 2: Historical Trends and Regression Summaries
### Objective
Turn benchmark runs into a time series so regressions and improvements are visible.

### Scope
- persist canonical summary artifacts
- compare latest run against previous canonical and previous release
- render a compact markdown regression summary for CI and release notes

### Files to Add
- `core/analysis/trends.py`
- `core/analysis/regressions.py`
- `core/analysis/history_index.py`
- `core/scripts/benchmarks/update_history_index.py`
- `core/scripts/benchmarks/render_regression_summary.py`
- `tests/test_benchmark_trends.py`
- `tests/test_benchmark_regression_summary.py`

### Existing Files to Update
- [core/benchmark/run_manifest.py](/home/cfregly/ai-performance-engineering/code/core/benchmark/run_manifest.py)
- [core/analysis/reporting/generator.py](/home/cfregly/ai-performance-engineering/code/core/analysis/reporting/generator.py)
- [core/analysis/report_generator.py](/home/cfregly/ai-performance-engineering/code/core/analysis/report_generator.py)
- [core/report_export.py](/home/cfregly/ai-performance-engineering/code/core/report_export.py)

### Artifact Contract
Write canonical history under:
- `artifacts/history/tier1/index.json`
- `artifacts/history/tier1/<run_id>/summary.json`
- `artifacts/history/tier1/<run_id>/regression_summary.md`
- `artifacts/history/tier1/<run_id>/trend_snapshot.json`

### CI Changes
- nightly workflow publishes latest trend snapshot as an artifact
- pull-request workflow renders “latest vs baseline” markdown into the job summary when tier-1 targets are affected
- release workflow stores the release summary as a durable artifact

### Acceptance Criteria
- every canonical run can be compared to the last canonical run with no manual path hunting
- regressions are categorized by latency, throughput, memory, verification status, and profiler availability
- PR and release summaries show top regressions and top improvements automatically

## Phase 3: Performance Anti-Pattern Checks
### Objective
Catch benchmark-distorting behavior before it gets merged.

### Scope
Extend the current sync-focused enforcement into broader measurement hygiene checks.

### Files to Add
- `tests/test_benchmark_antipatterns.py`
- `core/benchmark/antipatterns.py`

### Existing Files to Update
- [core/benchmark/contract.py](/home/cfregly/ai-performance-engineering/code/core/benchmark/contract.py)
- [core/harness/validity_checks.py](/home/cfregly/ai-performance-engineering/code/core/harness/validity_checks.py)
- [core/harness/benchmark_harness.py](/home/cfregly/ai-performance-engineering/code/core/harness/benchmark_harness.py)
- [core/scripts/linting/check_benchmarks.py](/home/cfregly/ai-performance-engineering/code/core/scripts/linting/check_benchmarks.py)
- [core/benchmark/defaults.py](/home/cfregly/ai-performance-engineering/code/core/benchmark/defaults.py)

### Checks to Add
| Anti-pattern | Linter | Runtime |
| --- | --- | --- |
| Input regeneration inside `benchmark_fn()` (`randn`, `randint`, `empty`, dataset reload) | Yes | Optional sampling |
| Host-device round trips inside `benchmark_fn()` (`.cpu()`, `.numpy()`, `.item()`, `.tolist()`) | Yes | Yes |
| `torch.compile()` / extension compilation in timed path | Yes | Yes |
| Disk or network I/O in timed path | Yes | Optional hooks |
| Data-dependent shape drift between baseline and optimized | Partial | Yes |
| Profiler calls from timed path | Yes | No |
| Repeated large allocations in timed path when reusable buffers are expected | No | Yes |
| Missing canonical artifact fields for publish-grade runs | No | Yes |

### CI Changes
- extend `.github/workflows/benchmark-validation.yml` to run the expanded anti-pattern test suite
- keep repo-wide contract gate blocking
- add a non-blocking “strict anti-pattern audit” job first, then promote to blocking after burn-in

### Acceptance Criteria
- linter failures are specific and high-signal
- runtime validity checks emit actionable diagnostics instead of generic benchmark failures
- known anti-patterns are represented by tests

## Phase 4: Shared Benchmark Bases
### Objective
Remove duplicated scaffolding in the most active benchmark families so new work is cheaper and less error-prone.

### Scope
Refactor high-duplication families first, not the whole repo at once.

### Files to Add
- `core/benchmark/bases/matmul_pair.py`
- `core/benchmark/bases/attention_pair.py`
- `core/benchmark/bases/decode_pair.py`
- `core/benchmark/bases/distributed_pair.py`
- `tests/test_benchmark_bases.py`

### Existing Families to Migrate First
| Family | Current locations | Shared base target |
| --- | --- | --- |
| Matmul / GEMM | `ch01`, `ch08`, `ch10`, `labs/blackwell_matmul`, `labs/custom_vs_cublas` | `matmul_pair.py` |
| Attention | `ch14`, `ch16`, `labs/flashattention4`, `labs/flashinfer_attention`, `labs/flexattention` | `attention_pair.py` |
| Decode / Serving | `ch15`, `ch17`, `ch18`, `labs/persistent_decode`, `labs/speculative_decode`, `labs/kv_optimization` | `decode_pair.py` |
| Distributed comms | `ch04`, selected cluster/fullstack labs | `distributed_pair.py` |

### Existing Files to Update
- [templates/benchmark_compliant.py](/home/cfregly/ai-performance-engineering/code/templates/benchmark_compliant.py)
- [templates/benchmark_template.py](/home/cfregly/ai-performance-engineering/code/templates/benchmark_template.py)
- [core/benchmark/examples.py](/home/cfregly/ai-performance-engineering/code/core/benchmark/examples.py)

### CI Changes
- add migration smoke tests for each new base
- require new benchmarks to derive from a shared base unless an exception is justified

### Acceptance Criteria
- new benchmark authoring surface is smaller and more uniform
- baseline/optimized pair setup, payload capture, metrics, and verification are centralized
- line count and duplication drop measurably for migrated families

## Phase 5: Evidence-First READMEs
### Objective
Make the top chapters and labs readable as performance case studies instead of code dumps.

### Scope
Rewrite the top-traffic READMEs first.

### Priority Targets
- [README.md](/home/cfregly/ai-performance-engineering/code/README.md)
- [ch10/README.md](/home/cfregly/ai-performance-engineering/code/ch10/README.md)
- [ch14/README.md](/home/cfregly/ai-performance-engineering/code/ch14/README.md)
- [ch18/README.md](/home/cfregly/ai-performance-engineering/code/ch18/README.md)
- [labs/block_scaling/README.md](/home/cfregly/ai-performance-engineering/code/labs/block_scaling/README.md)
- [labs/flashattention4/README.md](/home/cfregly/ai-performance-engineering/code/labs/flashattention4/README.md)
- [labs/persistent_decode/README.md](/home/cfregly/ai-performance-engineering/code/labs/persistent_decode/README.md)
- [labs/real_world_models/README.md](/home/cfregly/ai-performance-engineering/code/labs/real_world_models/README.md)

### README Template
Each target README should include:
1. problem statement
2. baseline and optimized targets
3. exact benchmark command
4. representative measured result table
5. profiler-backed explanation of the win
6. correctness and artifact notes
7. limitations and hardware assumptions

### Supporting Files to Add
- `docs/readme_evidence_template.md`
- `core/scripts/refresh_readmes.py` updates for evidence blocks
- `tests/test_readme_evidence_blocks.py`

### CI Changes
- optional README evidence block validation in CI for the top-priority targets
- fail if a top-priority README loses required evidence sections

### Acceptance Criteria
- top READMEs show commands, measured deltas, and profiler evidence
- contributors can see how to reproduce a claim without reading source code first
- docs stay synchronized with canonical benchmark outputs

## Prioritized Implementation Sequence
### Sprint 1
- add tier-1 suite definition and CLI entrypoint
- lock artifact schema and output paths
- add minimal nightly workflow skeleton

### Sprint 2
- add trend index, regression summary generator, and job summary rendering
- store canonical history artifacts

### Sprint 3
- add the next anti-pattern detectors
- burn in as non-blocking, then flip to blocking

### Sprint 4
- build shared bases for matmul and attention
- migrate the first benchmark families

### Sprint 5
- rewrite top READMEs using canonical run outputs and profiler artifacts
- add README evidence-block validation

## Governance Rules
- no benchmark claim without verification and artifacts
- no new benchmark family without a shared base unless an exception is documented
- no new anti-pattern waiver without a test and an explicit rationale
- no release without a fresh tier-1 canonical run

## Immediate Next Changes
1. Create `configs/benchmark_suites/tier1.yaml`.
2. Add `core/benchmark/suites/tier1.py` and CLI wiring.
3. Add `.github/workflows/tier1-nightly.yml`.
4. Add history index and regression summary generators.
5. Add anti-pattern checks for host round trips and input regeneration.
