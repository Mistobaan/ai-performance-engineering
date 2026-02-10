# Cluster Case Study Field Notes (Synchronized)

Last updated: 2026-02-10. Canonical run: `2026-02-09_fresh_full_suite_e2e_green_rootfix`.

## Table of Contents
1. [Scope](#scope)
2. [Synchronization Status](#synchronization-status)
3. [Required Issue Ledger](#required-issue-ledger)
4. [Root Cause + Fix Mapping](#root-cause--fix-mapping)
5. [Evidence Matrix](#evidence-matrix)
6. [Coherence + Smell Checks](#coherence--smell-checks)
7. [Repro Entry Point](#repro-entry-point)

## Scope
| Item | Value |
| --- | --- |
| Hosts | `node1,node2` |
| GPUs | `4` per host |
| Excluded hosts | none |
| Canonical run | `2026-02-09_fresh_full_suite_e2e_green_rootfix` |
| Manifest | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_manifest.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_manifest.json) |
| Suite steps | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_suite_steps.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_suite_steps.json) |

## Synchronization Status
| Surface | Status | Notes |
| --- | --- | --- |
| `field-report.md` | synced | All links now point to `2026-02-09_fresh_full_suite_e2e_green_rootfix`. |
| `field-report-notes.md` | synced | Same canonical run, issue ledger, and metrics framing. |
| Canonical suite status | green | `47/47` steps green; required artifact validation passed. |
| Visual coverage | restored | Primary report now references all `32` canonical figures (including per-node deep-dive bundles). |
| Weird/normal depth | restored | Report now uses one merged section (`Weird / New / Interesting (with Normal Baseline)`) with two subsections (`Baseline vs Weird Log`, `Deep-Dive Findings`) so operator reality and weirdness are unified without duplication. |
| Historical incident linkage | normalized | Prior incident-era links in older report revisions were superseded during cleanup; current report retains those lessons with canonical-run-backed evidence only. |
| Case-study requirement mapping | restored | Appendix now maps each explicit case-study prompt requirement to report sections + evidence links. |
| Guardrails against section loss | enabled | Template + AGENTS + validator script now enforce required sections and table-forward high-value coverage. |

## Required Issue Ledger
| Required issue (verbatim) | Status now | Evidence |
| --- | --- | --- |
| Missing node2 fio artifact in canonical package (node2_fio.json absent). | Resolved | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node2_fio.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node2_fio.json) |
| No multinode vLLM artifact in canonical package. | Resolved | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_multinode_serve.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_multinode_serve.json) |
| No nvbandwidth bundle in canonical package. | Resolved | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_nvbandwidth.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_nvbandwidth.json), [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node2_nvbandwidth.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node2_nvbandwidth.json) |
| Health suite had GDR requested, but effective GDR was false due non-CUDA IB local checks. | Resolved in canonical run (`requested=true`, `effective_enabled=true`) | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_health_suite_extended_node1node2_cluster_health_suite_summary.json) |
| Tail latency knee is severe at high concurrency (throughput up, TTFT/p99 TTFT much worse). | Confirmed ongoing risk | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_serve_sweep.csv](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_serve_sweep.csv), [docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_serve_ttft_vs_concurrency.png](docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_serve_ttft_vs_concurrency.png) |

## Root Cause + Fix Mapping
| Issue | Root cause | Fix |
| --- | --- | --- |
| Missing `node2_fio.json` | All-node storage capture was not hard-gated in earlier run packaging. | All-node fio collection + required artifact validation in suite flow. |
| Missing multinode vLLM artifacts | Worker lifecycle teardown could produce non-clean rc in earlier multinode run packaging. | Explicit worker stop semantics + rc normalization only for intentional shutdown after success. |
| Missing nvbandwidth bundle | Host nvbandwidth failed due PTX/user-mode mismatch. | Host compat-lib extraction/injection path; canonical host runtime now succeeds. |
| GDR requested but ineffective | Prereq/mode validation and unsupported mode handling were not strict enough in older path. | Strict prereq + mem-type probe validation before run; unsupported requests fail early. |
| Tail latency knee | Saturation behavior, not missing artifact bug. | Keep explicit risk framing and serving policy guardrails. |

## Evidence Matrix
| Claim | Data evidence | Visual evidence | Verdict |
| --- | --- | --- | --- |
| Canonical suite is fully green. | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_suite_steps.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_suite_steps.json) | [docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_cluster_story_dashboard.png](docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_cluster_story_dashboard.png) | Backed |
| Networking story is strong and stable. | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_health_suite_extended_node1node2_cluster_health_suite_summary.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_health_suite_extended_node1node2_cluster_health_suite_summary.json) | [docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_2nodes_nccl_bw_vs_msg.png](docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_2nodes_nccl_bw_vs_msg.png) | Backed |
| Inference knee remains severe at high concurrency. | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_serve_sweep.csv](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_serve_sweep.csv) | [docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_serve_ttft_vs_concurrency.png](docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_serve_ttft_vs_concurrency.png) | Backed |
| Multinode vLLM evidence is now complete. | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_multinode_serve.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_vllm_multinode_serve.json) | [docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_multinode_vllm_serve_total_tok_s_vs_concurrency.png](docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_multinode_vllm_serve_total_tok_s_vs_concurrency.png) | Backed |
| nvbandwidth bundle exists and is coherent across nodes. | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_nvbandwidth.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_nvbandwidth.json), [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node2_nvbandwidth.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node2_nvbandwidth.json) | [docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_nvbandwidth_sums.png](docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_node1_nvbandwidth_sums.png) | Backed |
| Merged weird/normal section retains historical intent with canonical evidence. | [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_preflight_services.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_preflight_services.json), [results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node_parity_summary.json](results/structured/2026-02-09_fresh_full_suite_e2e_green_rootfix_node_parity_summary.json) | [docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_cluster_story_dashboard.png](docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_cluster_story_dashboard.png), [docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_mamf_straggler.png](docs/figures/2026-02-09_fresh_full_suite_e2e_green_rootfix_mamf_straggler.png) | Backed |

## Coherence + Smell Checks
| Severity | Check | Outcome |
| --- | --- | --- |
| High | Any canonical link still points to `2026-02-09_fresh_full_suite_e2e_fixed` or `...green_strict`? | No |
| High | Any required issue unresolved in canonical package? | No for artifact presence; latency knee remains intentionally open risk |
| Medium | GDR requested/effective mismatch in canonical summary? | No (`true/true`) |
| Medium | Root-cause fix replaced by fallback semantics in canonical path? | No. Canonical nvbandwidth path is host runtime with explicit compat libs |

## Repro Entry Point
Use the canonical command in [field-report.md](field-report.md), section `Repro Steps`.
