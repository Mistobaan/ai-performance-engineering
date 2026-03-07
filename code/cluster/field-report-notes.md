# Cluster Case Study Field Notes (Published Canonical Package)

This is the evidence ledger for the published report package. For commands and artifact flow, start with `cluster/README.md`.

Last updated: 2026-03-05. Canonical run: `2026-03-05_localhost_modern_profile_r24_full20b`.

## Table of Contents
1. [Scope](#scope)
2. [Required Reliability Gates](#required-reliability-gates)
3. [Operator Friction + Monitoring](#operator-friction--monitoring)
4. [Required Issue Ledger](#required-issue-ledger)
5. [Root Cause + Fix Mapping](#root-cause--fix-mapping)
6. [Evidence Matrix](#evidence-matrix)
7. [Repro Entry Point](#repro-entry-point)

## Scope
| Item | Value |
| --- | --- |
| Host | `localhost` |
| GPU count | `1` |
| Canonical run | `2026-03-05_localhost_modern_profile_r24_full20b` |
| Manifest | [published/current/manifest.json](published/current/manifest.json) |
| Suite steps | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_suite_steps.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_suite_steps.json) |
| Operator dashboard | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_operator_checks_dashboard.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_operator_checks_dashboard.json) |
| Archived localhost comparison baseline | `2026-03-04_localhost_modern_profile_r22_fastcanon` in `cluster/archive/runs/2026-03-04_localhost_modern_profile_r22_fastcanon/` |
| Historical multi-node reference | `2026-02-10_full_suite_e2e_wire_qf_mon` is reference-only in this workspace; use `cluster/docs/advanced-runbook.md` for the current multi-node bring-up contract. |

## Required Reliability Gates
| Gate | Status | Evidence |
| --- | --- | --- |
| Hang triage readiness | `ok` | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_hang_triage_readiness.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_hang_triage_readiness.json) |
| Torchrun connectivity probe | `ok` | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_torchrun_connectivity_probe.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_torchrun_connectivity_probe.json) |
| NCCL env sensitivity | `ok` | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_nccl_env_sensitivity.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_nccl_env_sensitivity.json) |

## Operator Friction + Monitoring
| Check | Status | Evidence |
| --- | --- | --- |
| quick_friction | see artifact | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_quick_friction.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_quick_friction.json) |
| monitoring_expectations | see artifact | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_monitoring_expectations.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_monitoring_expectations.json) |
| operator checks dashboard (json) | generated | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_operator_checks_dashboard.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_operator_checks_dashboard.json) |
| operator checks dashboard (fig) | generated | [published/current/figures/2026-03-05_localhost_modern_profile_r24_full20b_operator_checks_dashboard.png](published/current/figures/2026-03-05_localhost_modern_profile_r24_full20b_operator_checks_dashboard.png) |

## Required Issue Ledger
| Required issue (verbatim) | Status in localhost package | Evidence |
| --- | --- | --- |
| Missing node2 fio artifact in canonical package (node2_fio.json absent). | Not applicable (single-node scope) | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_fio.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_fio.json) |
| No multinode vLLM artifact in canonical package. | Not applicable (single-node scope) | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_vllm_serve_sweep.csv](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_vllm_serve_sweep.csv) |
| No nvbandwidth bundle in canonical package. | Not applicable unless explicitly enabled in localhost package | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_suite_steps.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_suite_steps.json) |
| Health suite had GDR requested, but effective GDR was false due non-CUDA IB local checks. | Not applicable (`health-suite off`) | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_suite_steps.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_suite_steps.json) |
| Tail latency knee is severe at high concurrency (throughput up, TTFT/p99 TTFT much worse). | Not observed in localhost canary sweep by default | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_vllm_serve_sweep.csv](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_vllm_serve_sweep.csv) |

## Root Cause + Fix Mapping
| Issue | Root cause | Fix | Verification |
| --- | --- | --- | --- |
| preflight false negatives | pipeline-based service probing could be flaky under strict shell behavior | switch to deterministic `systemctl show -p LoadState` checks | clean preflight in canonical suite steps |
| NVLink topology parse fragility | header parsing assumptions were too strict | parser robustness for single-GPU/non-tab layouts | topology summary + figure generated in canonical package |
| quick-friction red-state noise on localhost | optional external tools may be absent by design | expected-failure classification (`expected_failed_checks` vs `unexpected_failed_checks`) | quick_friction artifact includes both lists |

## Evidence Matrix
| Claim | Evidence | Verdict |
| --- | --- | --- |
| Localhost suite is clean | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_suite_steps.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_suite_steps.json) | Backed |
| Operator checks are included | [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_quick_friction.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_quick_friction.json), [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_monitoring_expectations.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_localhost_monitoring_expectations.json), [published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_operator_checks_dashboard.json](published/current/structured/2026-03-05_localhost_modern_profile_r24_full20b_operator_checks_dashboard.json) | Backed |
| Visual package is present | [published/current/figures/2026-03-05_localhost_modern_profile_r24_full20b_operator_checks_dashboard.png](published/current/figures/2026-03-05_localhost_modern_profile_r24_full20b_operator_checks_dashboard.png) | Backed |

## Repro Entry Point
| Step | Command |
| --- | --- |
| Re-run localhost canonical package | `sg docker -c "cluster/scripts/run_cluster_eval_suite.sh --run-id 2026-03-05_localhost_modern_profile_r24_full20b --hosts localhost --labels localhost --ssh-user $(id -un) --primary-label localhost --skip-bootstrap-nodes --disable-fp4 --health-suite off --skip-vllm-multinode --model openai-community/gpt2 --tp 1 --isl 128 --osl 64 --concurrency-range '1 2' --fio-runtime 15 --skip-nvbandwidth"` |
