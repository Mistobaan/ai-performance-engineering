# Cluster Field Report Template

Last updated: YYYY-MM-DD

## Table of Contents
1. [TL;DR](#tldr)
2. [Scope + Artifacts](#scope--artifacts)
3. [Cluster Story (First Contact)](#cluster-story-first-contact)
4. [Weird / New / Interesting (with Normal Baseline)](#weird--new--interesting-with-normal-baseline)
5. [Capability Demonstration (Causal Debugging Workflow)](#capability-demonstration-causal-debugging-workflow)
6. [Benchmark A (Networking Story): NCCL `all_reduce_perf`](#benchmark-a-networking-story-nccl-all_reduce_perf)
7. [Benchmark B (Inference Story): vLLM Online Serving](#benchmark-b-inference-story-vllm-online-serving)
8. [Supporting: nvbandwidth Bundle](#supporting-nvbandwidth-bundle)
9. [Supporting: Compute Sanity (BF16 GEMM)](#supporting-compute-sanity-bf16-gemm)
10. [Supporting: Storage (fio)](#supporting-storage-fio)
11. [Supporting: Health / GDR / NUMA / Train Step / Checkpoint](#supporting-health--gdr--numa--train-step--checkpoint)
12. [Required Issues (Explicit)](#required-issues-explicit)
13. [Root Cause + Fix Mapping](#root-cause--fix-mapping)
14. [Report Completeness Delta](#report-completeness-delta)
15. [Gaps, Risks, and Smell Checks](#gaps-risks-and-smell-checks)
16. [Canonical Cleanup + Artifact Hygiene](#canonical-cleanup--artifact-hygiene)
17. [Implications For Small AI Teams](#implications-for-small-ai-teams)
18. [FP4 Extension Outcomes](#fp4-extension-outcomes)
19. [Reproducibility Package](#reproducibility-package)
20. [Repository Handoff (GitHub)](#repository-handoff-github)
21. [Case Study Requirement Mapping](#case-study-requirement-mapping)
22. [Repro Steps](#repro-steps)
23. [Appendix](#appendix)
24. [Activity Log](#activity-log)

| Rule | Requirement |
| --- | --- |
| Artifact paths | Prefer `results/structured/` and `docs/figures/`; link `results/raw/` only when raw logs are required to prove a root-cause claim. |
| Benchmark validity | GPU benchmark runs are valid only if clock locking succeeded; include clock-lock artifacts. |
| Stakeholder handoff | Include repo URL, commit/tag, and collaborator invite/access status. |
| Visual formatting | Show full-size inline images under each subsection; link each image to itself. |
| Evidence formatting | Put `Evidence data:` immediately below visuals, one link per line (`<br/>`), no comma-chained lists. |
| Table preference | Use tables for all high-value sections; narrative is additive and should not replace tables. |
| Chart quality gate | Any curve chart should have at least 3 unique x-values; if fewer, label as `canary/sparse` and state why. |

## Section Retention Gate (CRITICAL)
| Gate | Requirement |
| --- | --- |
| Mandatory sections | Do not remove or collapse required sections. Keep all ToC entries listed in this template. |
| Weird/normal structure | Keep a single merged section `Weird / New / Interesting (with Normal Baseline)` with both subsections: `Baseline vs Weird Log` and `Deep-Dive Findings`. |
| Required issue ledger | Include the `Required Issues (Explicit)` section with the 5 required issue lines verbatim, each with current status + evidence links. |
| Report parity | Include `Report Completeness Delta` section comparing prior report revision vs current report so section/visual loss is visible. |
| Risk transparency | Include `Gaps, Risks, and Smell Checks` even when suite is green. |
| Cleanup disclosure | Include `Canonical Cleanup + Artifact Hygiene` showing what superseded artifacts were removed and what canonical set was preserved. |
| Synchronization | `field-report.md` and `field-report-notes.md` must use the same canonical RUN_ID and issue status framing. |
| Pre-ship section check | Run: `rg -n '^## ' cluster/field-report.md` and verify every mandatory header is present before sign-off. |
| Pre-ship requirements check | Run: `cluster/scripts/validate_field_report_requirements.sh --report cluster/field-report.md --notes cluster/field-report-notes.md` and fail on any missing section/requirement. |

## Case Study Contract (CRITICAL)
| Prompt requirement | Required report coverage |
| --- | --- |
| Tell us a story about this cluster | `Cluster Story (First Contact)` with timeline table + interpretation. |
| What is weird, new, or interesting | One merged section: `Weird / New / Interesting (with Normal Baseline)` with two subsections (`Baseline vs Weird Log`, `Deep-Dive Findings`). |
| Use 1-2 relevant benchmarks for small AI teams | `Benchmark A` and `Benchmark B` are mandatory and must include why they matter to small teams. |
| Repeatable scripts and structured outputs | `Repro Steps` and `Reproducibility Package` must include concrete commands and JSON/CSV/JSONL artifact links. |
| Include visualizations | Each benchmark arc and each high-value supporting section must include at least one linked visual. |
| Provide unique operator insights | `Implications For Small AI Teams` and `Gaps, Risks, and Smell Checks` must include experience-derived insights. |

## TL;DR
| Item | Value |
| --- | --- |
| Cluster summary | <nodes, gpus/node, CPU/RAM, OS> |
| Primary RUN_ID | `<YYYY-MM-DD_provider_story_...>` |
| Benchmark A headline | <metric + value + short meaning> |
| Benchmark B headline | <metric + value + short meaning> |
| Key weird/new finding | <one sentence> |
| Small-team implication | <one sentence> |

## Scope + Artifacts
| Scope item | Value |
| --- | --- |
| Nodes in-scope | <node list> |
| Excluded nodes | <none or explicit list> |
| GPU count per node | <value> |
| Orchestration/runtime notes | <SSH/Slurm/K8s + Docker/Podman/Enroot + egress limits> |

| Discovery artifact | Link |
| --- | --- |
| Manifest | `results/structured/<RUN_ID>_manifest.json` |
| Cluster metadata | `results/structured/<RUN_ID>_<label>_meta.json` |
| NVLink topology | `results/structured/<RUN_ID>_<label>_nvlink_topology.json` |
| Runtime/CVE evidence | `results/structured/<RUN_ID>_<label>_container_runtime.txt` |

## Cluster Story (First Contact)
| UTC time | Milestone |
| --- | --- |
| <hh:mm:ss> | <event> |
| <hh:mm:ss> | <event> |
| <hh:mm:ss> | <event> |

Visualization:

<p><a href="docs/figures/<RUN_ID>_cluster_story_dashboard.png"><img src="docs/figures/<RUN_ID>_cluster_story_dashboard.png" alt="Cluster story dashboard" width="920"/></a></p>

Evidence data:

`results/structured/<RUN_ID>_health_suite_..._summary.json`<br/>
`results/structured/<RUN_ID>_preflight_services.json`<br/>
`results/structured/<RUN_ID>_node_parity_summary.json`

## Weird / New / Interesting (with Normal Baseline)
### Baseline vs Weird Log
| Area | Normal (canonical) | Weird / notable | Why it matters | Evidence |
| --- | --- | --- | --- | --- |
| Launch path | <normal> | <notable> | <impact> | <links> |
| Networking | <normal> | <notable> | <impact> | <links> |
| Services/health gates | <normal> | <notable> | <impact> | <links> |
| Storage/scratch | <normal> | <notable> | <impact> | <links> |
| Serving behavior | <normal> | <notable> | <impact> | <links> |

### Deep-Dive Findings
| Finding | Baseline anchor | Reinforcement insight | Evidence |
| --- | --- | --- | --- |
| 1 | <Baseline vs Weird row reference> | <what is weird/new and why it matters> | <links> |
| 2 | <Baseline vs Weird row reference> | <what is weird/new and why it matters> | <links> |
| 3 | <Baseline vs Weird row reference> | <what is weird/new and why it matters> | <links> |

### Weird/Normal Evidence
Visualization:

<p><a href="docs/figures/<RUN_ID>_cluster_story_dashboard.png"><img src="docs/figures/<RUN_ID>_cluster_story_dashboard.png" alt="Weird/normal baseline dashboard" width="920"/></a></p>
<p><a href="docs/figures/<RUN_ID>_finding1.png"><img src="docs/figures/<RUN_ID>_finding1.png" alt="Weird/normal deep-dive finding 1" width="920"/></a></p>
<p><a href="docs/figures/<RUN_ID>_finding2.png"><img src="docs/figures/<RUN_ID>_finding2.png" alt="Weird/normal deep-dive finding 2" width="920"/></a></p>

Evidence data:

`results/structured/<RUN_ID>_preflight_services.json`<br/>
`results/structured/<RUN_ID>_node_parity_summary.json`<br/>
`results/structured/<RUN_ID>_finding1.json`<br/>
`results/structured/<RUN_ID>_finding1.csv`<br/>
`results/structured/<RUN_ID>_finding2.json`

## Capability Demonstration (Causal Debugging Workflow)
| Phase | Content |
| --- | --- |
| Symptom detection | <what broke + where detected> |
| Hypothesis | <candidate root cause> |
| Isolation | <tests that ruled alternatives out> |
| Mitigation | <what changed> |
| Verification | <before/after metrics + artifacts> |

Visualization:

<p><a href="docs/figures/<RUN_ID>_capability_demo.png"><img src="docs/figures/<RUN_ID>_capability_demo.png" alt="Capability demonstration" width="920"/></a></p>

Evidence data:

`results/structured/<artifact1>.json`<br/>
`results/structured/<artifact2>.json`

## Benchmark A (Networking Story): NCCL `all_reduce_perf`
| Field | Content |
| --- | --- |
| Why | Explain intra-node vs inter-node behavior and scaling. |
| Config | <GPUs used, nodes, message size range, warmup/iters> |
| Repro commands | `scripts/run_nccl_all_reduce.sh ...`<br/>`python3 analysis/plot_nccl.py ...` |
| Interpretation | <2-4 sentences: topology, oversubscription signals, stability> |

Visualization:

<p><a href="docs/figures/<RUN_ID>_nccl_bw_vs_msg.png"><img src="docs/figures/<RUN_ID>_nccl_bw_vs_msg.png" alt="NCCL bandwidth" width="920"/></a></p>
<p><a href="docs/figures/<RUN_ID>_nccl_scaling_efficiency.png"><img src="docs/figures/<RUN_ID>_nccl_scaling_efficiency.png" alt="NCCL scaling" width="920"/></a></p>

Evidence data:

`results/structured/<RUN_ID>_nccl.json`

## Benchmark B (Inference Story): vLLM Online Serving
| Field | Content |
| --- | --- |
| Why | Explain throughput vs concurrency and latency knees. |
| Config | <model, TP, ISL/OSL, sweep values> |
| Repro commands | `scripts/repro/run_vllm_serve_sweep_container.sh ...`<br/>`python3 analysis/plot_vllm_serve_sweep.py ...` |
| Optional multinode path | `scripts/repro/run_vllm_serve_multinode_container.sh ...` |
| Sparse-data policy | If unique concurrency values < 3, explicitly mark as `canary` and do not claim a full curve conclusion. |

Visualization:

<p><a href="docs/figures/<RUN_ID>_<label>_vllm_serve_total_tok_s_vs_concurrency.png"><img src="docs/figures/<RUN_ID>_<label>_vllm_serve_total_tok_s_vs_concurrency.png" alt="vLLM throughput" width="920"/></a></p>
<p><a href="docs/figures/<RUN_ID>_<label>_vllm_serve_ttft_vs_concurrency.png"><img src="docs/figures/<RUN_ID>_<label>_vllm_serve_ttft_vs_concurrency.png" alt="vLLM TTFT" width="920"/></a></p>

Evidence data:

`results/structured/<RUN_ID>_<label>_vllm_serve_sweep.csv`<br/>
`results/structured/<RUN_ID>_<label>_vllm_serve_sweep.jsonl`

## Supporting: nvbandwidth Bundle
| Field | Content |
| --- | --- |
| Why | Add direct host-device and GPU-GPU throughput evidence with strict lock metadata. |
| Repro | `scripts/repro/run_nvbandwidth_bundle.sh --runtime host ...`<br/>`python3 analysis/plot_nvbandwidth_sums.py ...` |

Visualization:

<p><a href="docs/figures/<RUN_ID>_<label>_nvbandwidth_sums.png"><img src="docs/figures/<RUN_ID>_<label>_nvbandwidth_sums.png" alt="nvbandwidth sums" width="920"/></a></p>

Evidence data:

`results/structured/<RUN_ID>_<label>_nvbandwidth.json`<br/>
`results/structured/<RUN_ID>_<label>_nvbandwidth_sums.csv`<br/>
`results/structured/<RUN_ID>_<label>_nvbandwidth_clock_lock.json`

## Supporting: Compute Sanity (BF16 GEMM)
| Field | Content |
| --- | --- |
| Why | Catch per-node/per-GPU throughput deltas quickly. |
| Repro | `scripts/run_gemm_sanity_all_nodes.sh ...` |

Visualization:

<p><a href="docs/figures/<RUN_ID>_gemm_gpu_sanity.png"><img src="docs/figures/<RUN_ID>_gemm_gpu_sanity.png" alt="GEMM sanity" width="920"/></a></p>

Evidence data:

`results/structured/<RUN_ID>_<label>_gemm_gpu_sanity.csv`

## Supporting: Storage (fio)
| Field | Content |
| --- | --- |
| Why | Baseline sequential MB/s + random IOPS and node parity. |
| Repro | `scripts/run_fio_bench.sh ...`<br/>`python3 analysis/plot_fio.py ...` |

Visualization:

<p><a href="docs/figures/<RUN_ID>_<label>_fio.png"><img src="docs/figures/<RUN_ID>_<label>_fio.png" alt="fio summary" width="920"/></a></p>

Evidence data:

`results/structured/<RUN_ID>_<label>_fio.json`

## Supporting: Health / GDR / NUMA / Train Step / Checkpoint
| Area | Repro | Evidence data |
| --- | --- | --- |
| Health suite | `scripts/run_cluster_health_suite.sh ...` | `results/structured/<RUN_ID>_..._cluster_health_suite_summary.json` |
| GDR checks | `scripts/run_cluster_health_suite.sh --gdr ...` | `ib_gdr` section in summary payload |
| C2C/NUMA probes | `scripts/run_c2c_memcpy_bench.sh ...`<br/>`scripts/run_numa_mem_bw_all_nodes.sh ...` | `results/structured/<RUN_ID>_<label>_c2c_memcpy.json`<br/>`results/structured/<RUN_ID>_<label>_numa_mem_bw.json` |
| Train step | `scripts/run_torchrun_transformer_train_step.sh ...` | `results/structured/<RUN_ID>_<label>_torchrun_train_step.json` |
| Checkpoint I/O | `scripts/run_checkpoint_io_all_nodes.sh ...` | `results/structured/<RUN_ID>_<label>_checkpoint_io.json` |

## Required Issues (Explicit)
| Required issue (verbatim) | Current status | Evidence |
| --- | --- | --- |
| Missing node2 fio artifact in canonical package (node2_fio.json absent). | <resolved/open + short note> | <artifact links> |
| No multinode vLLM artifact in canonical package. | <resolved/open + short note> | <artifact links> |
| No nvbandwidth bundle in canonical package. | <resolved/open + short note> | <artifact links> |
| Health suite had GDR requested, but effective GDR was false due non-CUDA IB local checks. | <resolved/open + short note> | <artifact links> |
| Tail latency knee is severe at high concurrency (throughput up, TTFT/p99 TTFT much worse). | <resolved/open + short note> | <artifact links> |

## Root Cause + Fix Mapping
| Issue | Root cause | Fix shipped | Current canonical result |
| --- | --- | --- | --- |
| <issue> | <root cause> | <what changed> | <status/metric> |

## Report Completeness Delta
| Area | Prior revision | Current revision |
| --- | --- | --- |
| Canonical run alignment | <prior> | <current> |
| Benchmark coverage depth | <prior> | <current> |
| Visual coverage | <prior figure count> | <current figure count> |
| Required issue handling | <prior> | <current> |
| Suite status clarity | <prior> | <current> |

## Gaps, Risks, and Smell Checks
| Severity | Finding | Why it matters | Evidence |
| --- | --- | --- | --- |
| High | <finding> | <impact> | <links> |
| Medium | <finding> | <impact> | <links> |
| Low | <finding> | <impact> | <links> |

## Canonical Cleanup + Artifact Hygiene
| Item | Status | Evidence |
| --- | --- | --- |
| Canonical RUN_ID preserved | <yes/no> | <manifest link> |
| Superseded intermediate runs removed | <yes/no + counts> | <cleanup evidence> |
| Report links point only to canonical artifacts | <yes/no> | <link-check command output summary> |

## Implications For Small AI Teams
| Focus area | Practical implication |
| --- | --- |
| Week-1 setup | <guidance> |
| Serving operations | <guidance> |
| Node acceptance | <guidance> |
| Queue discipline | <guidance> |
| Observability | <guidance> |

## FP4 Extension Outcomes
| Area | What you implemented/prototyped | Why it should be upstreamed |
| --- | --- | --- |
| <example> | <script/analysis/doc path> | <impact> |
| <example> | <script/analysis/doc path> | <impact> |

## Reproducibility Package
| Bundle | Artifact links |
| --- | --- |
| Baseline package | `results/structured/<RUN_ID>_manifest.json`<br/>`results/structured/<RUN_ID>_cluster_meta.json`<br/>`results/structured/<RUN_ID>_preflight_services.json`<br/>`results/structured/<RUN_ID>_health_suite_..._summary.json` |
| Multinode vLLM bundle | `results/structured/<RUN_ID>_<label>_vllm_multinode_serve.json`<br/>`results/structured/<RUN_ID>_<label>_vllm_multinode_serve.csv` |
| NVLink topology bundle | `results/structured/<RUN_ID>_node1_nvlink_topology.json`<br/>`results/structured/<RUN_ID>_node2_nvlink_topology.json` |
| nvbandwidth bundle | `results/structured/<RUN_ID>_node1_nvbandwidth.json`<br/>`results/structured/<RUN_ID>_node2_nvbandwidth.json`<br/>`results/structured/<RUN_ID>_node1_nvbandwidth_sums.csv`<br/>`results/structured/<RUN_ID>_node2_nvbandwidth_sums.csv` |
| fio parity bundle | `results/structured/<RUN_ID>_node1_fio.json`<br/>`results/structured/<RUN_ID>_node2_fio.json` |

Visualization:

<p><a href="docs/figures/<RUN_ID>_node1_nvlink_topology.png"><img src="docs/figures/<RUN_ID>_node1_nvlink_topology.png" alt="Node1 NVLink topology" width="920"/></a></p>
<p><a href="docs/figures/<RUN_ID>_node2_nvlink_topology.png"><img src="docs/figures/<RUN_ID>_node2_nvlink_topology.png" alt="Node2 NVLink topology" width="920"/></a></p>
<p><a href="docs/figures/<RUN_ID>_node1_nvbandwidth_sums.png"><img src="docs/figures/<RUN_ID>_node1_nvbandwidth_sums.png" alt="Node1 nvbandwidth sums" width="920"/></a></p>
<p><a href="docs/figures/<RUN_ID>_node2_nvbandwidth_sums.png"><img src="docs/figures/<RUN_ID>_node2_nvbandwidth_sums.png" alt="Node2 nvbandwidth sums" width="920"/></a></p>

Evidence data:

`results/structured/<RUN_ID>_node1_nvlink_topology.json`<br/>
`results/structured/<RUN_ID>_node2_nvlink_topology.json`<br/>
`results/structured/<RUN_ID>_node1_nvbandwidth.json`<br/>
`results/structured/<RUN_ID>_node2_nvbandwidth.json`

## Repository Handoff (GitHub)
| Field | Value |
| --- | --- |
| Repository URL | `<url>` |
| Commit/Tag for review | `<commit or tag>` |
| Collaborator access (`JordanNanos`) status | `<invited|already has access>` |

## Case Study Requirement Mapping
| Case-study requirement | Where addressed in report | Evidence links |
| --- | --- | --- |
| Tell the cluster story | `Cluster Story (First Contact)` | <links> |
| Highlight weird/new/interesting aspects | `Weird / New / Interesting (with Normal Baseline)` (`Baseline vs Weird Log` + `Deep-Dive Findings`) | <links> |
| Use 1-2 relevant benchmarks for small-team AI | `Benchmark A` + `Benchmark B` | <links> |
| Provide reproducible scripts/notes + structured outputs | `Repro Steps` + `Reproducibility Package` | <links> |
| Include visualizations | benchmark and supporting visuals | <links> |
| Provide unique operator insights | `Gaps, Risks, and Smell Checks` + `Implications For Small AI Teams` | <links> |

## Repro Steps
| Profile | Command |
| --- | --- |
| Portable baseline | `scripts/run_cluster_eval_suite.sh --run-id <RUN_ID> --hosts <h1,h2> --labels <l1,l2> --ssh-key <key> --oob-if <iface> --socket-ifname <iface> --nccl-ib-hca <hcas> --health-suite extended --disable-fp4` |
| Full GB200 (FP4 enabled) | `scripts/run_cluster_eval_suite.sh --run-id <RUN_ID> --hosts <h1,h2> --labels <l1,l2> --ssh-key <key> --oob-if <iface> --socket-ifname <iface> --nccl-ib-hca <hcas> --health-suite extended --fp4-runtime host` |
| Multinode vLLM sweep add-on | `--run-vllm-multinode --vllm-multinode-concurrency-range "16 32 64 128" --vllm-multinode-num-prompts <n>` |

## Appendix
| Topic | Notes |
| --- | --- |
| Discovery links | <links> |
| Tuning deltas | <sysctl/MTU/NCCL env changes> |
| Historical incident retention | <why retained and what changed decisions> |

## Activity Log
<!-- ACTIVITY_LOG_START -->
| Date | Update |
| --- | --- |
| YYYY-MM-DD | <entry> |
<!-- ACTIVITY_LOG_END -->
