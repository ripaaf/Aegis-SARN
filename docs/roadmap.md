# Roadmap and Release Gates

## 1. Roadmap Rules

This roadmap is ordered by dependency and evidence, not by how advanced a technique sounds. Phases 1–4 build and qualify the SARN-Dense control, evaluation laboratory, and efficient-attention foundation. Phases 5–9 construct and test proposed SARN-Hybrid mechanisms; Phases 10–16 add governed system capabilities; Phases 17–18 are integration gates. Framework and model tracks can advance in parallel after their shared contracts exist. A phase may end in a documented rejection; rejected mechanisms are replaced by their controls rather than forced into the integrated architecture.

No calendar dates are assigned until maintainers declare people, hardware, data, and budget. Each phase receives an issue set, owner, estimate, and frozen exit metrics when scheduled.

## Implementation Progress - 2026-07-19

- **Phase 0 specification:** complete for the initial implementation boundary.
- **Phase 1 dense baseline:** implemented and hardened with CPU tests covering configuration, causal isolation, shapes, RoPE, deterministic initialization, backward loss, generated tasks, checkpoint/optimizer resume, smoke overfit, Aegis backends/controller/budgets, structured traces, CLI output, opt-in KV caching, sampled decoding, toy evaluation, CPU benchmarking, and reproducibility manifests.
- **Phase 2 reproducible baseline lab:** implemented for the SARN-Dense control baseline with a local run registry, `list-runs`, baseline Markdown/JSON report generation, `eval-multiseed`, `reproduce-phase2`, dataset/model cards, PowerShell-friendly commands, and focused tests.
- **Phase 3 baseline scaling/quality gates:** implemented as SARN-Dense baseline/evaluation work with `sweep-baseline`, `compare-baselines`, `check-gates`, `eval-tasks`, richer deterministic toy tasks, artifact policy documentation, and common manifest fields.
- **Phase 4 efficient attention foundation:** implemented within SARN-Dense with MHA retained as the default/control, configurable experimental GQA, reduced stored KV-head/cache shapes, matched attention sweeps, comparison reports, attention metadata, checkpoint compatibility, and Phase 4 gates.
- **Acceptance workflow:** the documented CPU path installs, tests, reproduces Phase 2, sweeps tiny SARN-Dense sizes and matched MHA/GQA variants, evaluates task-level metrics, benchmarks generation/cache cost, records registry entries, compares results, checks quality gates, and generates reports while preserving structured metrics and traces.
- **Later phases:** Phase 5 is next/planned. No SARN-Hybrid model path, MoE, graph workspace, resettable working memory, SSM/Mamba, retrieval, tools, VLM, SAM, LAM, advanced safety system, or multimodal implementation has started.

This progress note records the verified Phase 1-4 SARN-Dense contract. Phase 4 adds an experimental GQA option and matched evidence path inside SARN-Dense; it does not implement or validate SARN-Hybrid. All later model and system mechanisms remain proposals until their evidence gates pass.

## Canonical Aegis-X Roadmap

This is the authoritative high-level sequence. Detailed workstreams below preserve implementation and evidence requirements without creating a competing phase numbering system.

| Phase | Target | Evidence-gated outcome |
|---:|---|---|
| 0 | Documentation and research specification | architecture, risks, interfaces, ADRs, and first issue are implementation-ready |
| 1 | SARN-Dense baseline | causal Transformer trains, saves, loads, generates, and supplies a reproducible control |
| 2 | Evaluation harness and benchmarks | structural, systems, safety, and statistical evaluation pipelines establish baseline variance |
| 3 | Baseline scaling and quality gates | SARN-Dense scaling comparisons quantify variance and pass frozen experiment-quality thresholds |
| 4 | Efficient attention foundation | RoPE/GQA path passes correctness and quality-cost comparisons |
| 5 | Latent workspace prototype | bounded slots and token/workspace routing run behind feature flags |
| 6 | Graph message-passing reasoning | graph cycles pass null-edge, equal-compute, and length-extrapolation tests |
| 7 | Resettable working memory | temporary memory passes capacity, conflict, reset, isolation, and poisoning tests |
| 8 | Sparse expert routing | optional experts pass routing stability, total-memory, and active-cost gates or remain disabled |
| 9 | Hybrid sequence engine | optional local/linear attention and SSM blocks pass workload-specific controls or remain disabled |
| 10 | Retrieval and external memory | governed evidence and memory pass provenance, faithfulness, injection, access, and deletion tests |
| 11 | Verifier, critic, and simulator loop | bounded checking improves accepted correctness after false-pass, coverage, and cost accounting |
| 12 | Interpretability with sparse autoencoders | offline feature dictionaries and causal interventions clarify selected model mechanisms |
| 13 | Advanced architecture safety tests | reward hacking, goal shift, oversight shift, tool, memory, and persistence evaluations run |
| 14 | Hardware-adaptive runtime | Nano through Max profiles fit, disclose degradation, and preserve safety invariants |
| 15 | Compression and efficient deployment | quantization, distillation, pruning, speculative decoding, and lottery-ticket studies move a Pareto frontier |
| 16 | Multimodal and action adapters | selected VLM, segmentation, audio, sensor, or action adapter passes its own safety case |
| 17 | SARN-Hybrid Alpha | accepted model mechanisms integrate end to end and survive interaction ablations |
| 18 | Aegis-X experimental system | SARN-Hybrid, Aegis runtime, memory, retrieval, verification, safety, and supported adapters integrate under one traceable contract |

Post-training and preference alignment are cross-cutting activities introduced only after the relevant base capability, evaluation, and safety harnesses exist.

## Detailed Evidence Workstreams

### Workstream A — Authoritative Specification

**Goal:** establish scope, vocabulary, boundaries, evidence rules, and the first executable slice.

**Deliverables:** documentation map; vision and non-goals; Aegis/SARN separation; architecture and interfaces; model research plan; data, training, evaluation, safety, runtime, risk, and repository plans; decision log; open-question register.

**Exit gate:** links and terminology are consistent; every proposed component is marked as baseline, framework, experiment, risk, or future extension; the first coding issue can be implemented without inventing a new architecture.

### Workstream B — Dense Micro-Model and Framework Spine

**Goal:** prove end-to-end correctness on one machine.

**SARN deliverables:** validated configs; reference tokenizer path; RoPE causal attention; decoder block and dense micro-model; tiny generated data; one-batch overfit; training/evaluation loop; atomic checkpoint; greedy and sampled generation.

**Aegis deliverables:** run schemas; configuration resolver; artifact manifest; deterministic fake backend; SARN backend; session state machine; token/time budgets; structured trace; CLI.

**Exit gate:** automated causal-mask, checkpoint-resume, KV-cache parity, config, state-machine, and end-to-end tests pass; a clean clone can reproduce the smoke run from documentation.

### Workstream C — Reproducible Tiny Baseline

**Goal:** create the scientific control for all architecture research.

**Deliverables:** tokenizer study and frozen tokenizer; governed initial datasets; micro and tiny SARN-Dense checkpoints; multi-seed training; language loss and synthetic generalization baselines; model/data/evaluation cards; systems measurements on reference CPU and accelerator where available.

**Exit gate:** baseline runs are reproducible within declared tolerances; held-out evaluation is separated; all artifacts and lineage resolve; no unresolved correctness defect invalidates metrics.

### Workstream D — Evaluation and Experiment Platform

**Goal:** make novel claims cheap to test and hard to fool ourselves about.

**Deliverables:** versioned synthetic generators; structural splits; experiment preregistration schema; run registry; aggregation with confidence intervals; ablation templates; contamination checks; scaling observatory; failure-preserving reports.

**Exit gate:** an intentionally null architecture change produces no false “win”; repeated baseline runs quantify normal variance; raw-to-report lineage is auditable.

## SARN-Hybrid Evidence Workstreams

These phases construct one compatible hybrid model spine. Each phase first proves its mechanism in isolation, then preserves the tensor/configuration contracts needed for later integration. “Construction” states architectural direction; it does not waive an acceptance gate.

### Workstream E — Efficient Attention Foundation

**Goal:** establish and measure the RoPE/GQA attention foundation inside the SARN-Dense control path while reducing decode memory/cost without unmeasured quality loss.

**Experiments:** MHA versus GQA; KV-cache precision; optional local attention; RoPE context behavior and carefully selected scaling methods.

**Exit gate:** at least one profile shows a practically meaningful measured benefit at or above its quality floor, or all tested variants are documented as rejected. Accepted settings remain configurable rather than deleting the reference path.

**Current implementation:** Phase 4 provides correctness tests and matched tiny CPU MHA/GQA sweeps. MHA remains the default. The measured cache reduction is an implementation property; toy quality and local timing results are not sufficient to accept a SARN-Hybrid path.

### Workstream F — Latent Workspace and Graph Core

**Goal:** construct and test the SARN-Hybrid latent workspace, graph message-passing cycles, and gated token-state writeback.

**Deliverables:** workspace-only control; graph variants; soft/hard routing; top-k diagnostics; multi-cycle execution; visualizations labeled as latent states; full equal-compute and null ablations.

**Exit gate:** a variant passes the [benchmark acceptance gates](benchmarks.md), including held-out structural shifts and multi-seed evidence, or the hypothesis is narrowed/rejected. No human-concept claim is accepted without causal interpretability evidence.

### Workstream G — Resettable Working Memory Core

**Goal:** add bounded temporary association/state to the hybrid path without online mutation of base weights.

**Deliverables:** at least one simple key/value control; one neural or fast-weight mechanism; optional Hebbian-style update; strict allocation/reset/isolation; conflict, capacity, poisoning, and leakage evaluations; Aegis lifecycle integration.

**Exit gate:** improvement over both token-context and simple external-memory controls at matched budget, with complete reset and no cross-session leakage. Otherwise prefer the simpler framework memory.

### Workstream H — Hybrid Sequence Engine

**Goal:** evaluate and, if justified, integrate selective SSM blocks beside the accepted attention foundation on relevant long-sequence workloads.

**Deliverables:** selected Mamba-style or equivalent SSM reference; attention-only, SSM-only, and hybrid controls behind one sequence-block contract; kernel and complexity profiling; long-context and ordinary-language regression suite; integrated SARN-Hybrid configuration if the gate passes.

**Exit gate:** a target profile/workload has better quality-cost tradeoff with stable training. “Linear complexity” alone is not an exit criterion.

### Workstream I — Sparse Expert Capacity

**Goal:** test conditional expert capacity as the optional sparse-capacity stage of SARN-Hybrid at a scale and hardware topology where it can matter.

**Deliverables:** dense control; top-k router; capacity handling; load-balancing metrics/loss; router stability and expert analysis; total versus active parameter and memory reports; expert-parallel measurements if distributed.

**Exit gate:** specialization or quality improves at matched active cost without unacceptable collapse, communication, memory, or edge regressions. MoE may remain server-only or be rejected.

### Integrated SARN-Hybrid Gate

After Workstreams E–I produce viable candidates, train the strongest evidence-supported combination end to end. Compare it with SARN-Dense and equal-compute controls; ablate attention configuration, workspace, graph cycles, working memory, optional SSM blocks, and optional experts. The integrated result must report interaction effects, stability, active/total cost, and hardware-specific behavior. Passing isolated module gates does not guarantee the Phase 17 integration gate passes.

## Aegis Knowledge and Assurance Workstreams

### Workstream J — Retrieval, Memory Service, and Provenance

**Goal:** give small models governed access to external knowledge.

**Deliverables:** document ingestion; versioned lexical baseline and optional vector/hybrid retrieval; reranking; typed evidence; citation linkage; session memory; persistent-memory proposal/approval path; access-control, deletion, poisoning, and injection tests.

**Exit gate:** retrieval and end-to-end metrics pass independently; unauthorized content is isolated; citations resolve; deletion propagates under the documented SLA; the no-retrieval fallback is explicit.

### Workstream K — Verification, Repair, and Read-Only Tools

**Goal:** improve time-to-correct-answer while containing effects.

**Deliverables:** verifier registry and assurance classes; deterministic checks for initial code/structured tasks; bounded repair; read-only tool executor; approval and audit flow; adversarial tool tests.

**Exit gate:** check-passing accuracy improves after counting false accepts, false rejects, repair regressions, latency, and coverage; unauthorized calls fail closed; loop budgets cannot be bypassed.

## Training and Alignment Workstream

**Goal:** make accepted checkpoints useful through instructions while preserving measured capability and boundaries.

**Deliverables:** governed SFT set; SFT checkpoint; preference schema and annotator guide; DPO experiment; shortcut audits; calibration and safety suites; structured-call training if needed.

**Exit gate:** practical instruction-following gain with acceptable base-capability, truthfulness, calibration, and safety deltas. Preference tuning is not represented as proof of alignment.

## Deployment and Compression Workstreams

These phases turn accepted SARN-Dense and SARN-Hybrid configurations into measured hardware profiles. They optimize deployment without changing the model/runtime safety boundary.

### Workstream L — Hardware-Adaptive Runtime

**Goal:** choose evaluated configurations for real devices.

**Deliverables:** hardware probes; fit estimator; Nano/Lite/Balanced/Pro profile definitions based on measurements; evaluated quantized artifacts; context/cache planner; capability disclosure; local service; compatibility matrix.

**Exit gate:** each claimed profile passes load, memory pressure, generation, cancellation, quality-floor, and security tests on representative hardware; unsafe fits and missing controls fail closed.

### Workstream M — Distillation, Pruning, and Fast Decoding

**Goal:** improve deployment Pareto frontiers.

**Experiments:** teacher/student distillation; structured and lottery-ticket-inspired pruning; weight/activation/cache quantization; speculative decoding; optional optimizer studies such as Shampoo for future training efficiency.

**Exit gate:** each artifact improves at least one declared frontier (quality, latency, memory, energy, storage) without crossing required floors, and carries its own evaluation card.

## Interpretability and Expansion Workstreams

### Workstream N — Interpretability Program

**Goal:** causally investigate representations and failures in accepted models.

**Deliverables:** activation capture; probes and interventions; SAE artifacts; router/workspace/memory analysis; superposition studies; checkpoint studies around grokking if observed; privacy review.

**Exit gate:** claims have correlation and intervention evidence with stated limits. No interpretability output is promoted directly to a safety authority.

### Workstream O — Multimodal and Constrained Action Extensions

**Goal:** add one modality or action domain at a time without weakening the core boundary.

**Candidate sequence:** vision input adapter; dedicated segmentation tool; speech; structured action planner; reversible tools; robotics only after independent physical safety design.

**Exit gate:** each extension has its own data/evaluation card, permission model, failure containment, resource profile, and regression suite. Acronym coverage is never an exit criterion.

## Release Milestones

- **0.1 Docs and ADR complete:** Phase 0 exits with coherent specifications and an executable first issue.
- **0.2 Dense baseline running:** SARN-Dense supplies a reproducible control checkpoint and training report.
- **0.3 Benchmark harness complete:** capability, systems, safety, and statistical controls are operational.
- **0.4 First SARN-Hybrid module:** one mechanism passes its isolated gate behind a feature flag.
- **0.5 Latent workspace + graph prototype:** integrated workspace and message passing pass initial null/equal-compute controls.
- **0.6 Working-memory prototype:** bounded memory passes initial reset, conflict, and isolation gates.
- **0.7 Hardware-adaptive runtime:** multiple measured profiles select artifacts and budgets with disclosed degradation.
- **1.0 Aegis-X Alpha:** the accepted SARN-Hybrid path and governed Aegis services operate as one experimental, traceable system.

`1.0 Aegis-X Alpha` is a research-integration milestone, not a claim of production maturity, perfection, or universal capability.

## Immediate Backlog

1. choose Python/PyTorch and supported CPU/CUDA versions through an ADR;
2. define configuration and run-manifest schemas;
3. implement RoPE attention and causal-mask tests;
4. implement the micro decoder model and generated-copy dataset;
5. implement one-batch overfit and checkpoint-resume tests;
6. implement Aegis fake backend, session controller, and CLI;
7. publish the first reproducible smoke report.
