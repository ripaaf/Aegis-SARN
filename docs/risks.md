# Risk Register

## 1. Scoring

Likelihood and impact use `low`, `medium`, `high`, or `critical`. Status is `open`, `mitigating`, `accepted`, or `closed`. This document is a planning baseline; owners and review dates are assigned when implementation begins.

## 2. Scientific and Model Risks

| ID | Risk | Likelihood | Impact | Mitigation / evidence trigger |
|---|---|---:|---:|---|
| R-M01 | Latent graph adds complexity but no generalization | high | high | equal-compute controls, null edges, multi-seed structural splits; reject if gate fails |
| R-M02 | Graph nodes are mislabeled as human concepts | high | medium | neutral naming, causal interpretability criteria, no belief claims without calibration |
| R-M03 | Hard top-k routing causes poor gradients or brittle behavior | medium | high | soft-routing control, schedules/straight-through study, route stability metrics |
| R-M04 | Neural/Hebbian memory saturates, overwrites, or leaks | high | high | bounded state, simple external-memory control, conflict/capacity/isolation tests |
| R-M05 | MoE collapses or costs more than dense execution | high | high | delay scale-up, load metrics, total-memory and communication accounting |
| R-M06 | SSM hybrid is slower on available kernels/hardware | medium | medium | measure end-to-end against attention; accept workload-specific outcomes |
| R-M07 | Compression causes hidden capability/safety loss | high | high | per-artifact evaluation, subgroup/task deltas, no bit-width-only claims |
| R-M08 | SAE features are unstable or misleading | high | medium | reconstruction/sparsity reporting, cross-run tests, causal interventions |
| R-M09 | Delayed generalization is mistaken for universal grokking | medium | medium | fixed definitions, full curves, negative runs retained |
| R-M10 | Too many combined modules prevent attribution | high | high | baseline-first sequence, feature flags, ablations, interaction studies |

## 3. Data and Evaluation Risks

| ID | Risk | Likelihood | Impact | Mitigation / evidence trigger |
|---|---|---:|---:|---|
| R-D01 | Training/evaluation contamination inflates claims | high | high | source-group splits, scans, sealed tests, contamination disclosures |
| R-D02 | Dataset licensing prevents model redistribution | medium | critical | manifest and legal/license review before training, release-specific review |
| R-D03 | Private or sensitive data is learned or logged | medium | critical | collection limits, PII audits, access control, minimization, extraction testing |
| R-D04 | Synthetic templates reward shortcuts | high | high | structural holds, template/vocabulary permutations, adversarial generators |
| R-D05 | Preference data rewards verbosity, agreement, or refusal style | high | high | shortcut audits, independent correctness/safety measures, diverse pairs |
| R-D06 | Benchmark selection is optimized post hoc | medium | high | preregistration, primary metrics, sealed final tests, correction/disclosure |
| R-D07 | Aggregate scores hide language/domain harms | high | medium | stratified reports and worst-group analysis |

## 4. Framework, Runtime, and Operations Risks

| ID | Risk | Likelihood | Impact | Mitigation / evidence trigger |
|---|---|---:|---:|---|
| R-F01 | Model bypasses policy through an adapter | medium | critical | model has no effect APIs; contract/security tests; least-privilege executor |
| R-F02 | Hardware planner OOMs or silently degrades required checks | medium | high | conservative fit model, safety margin, explicit capability disclosure, fail closed |
| R-F03 | Backend differences change outputs or sampling semantics | high | medium | contract suite, manifest compatibility, tolerance and reproducibility disclosures |
| R-F04 | Repair loop becomes unbounded or worsens correct answers | medium | high | hard budgets, re-check from scratch, regression and coverage reporting |
| R-F05 | Verifier is treated as an oracle | high | high | assurance classes, deterministic preference, false-pass measurement |
| R-F06 | Artifact or dependency supply-chain compromise | medium | critical | checksums/signatures, safe formats, pinned/scanned dependencies, provenance |
| R-F07 | Trace content leaks user data/secrets | medium | critical | content minimization, redaction, separated retention/access, privacy tests |
| R-F08 | Schema churn blocks all development | medium | medium | narrow v1 contracts, semantic versioning, migrations, contract tests |
| R-F09 | Edge performance claims do not survive thermal/runtime reality | high | medium | on-device sustained benchmarks with environment metadata |
| R-F10 | Experimental feature becomes production default accidentally | medium | high | explicit status flags, config validation, release gate and cards |

## 5. Safety and Security Risks

| ID | Risk | Likelihood | Impact | Mitigation / evidence trigger |
|---|---|---:|---:|---|
| R-S01 | Retrieved prompt injection causes unauthorized action | high | critical | typed trust zones plus external permissions; adversarial tests |
| R-S02 | Persistent memory is poisoned or crosses users | medium | critical | authenticated scope, provenance, approval, isolation and deletion tests |
| R-S03 | Reward proxy produces unsafe shortcut behavior | high | high | hidden tests, multi-objective evaluation, model cannot edit evaluators |
| R-S04 | Team claims to detect deception without evidence | medium | high | prohibit universal-detector claims; observable behavior tests and hard limits |
| R-S05 | Tool arguments exploit paths, network, or commands | high | critical | schema and semantic validation, sandbox, allowlists, approval, no raw text execution |
| R-S06 | Model-generated code is executed unsafely during verification | medium | critical | isolated disposable runner, no secrets/network by default, resource limits |
| R-S07 | Capability expansion adds autonomy without threat review | medium | critical | capability-tier safety case and explicit ADR before enabling |
| R-S08 | Model memorizes dangerous or private training content | medium | high | data controls, release evals, access and distribution decision by risk |

## 6. Project and Governance Risks

| ID | Risk | Likelihood | Impact | Mitigation / evidence trigger |
|---|---|---:|---:|---|
| R-P01 | Scope expands to every modality before a baseline exists | high | critical | text-first roadmap and phase gates; adapters later |
| R-P02 | Compute/data cost exceeds available resources | high | high | micro/tiny scale, explicit budgets, stop rules, external checkpoints where useful |
| R-P03 | Documentation drifts from code | high | high | docs in change definition, link/schema checks, implementation-status labels |
| R-P04 | Key knowledge lives only in chats/notebooks | medium | high | ADRs, experiment registry, reports and artifact manifests in repo |
| R-P05 | “Near perfect” rhetoric drives dishonest claims | medium | high | operational metrics, non-goals, limitations in every release card |
| R-P06 | Single-maintainer bottleneck or abandoned components | medium | high | narrow stable interfaces, ownership, maintenance policy, bus-factor review |
| R-P07 | External names/IP conflict with project naming | low | medium | name/trademark check before public branding or package registry publication |

## 7. Review Procedure

At each phase boundary, review every open high/critical risk, assign an owner and next evidence, add newly discovered risks, and close only with a linked test/report/decision. A risk can be accepted explicitly when mitigation cost exceeds the scoped impact; silent acceptance is not allowed.
