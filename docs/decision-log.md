# Architecture Decision Log

This file indexes accepted project-level decisions. Once implementation begins, substantial decisions should use individual ADR files under `reports/decisions/` with context, options, consequences, evidence, status, and supersession links.

## ADR-0001 — SARN-Dense Is the Control; SARN-Hybrid Is the Target

**Status:** accepted, 2026-07-05. Clarifies and extends D-004.

**Decision:** the project starts with SARN-Dense, a decoder-only Transformer, for correctness, reproducibility, and comparison. The long-term model research target is SARN-Hybrid: an integrated architecture combining efficient RoPE/GQA sequence modeling, optional selective SSM blocks, optional sparse experts, a latent graph workspace, resettable working memory, and gated token-state writeback. Aegis supplies retrieval, persistent memory, verification, interpretability infrastructure, policy, and hardware-adaptive deployment around the learned model.

**Reason:** the project needs both a falsifiable control and an unmistakable architecture thesis. Treating every mechanism as an unrelated experiment obscures the intended integration; treating every mechanism as already proven would abandon scientific discipline.

**Consequences:**

- no hybrid mechanism is accepted because of novelty or hype;
- all SARN-Hybrid mechanisms remain behind explicit feature flags until accepted;
- dense and null controls remain available for every hybrid stage;
- architecture claims require matched baselines, ablations, system costs, and held-out evaluation;
- isolated module wins do not establish success until the combined hybrid path passes its integration gate;
- a rejected optional accelerator changes the final configuration without ending the SARN-Hybrid program.

## D-001 — Documentation-First Foundation

**Status:** accepted, 2026-07-05.

**Decision:** define the architecture, boundaries, roadmap, risk register, and evidence rules before growing the implementation.

**Reason:** the project spans model research and a runtime. Shared vocabulary and falsifiable gates reduce uncontrolled scope and chat-only knowledge.

**Consequence:** documents must distinguish current implementation from proposals and change with code.

## D-002 — Two Deliverables: Aegis and SARN

**Status:** accepted, 2026-07-05.

**Decision:** Aegis is the framework/control plane; SARN is the model family/research program.

**Reason:** retrieval, policy, tools, persistent memory, and deployment have different trust and lifecycle requirements from learned tensor computation.

**Consequence:** `sarn` cannot directly perform side effects; Aegis can host non-SARN compatible models.

## D-003 — Canonical Name Is Aegis-SARN

**Status:** accepted pending public name review, 2026-07-05.

**Decision:** use repository spelling **Aegis-SARN**, expanded as **Sparse Adaptive Reasoning Network**. Treat Aegis-SERN and Aegis-X as historical working names.

**Reason:** the current repository already uses SARN and the expansion matches the core hypothesis.

**Consequence:** public branding still requires conflict/trademark review.

## D-004 — Dense Decoder-Only Transformer Is the Baseline

**Status:** accepted, 2026-07-05.

**Decision:** implement SARN-Dense, a small causal decoder-only Transformer, before graph, memory, MoE, SSM, or multimodal additions. It is the control architecture for the SARN-Hybrid construction path defined by ADR-0001.

**Reason:** GPT-style language models are Transformers; a conventional baseline is needed to validate the pipeline and attribute gains.

**Consequence:** advanced modules are isolated variants and can be rejected without blocking the project.

## D-005 — RoPE Baseline; GQA Is Configurable

**Status:** accepted for initial specification, 2026-07-05.

**Decision:** use RoPE for the reference positional method and initially support MHA, then compare GQA through configuration.

**Reason:** this gives a modern, understandable baseline while preserving a clean memory/speed comparison.

**Consequence:** long-context extrapolation and GQA quality are measured, not assumed.

## D-006 — No Uncontrolled Online Weight Updates

**Status:** accepted, 2026-07-05.

**Decision:** serving treats base weights as immutable. Adaptation occurs through bounded temporary state, retrieval, approved persistent memory, adapters, or an offline training/release process.

**Reason:** online weight mutation is difficult to isolate, audit, reverse, and secure.

**Consequence:** Hebbian/fast-weight work is resettable experimental state and never silently becomes a new checkpoint.

## D-007 — Model Output Is an Untrusted Proposal

**Status:** accepted, 2026-07-05.

**Decision:** only Aegis may authorize memory writes, tools, network/filesystem access, or physical actions.

**Reason:** natural-language alignment is not a security boundary.

**Consequence:** all effects use typed calls, external policy, budgets, approval, and audit.

## D-008 — Verification Has Assurance Classes and Hard Bounds

**Status:** accepted, 2026-07-05.

**Decision:** distinguish deterministic, grounded, statistical, and generative checkers. Repair loops have hard iteration/resource limits and re-run checks.

**Reason:** a learned critic can hallucinate and an endless repair loop can waste resources or regress answers.

**Consequence:** results name checks and coverage rather than claiming generic verification.

## D-009 — Hardware Adaptation Means Profiles, Not Equal Capability

**Status:** accepted, 2026-07-05.

**Decision:** preserve framework/API and safety invariants across device profiles while allowing different checkpoints, precision, context, and optional features.

**Reason:** equal frontier capability on arbitrary hardware with no loss violates physical resource constraints.

**Consequence:** every response/release discloses its effective profile and limitations.

## D-010 — Multimodality Enters Through Adapters Later

**Status:** accepted, 2026-07-05.

**Decision:** text and the runtime spine come first. Vision, segmentation, speech, and action are later capability-specific adapters or models.

**Reason:** listing LLM, MLM, VLM, SAM, LAM, and SLM does not define a coherent trainable architecture and would destroy early attribution.

**Consequence:** each modality requires a separate data plan, metric suite, hardware budget, and safety case.

## D-011 — Safety Risks Are Not Fictional Detectors

**Status:** accepted, 2026-07-05.

**Decision:** reward hacking, goal misgeneralization, mesa-optimization, deceptive alignment, and instrumental behavior shape evaluations and hard capability limits. The architecture does not claim a universal internal-goal or deception monitor.

**Reason:** current interpretability and behavioral methods cannot establish such a guarantee.

**Consequence:** observable evaluations are paired with controls effective even when model internals are misunderstood.

## D-012 — Experiment Acceptance Is Predeclared and Cost-Aware

**Status:** accepted, 2026-07-05.

**Decision:** architecture modules require matched baselines, held-out metrics, multiple seeds where relevant, ablations, systems cost, robustness, and a decision entry.

**Reason:** adding parameters or compute can masquerade as architectural innovation.

**Consequence:** negative results complete research milestones and remain documented.

## D-013 — SAEs Begin as Offline Interpretability Tools

**Status:** accepted, 2026-07-05.

**Decision:** sparse autoencoders are initially trained on frozen activations outside the inference path.

**Reason:** SAE features are approximate learned decompositions, not guaranteed human concepts or a free reasoning interface.

**Consequence:** using SAE features inside SARN requires a separate causal architecture experiment.

## D-014 — Persistent Memory Is Append-Oriented and User-Governed

**Status:** accepted, 2026-07-05.

**Decision:** persistent facts retain provenance, sensitivity, owner, authorization, expiry, and supersession lineage, with inspection/export/deletion support.

**Reason:** silent opaque memory creates privacy, poisoning, and correction failures.

**Consequence:** a model may propose but never commit a persistent write.

## Pending ADRs

- implementation stack and supported environment (Q-001);
- tokenizer/data scope (Q-002);
- micro/tiny baseline sizes and compute budget (Q-003);
- artifact store (Q-004);
- first real inference backend and public package naming.
