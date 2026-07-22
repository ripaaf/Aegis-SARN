# Documentation Map

This is the authoritative navigation page. Documents are split by responsibility so framework engineering, model research, and governance can evolve without becoming one untestable master document.

## Project Definition

- [Aegis-X Grand Plan](aegis-x-grand-plan.md) — long-term complete system hypothesis and acceptance gates
- [Vision](vision.md) — mission, success definition, scope, and non-goals
- [Engineering principles](principles.md) — rules used to resolve design tradeoffs
- [Glossary](glossary.md) — canonical vocabulary and overloaded terms
- [Decision log](decision-log.md) — accepted architectural decisions
- [Open questions](open-questions.md) — unresolved questions with closure criteria

## System Design

- [Architecture](architecture.md) — whole-system boundaries and end-to-end flows
- [Aegis Framework](framework.md) — runtime, orchestration, policy, tools, observability, and APIs
- [SARN-Hybrid Target Algorithm](model.md#2-sarn-hybrid-target-algorithm) — integrated model hypothesis and ablatable mechanisms
- [SARN model specification](model.md) — baseline control model, target algorithm, and evidence-gated variants
- [Research matrix](research-matrix.md) — every discussed idea classified by hybrid role, status, and evidence
- [Interfaces](interfaces.md) — shared schemas and module contracts
- [Memory and retrieval](memory.md) — context, working memory, persistence, retrieval, and provenance
- [Runtime and deployment](runtime.md) — hardware profiles, quantization, packaging, and graceful degradation
- [Repository layout](repository-layout.md) — intended source tree and dependency boundaries

## Building and Training

- [Data](data.md) — dataset governance, mixtures, processing, and contamination controls
- [Phase 2-8 toy dataset cards](datasets.md) — generated-fixture provenance, scope, and limitations
- [SARN-Dense model card](model-card-sarn-dense.md) — baseline/control capabilities and limitations
- [Training](training.md) — stages, objectives, checkpoints, optimizer studies, and reproducibility
- [Development workflow](development.md) — implementation standards, testing, experiments, and contribution flow
- [Roadmap](roadmap.md) — ordered milestones, deliverables, and exit gates

## Evidence and Assurance

- [Benchmarks](benchmarks.md) — capability, systems, ablation, robustness, and statistical evaluation
- [Artifact policy](artifacts.md) — generated-output locations, Git policy, cleanup, and Windows write behavior
- [Safety](safety.md) — threat model, capability boundaries, tool controls, and alignment evaluation
- [Interpretability](interpretability.md) — probes, sparse autoencoders, interventions, and limitations
- [Risk register](risks.md) — technical, scientific, operational, legal, and project risks
- [Research references](references.md) — primary sources and how they inform this project

## Document Status Vocabulary

- **Normative**: required behavior for an implementation claiming compatibility.
- **Proposed**: intended design that has not yet been validated.
- **Experimental**: implemented only behind a feature flag or research configuration.
- **Accepted**: passed its stated evidence gate.
- **Rejected**: tested and not adopted; results remain documented.

Phases 1-8 are implemented and verified on CPU through the SARN-Dense research harness. MHA and dense FFNs remain the defaults; GQA, the bounded latent workspace, graph message passing, resettable working memory, and sparse expert routing are experimental and configurable, with all optional modules disabled by default. Phase 8 experts are a local top-k reference path, not distributed/full MoE or proof of specialization. SARN-Dense remains the only complete implemented model path; persistent memory, retrieval, Phase 9+ mechanisms, and SARN-Hybrid remain unimplemented.
