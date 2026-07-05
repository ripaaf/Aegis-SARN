# Aegis-SARN

Aegis-SARN is a documentation-first research program to design, implement, and evaluate a **new hardware-adaptive hybrid AI architecture and governed runtime**. It begins with a standard Transformer control for measurement, then follows an evidence-gated path toward SARN-Hybrid and the complete Aegis-X system.

This repository is not merely an optimization wrapper around existing models, and it does not claim that a finished replacement algorithm exists today. It is the engineering and research path for proving or rejecting a hybrid architecture that combines efficient sequence modeling, sparse conditional capacity, latent graph computation, resettable working memory, retrieval, verification, interpretability, alignment, and adaptive deployment.

The program has three named parts:

1. **Aegis Framework** — the hardware-aware runtime and control plane for model execution, retrieval, persistent memory, tools, verification, safety policy, observability, evaluation, and deployment.
2. **SARN-Dense** — the reproducible decoder-only Transformer baseline and scientific control used to validate the stack and measure every architectural claim.
3. **SARN-Hybrid** — the target experimental model architecture assembled through evidence-gated stages, with every mechanism independently configurable, removable, and measurable.
4. **Aegis-X** — the long-term complete system hypothesis: SARN-Hybrid plus the Aegis runtime, external memory, tools, verification, multimodal adapters, safety controls, and hardware profiles.

The ambition is architectural, not rhetorical. The project does not claim that SARN-Hybrid or Aegis-X already works, that one small model can provide frontier intelligence on every device, or that compression has no loss. Its long-term aim is to **maximize verified intelligence per active compute, memory, watt, and latency budget**. Different hardware tiers may use different checkpoints and features while preserving one framework contract and one set of safety invariants.

## Status

**Phase 0 — specification.** There is currently no model, training pipeline, or runtime implementation in this repository. Phase 1 builds SARN-Dense and the Aegis spine; later phases construct SARN-Hybrid and ultimately an Aegis-X experimental system. The documentation defines what will be built, in what order, and what evidence is required before an experimental mechanism becomes a default part of the architecture.

## Start Here

1. [Documentation map](docs/index.md)
2. [Aegis-X Grand Plan](docs/aegis-x-grand-plan.md)
3. [Vision, scope, and non-goals](docs/vision.md)
4. [System architecture](docs/architecture.md)
5. [SARN model specification](docs/model.md)
6. [Roadmap and release gates](docs/roadmap.md)
7. [Aegis Framework specification](docs/framework.md)
8. [Repository and package layout](docs/repository-layout.md)

## Working Rules

- Baselines come before novel modules.
- SARN-Dense is the control group; SARN-Hybrid is the architecture target.
- Every experiment changes one controlled variable where practical.
- “Reasoning,” “memory,” “safety,” and “interpretability” require operational definitions and tests.
- Sparse activation reduces active computation, not necessarily storage or memory traffic.
- Retrieval output, model output, and verified facts are different data types.
- No component may silently write persistent memory, change model weights, or execute a tool.
- A feature is not accepted because it sounds advanced; it is accepted when it beats a relevant baseline under a declared budget.

## Naming

The canonical repository and research-program name is **Aegis-SARN**. **SARN** means **Sparse Adaptive Reasoning Network** and **Aegis** means **Adaptive Graph-Expert Intelligence System**. **Aegis-X** names the long-term complete system architecture; it is not the current implementation or a replacement repository name. “Aegis-SERN” remains a historical misspelling.

## License

See [LICENSE](LICENSE). Dataset, model-weight, and third-party-component licenses must be tracked separately; this repository license does not automatically cover them.
