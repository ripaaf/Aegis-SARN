# Aegis-SARN

Aegis-SARN is a documentation-first research program for a **new hardware-adaptive hybrid cognitive architecture and runtime**. Its intended destination is SARN-Hybrid: a learned architecture that combines efficient sequence modeling, sparse conditional capacity, a latent graph workspace, resettable working memory, and explicit integration points for an evidence-governed runtime.

The program has three named parts:

1. **Aegis Framework** — the hardware-aware runtime and control plane for model execution, retrieval, persistent memory, tools, verification, safety policy, observability, evaluation, and deployment.
2. **SARN-Dense** — the reproducible decoder-only Transformer baseline and scientific control used to validate the stack and measure every architectural claim.
3. **SARN-Hybrid** — the target experimental model architecture assembled through evidence-gated stages, with every mechanism independently configurable, removable, and measurable.

The ambition is architectural, not rhetorical. The project does not claim that SARN-Hybrid already works, that one small model can provide frontier intelligence on every device, or that compression has no loss. Its measurable objective is **the best verified capability attainable under explicit compute, memory, energy, and latency budgets**. Different hardware tiers may use different checkpoints and features while preserving one framework contract and one set of safety invariants.

## Status

**Phase 0 — specification.** There is currently no model, training pipeline, or runtime implementation in this repository. Phase 1 builds SARN-Dense and the Aegis spine; later phases construct SARN-Hybrid. The documentation defines what will be built, in what order, and what evidence is required before an experimental mechanism becomes a default part of the hybrid architecture.

## Start Here

1. [Documentation map](docs/index.md)
2. [Vision, scope, and non-goals](docs/vision.md)
3. [System architecture](docs/architecture.md)
4. [Aegis Framework specification](docs/framework.md)
5. [SARN model specification](docs/model.md)
6. [Roadmap and release gates](docs/roadmap.md)
7. [Repository and package layout](docs/repository-layout.md)

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

The canonical repository and project name is **Aegis-SARN**. **SARN** means **Sparse Adaptive Reasoning Network** and **Aegis** means **Adaptive Graph-Expert Intelligence System**. “Aegis-SERN” and “Aegis-X,” used in earlier discussion, are treated as historical working names unless a future architecture decision record changes this.

## License

See [LICENSE](LICENSE). Dataset, model-weight, and third-party-component licenses must be tracked separately; this repository license does not automatically cover them.
