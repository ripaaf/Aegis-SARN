# Aegis-SARN

Aegis-SARN is a documentation-first research and engineering project with two connected deliverables:

1. **Aegis Framework** — a hardware-aware runtime for model execution, memory, retrieval, tools, verification, safety policy, observability, evaluation, and deployment.
2. **SARN model family** — a reproducible language-model baseline and a sequence of measured experiments in sparse routing, explicit working memory, latent graph computation, efficient sequence layers, and interpretability.

The project is not a claim that one small model can provide frontier intelligence on every device without loss. Its measurable objective is **the best verified capability attainable under an explicit compute, memory, energy, and latency budget**. Different hardware tiers may use different checkpoints and features while preserving one framework contract.

## Status

**Phase 0 — specification.** There is currently no model, training pipeline, or runtime implementation in this repository. The documentation defines what will be built, in what order, and what evidence is required before an experimental idea becomes part of the architecture.

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
