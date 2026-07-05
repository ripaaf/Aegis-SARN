# Aegis-SARN

Aegis-SARN is a documentation-first research program to design, implement, and evaluate a **new hardware-adaptive hybrid AI architecture and governed runtime**. It begins with a standard Transformer control for measurement, then follows an evidence-gated path toward SARN-Hybrid and the complete Aegis-X system.

This repository is not merely an optimization wrapper around existing models, and it does not claim that a finished replacement algorithm exists today. It is the engineering and research path for proving or rejecting a hybrid architecture that combines efficient sequence modeling, sparse conditional capacity, latent graph computation, resettable working memory, retrieval, verification, interpretability, alignment, and adaptive deployment.

The program has four named parts:

1. **Aegis Framework** — the hardware-aware runtime and control plane for model execution, retrieval, persistent memory, tools, verification, safety policy, observability, evaluation, and deployment.
2. **SARN-Dense** — the reproducible decoder-only Transformer baseline and scientific control used to validate the stack and measure every architectural claim.
3. **SARN-Hybrid** — the target experimental model architecture assembled through evidence-gated stages, with every mechanism independently configurable, removable, and measurable.
4. **Aegis-X** — the long-term complete system hypothesis: SARN-Hybrid plus the Aegis runtime, external memory, tools, verification, multimodal adapters, safety controls, and hardware profiles.

The ambition is architectural, not rhetorical. The project does not claim that SARN-Hybrid or Aegis-X already works, that one small model can provide frontier intelligence on every device, or that compression has no loss. Its long-term aim is to **maximize verified intelligence per active compute, memory, watt, and latency budget**. Different hardware tiers may use different checkpoints and features while preserving one framework contract and one set of safety invariants.

## Status

**Phase 1 — minimum baseline implemented.** The repository now contains a CPU-first SARN-Dense micro model, deterministic generated tasks, smoke training/checkpoint resume, and the minimal Aegis request/backend/trace/CLI spine. SARN-Hybrid, retrieval, tools, working memory, SSM, MoE, and multimodal modules remain unimplemented by design.

## Phase 1 Quickstart

Python 3.11+ and PyTorch 2.2+ are supported. No GPU is required.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on POSIX, then run:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

Run the deterministic CPU smoke trainer. It overfits a generated repeated-pattern batch, resumes the optimizer from its checkpoint, evaluates loss, generates tokens, and writes a JSON manifest:

```bash
aegis-sarn train-smoke --output-dir artifacts/phase1 --device cpu
```

Run the saved micro checkpoint through the Aegis controller and print the structured result and trace:

```bash
aegis-sarn run \
  --checkpoint artifacts/phase1/sarn-dense-smoke.pt \
  --prompt 'aegis sarn ' \
  --max-new-tokens 8 \
  --device cpu
```

The byte tokenizer and toy corpus validate the pipeline; this checkpoint is not a useful natural-language model. Generated artifacts are ignored by Git.

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
