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

**Phase 1 - dense baseline and hardening implemented.** The repository contains a CPU-first SARN-Dense micro model, deterministic generated tasks, smoke training/checkpoint resume, optional KV-cached generation, greedy and sampled decoding, toy evaluation and CPU benchmarking, and the minimal Aegis request/backend/manifest/trace/CLI spine.

**Phase 2 - reproducible baseline evaluation lab implemented.** The repository adds a local run registry, multi-seed toy evaluation, baseline report generation, dataset and model cards, and a CPU reproduction pipeline.

**Phase 3 - baseline scaling, quality gates, and experiment harness implemented.** The repository adds SARN-Dense scaling sweeps, baseline comparison reports, experiment quality gates, richer deterministic toy tasks, task-level evaluation, common manifest fields, and artifact policy documentation.

**Phase 4 - efficient attention foundation implemented within SARN-Dense.** Multi-head attention (MHA) remains the default/control. Experimental grouped-query attention (GQA) is configurable, uses fewer stored KV heads, preserves RoPE and cached decoding, and is evaluated through matched CPU sweep, comparison, manifest, and gate paths.

**Phase 5 - latent workspace prototype implemented within the SARN-Dense research harness.** The bounded slot module is disabled by default and runs only when configured. It supports causal token-to-slot routing, optional gated writeback, transient cached slot state, diagnostics, matched null controls, sweeps, comparison reports, and correctness gates.

**Phase 6 - graph message-passing prototype implemented within the SARN-Dense research harness.** The graph is disabled by default and operates only over enabled Phase 5 latent slots for a fixed small number of cycles. It includes dense, workspace-only, null-edge, frozen-identity, and learned-edge controls; structural toy tasks; diagnostics; sweeps; comparisons; and correctness-oriented gates.

SARN-Dense is still the only complete implemented model path. Phase 6 is not resettable or persistent memory, formal symbolic logic, human-like reasoning, or SARN-Hybrid. Resettable working memory, MoE, SSM/Mamba, retrieval, tools, VLM, SAM, LAM, multimodal modules, and advanced safety systems remain future work.

## Phase 1-6 Quickstart

Python 3.11+ and PyTorch 2.2+ are supported. No GPU is required.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on POSIX, then run:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

### Windows PowerShell

These one-line examples are CPU-compatible and can be pasted directly into PowerShell. Activating `.venv` puts the `aegis-sarn` console command on `PATH`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest

aegis-sarn reproduce-phase2 --output-dir artifacts/phase2-check --device cpu --seed 123

aegis-sarn list-runs --registry artifacts/phase2-check/runs/registry.json

aegis-sarn report-baseline --help
aegis-sarn report-baseline --run-dir artifacts/phase2-check --output-dir artifacts/phase2-check/reports --registry artifacts/phase2-check/runs/registry.json
aegis-sarn eval-multiseed --checkpoint artifacts/phase2-check/train/sarn-dense-smoke.pt --output-dir artifacts/phase2-check/runs --num-seeds 3 --json

.\.venv\Scripts\aegis-sarn.exe sweep-baseline --output-dir artifacts/phase3-sweep --device cpu --seed 123
.\.venv\Scripts\aegis-sarn.exe compare-baselines --input artifacts/phase3-sweep --output-dir artifacts/reports
.\.venv\Scripts\aegis-sarn.exe check-gates --summary artifacts/phase3-sweep/sweep-summary.json
.\.venv\Scripts\aegis-sarn.exe eval-tasks --checkpoint artifacts/phase2-check/train/sarn-dense-smoke.pt --output-dir runs --json

.\.venv\Scripts\aegis-sarn.exe sweep-attention --output-dir artifacts/phase4-attention --device cpu --seed 123
.\.venv\Scripts\aegis-sarn.exe compare-attention --input artifacts/phase4-attention --output-dir artifacts/reports
.\.venv\Scripts\aegis-sarn.exe check-gates --summary artifacts/phase4-attention/attention-sweep-summary.json

.\.venv\Scripts\aegis-sarn.exe sweep-workspace --output-dir artifacts/phase5-workspace --device cpu --seed 123
.\.venv\Scripts\aegis-sarn.exe compare-workspace --input artifacts/phase5-workspace --output-dir artifacts/reports
.\.venv\Scripts\aegis-sarn.exe check-gates --summary artifacts/phase5-workspace/workspace-sweep-summary.json

.\.venv\Scripts\aegis-sarn.exe sweep-graph --output-dir artifacts/phase6-graph --device cpu --seed 123
.\.venv\Scripts\aegis-sarn.exe compare-graph --input artifacts/phase6-graph --output-dir artifacts/reports
.\.venv\Scripts\aegis-sarn.exe check-gates --summary artifacts/phase6-graph/graph-sweep-summary.json
```

`reproduce-phase2` creates the stable checkpoint path `artifacts/phase2-check/train/sarn-dense-smoke.pt` plus train, evaluation, benchmark, registry, and baseline-report artifacts. Generated artifacts remain local and are ignored by Git.

Normal command examples after the environment is activated:

```powershell
aegis-sarn sweep-baseline --output-dir artifacts/phase3-sweep --device cpu --seed 123
aegis-sarn compare-baselines --input artifacts/phase3-sweep --output-dir artifacts/reports
aegis-sarn check-gates --summary artifacts/phase3-sweep/sweep-summary.json
aegis-sarn eval-tasks --checkpoint artifacts/phase2-check/train/sarn-dense-smoke.pt --output-dir runs --json
aegis-sarn sweep-attention --output-dir artifacts/phase4-attention --device cpu --seed 123
aegis-sarn compare-attention --input artifacts/phase4-attention --output-dir artifacts/reports
aegis-sarn check-gates --summary artifacts/phase4-attention/attention-sweep-summary.json
aegis-sarn sweep-workspace --output-dir artifacts/phase5-workspace --device cpu --seed 123
aegis-sarn compare-workspace --input artifacts/phase5-workspace --output-dir artifacts/reports
aegis-sarn check-gates --summary artifacts/phase5-workspace/workspace-sweep-summary.json
aegis-sarn sweep-graph --output-dir artifacts/phase6-graph --device cpu --seed 123
aegis-sarn compare-graph --input artifacts/phase6-graph --output-dir artifacts/reports
aegis-sarn check-gates --summary artifacts/phase6-graph/graph-sweep-summary.json
```

Phase 5 is the first bounded latent-workspace experiment. It compares disabled, no-writeback, two-slot, and four-slot configurations while holding the tiny control model, data, seed, and run path fixed. The workspace is disabled by default. Its slots are transient learned tensor states, not graph nodes, memory records, concepts, or evidence that SARN-Hybrid works.

Phase 6 is the first bounded graph message-passing experiment. It compares the dense and workspace-only controls with null-edge, frozen-identity, and one/two-cycle learned dense graphs. The graph is disabled by default, runs only over transient latent workspace slots, and records mechanical diagnostics and toy/structural metrics. It is not resettable memory, persistent memory, formal symbolic reasoning, proof of human-like reasoning, or evidence that SARN-Hybrid works.

Run the deterministic CPU smoke trainer. It overfits a generated repeated-pattern batch, resumes the optimizer from its checkpoint, evaluates loss, generates tokens, and writes a JSON manifest:

```bash
aegis-sarn train-smoke --output-dir artifacts/phase1 --device cpu
```

Evaluate the toy validation task and write a JSON metrics manifest under `runs/`:

```bash
aegis-sarn eval-toy --checkpoint artifacts/phase1/sarn-dense-smoke.pt --output-dir runs --json
```

Measure CPU generation throughput, dense parameter counts, and approximate model/cache memory:

```bash
aegis-sarn bench --checkpoint artifacts/phase1/sarn-dense-smoke.pt --output-dir runs --use-kv-cache --json
```

Run the saved micro checkpoint through the Aegis controller and print the structured result and trace:

```bash
aegis-sarn run \
  --checkpoint artifacts/phase1/sarn-dense-smoke.pt \
  --prompt 'aegis sarn ' \
  --max-new-tokens 8 \
  --use-kv-cache \
  --device cpu
```

Sampling is explicit and reproducible from a fixed seed:

```bash
aegis-sarn run --checkpoint artifacts/phase1/sarn-dense-smoke.pt --prompt 'aegis sarn ' --strategy sample --temperature 0.8 --top-k 16 --top-p 0.9 --seed 7 --output-dir runs
```

Every train, evaluation, benchmark, and run command records resolved configuration, seed, package version, timestamp, device information, command arguments, metrics, trace events, and the Git commit when available. Phases 2-6 also record runs in local registries, compare tiny dense, attention, workspace, and graph configurations, check experiment gates, and generate Markdown/JSON reports. Graph manifests record enablement, cycle count, edge mode, optional top-k, variant name, parameters, gate mean, message norm, and slot norm. The byte tokenizer and toy corpus validate the pipeline; SARN-Dense is a baseline/control, and these checkpoints are not useful natural-language models. Generated artifacts are ignored by Git.

## Start Here

1. [Documentation map](docs/index.md)
2. [Aegis-X Grand Plan](docs/aegis-x-grand-plan.md)
3. [Vision, scope, and non-goals](docs/vision.md)
4. [System architecture](docs/architecture.md)
5. [SARN model specification](docs/model.md)
6. [Roadmap and release gates](docs/roadmap.md)
7. [Aegis Framework specification](docs/framework.md)
8. [Repository and package layout](docs/repository-layout.md)
9. [Artifact policy](docs/artifacts.md)

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
