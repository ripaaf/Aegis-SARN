# Open Questions Register

An open question needs a decision owner and evidence path when its phase begins. Questions are closed through the decision log, not by deleting them.

## Immediate Blockers

### Q-001 — Implementation stack

Should Phase 1 standardize on Python and PyTorch, and which minimum CPU/CUDA/Python versions are supportable on the actual development machine?

**Closure evidence:** environment smoke test, dependency/license review, minimal CPU and CUDA tensor test, ADR.

### Q-002 — Baseline tokenizer and corpus

Which legally usable tokenizer training corpus and initial language mixture match the intended English/Indonesian/code scope?

**Closure evidence:** tokenizer comparison, data manifests, license review, bytes/token and downstream smoke results.

### Q-003 — Baseline scale and budget

What micro/tiny dimensions and token budget fit available training hardware while being large enough to expose architecture differences?

**Closure evidence:** memory/throughput probe, short scaling runs, fixed compute budget, ADR.

### Q-004 — Artifact storage

Where will checkpoints, datasets, traces, and reports live locally and for shared work?

**Closure evidence:** storage budget, privacy/access requirements, backup/retention plan, registry proof of concept.

## Model Research

### Q-101 — Workspace insertion and readout

At which layers should latent workspace nodes read token state and write back? **Evidence:** matched sweep with layer, cycle, and no-writeback ablations.

### Q-102 — Graph structure

Are learned static edges, dynamic edges, attention between slots, or no explicit edges the strongest control? **Evidence:** structural-generalization tasks, stability, interpretability, and cost.

### Q-103 — Sparse selection

Can hard top-k routing train stably, or is soft/scheduled sparsity necessary? **Evidence:** gradient/router metrics and multi-seed quality.

### Q-104 — Working-memory mechanism

Does a Hebbian/fast-weight mechanism beat explicit key/value slots and Aegis external memory? **Evidence:** capacity, interference, reset, leakage, quality-cost curves.

### Q-105 — Efficient sequence blocks

Which SSM implementation is reproducible and performant on target hardware, and on which workload does it beat attention? **Evidence:** kernel and end-to-end benchmarks.

### Q-106 — MoE threshold

At what model, batch, and hardware scale does sparse routing justify total memory and communication? **Evidence:** dense/MoE Pareto comparison; no fixed expert count is assumed.

### Q-107 — SAE integration

Should SAEs remain offline analysis tools, or can an SAE feature basis improve workspace routing without harmful reconstruction loss? **Evidence:** separate causal experiment.

### Q-108 — Adaptive computation

Can a halting rule allocate cycles better than a fixed budget without underthinking hard cases? **Evidence:** calibration and quality per total compute across difficulty.

## Framework and Runtime

### Q-201 — First real backend

Should the first backend be native PyTorch only, or should a second local inference engine enter before API stabilization? **Evidence:** contract complexity and target-device measurements.

### Q-202 — Profile thresholds

What measured RAM/VRAM, context, and latency boundaries define Nano through Pro? **Evidence:** representative hardware matrix; avoid arbitrary parameter labels.

### Q-203 — Quantization formats

Which formats and kernels are actually supported on target CPU/GPU devices? **Evidence:** per-backend memory, latency, and quality card.

### Q-204 — Persistent memory product policy

Which facts may be stored, for how long, and with what approval and deletion guarantees? **Evidence:** user model, privacy/legal review, threat tests.

### Q-205 — Default verifiers

Which deterministic checks are cheap and useful enough for default-on operation? **Evidence:** false pass/reject, coverage, repair benefit, and latency.

### Q-206 — Remote service scope

When does a local-only service become network-accessible, and what authentication/multi-user isolation is required? **Evidence:** deployment need and network threat model.

## Data, Evaluation, and Release

### Q-301 — Primary architecture benchmarks

Which two or three structural generalization suites will be primary rather than a broad uncorrected benchmark hunt? **Evidence:** generator validity, shortcut audits, relevance to hypotheses.

### Q-302 — Practical acceptance thresholds

What quality and systems effect sizes justify each module's complexity? **Evidence:** baseline variance, target-device/user value, maintenance estimate.

### Q-303 — External benchmark policy

Which public tasks are appropriate for tiny models and license-compatible reporting? **Evidence:** contamination analysis and evaluation governance.

### Q-304 — Release license and branding

Does `Aegis-SARN` conflict with existing projects/marks, and which weight/data licenses can releases use? **Evidence:** review before public package/model publication.

### Q-305 — Multimodal entry point

Which single future modality has enough user value and data/safety support to enter first? **Evidence:** proposal after the text/runtime gates; no decision needed now.
