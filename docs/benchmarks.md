# Evaluation and Benchmarks

## 1. Evaluation Contract

Every reported result names the artifact and configuration digest, data and benchmark versions, prompting/evaluation code revision, hardware/backend, precision, context and generation limits, seed(s), sample count, uncertainty, and whether the result was exploratory or held out.

Scores without systems cost are incomplete for this project. Systems numbers without task quality are equally incomplete.

## 2. Breakthrough Metrics

SARN-Hybrid claims require simultaneous model-quality and system-cost evidence. “Intelligence” is never reported as an undefined scalar; each efficiency ratio names the task suite, quality aggregation, denominator, and hardware measurement method.

### Capability-Efficiency Frontiers

- **verified quality per active FLOP:** predeclared composite or task-specific check-passing quality divided by measured/estimated active forward compute, with the aggregation and FLOP method published;
- **verified quality per watt:** check-passing task quality alongside average and peak power plus energy per completed request; wattage alone is not energy efficiency;
- **verified quality per MB of RAM/VRAM:** check-passing task quality against peak resident memory and, separately, artifact storage;
- **time-to-verified-correct-answer:** end-to-end latency through retrieval, generation, checking, repair, and final correct acceptance;
- **active/total capacity frontier:** quality versus active parameters, total resident parameters, and model bytes so sparse systems cannot hide storage cost.

Ratios supplement rather than replace Pareto plots. A model can game a ratio by becoming extremely small and useless, so every profile also has minimum absolute quality and safety floors.

### Hybrid-Architecture Stress Tests

- **long-context recall under memory pressure:** recall as context grows while RAM/VRAM or KV-cache budgets are fixed;
- **structural generalization:** performance on held-out compositions, values, templates, graph sizes, and state transitions;
- **graph reasoning length extrapolation:** accuracy and calibration beyond trained path/rule-chain lengths, with dense equal-compute controls;
- **working-memory conflict handling:** retention, overwrite, contradiction, interference, expiry, and complete-reset curves;
- **sparse expert collapse and routing stability:** utilization, routing entropy, overflow/drop rate, specialization, assignment stability, and worst-expert load through training;
- **hardware degradation curve:** quality, latency, energy, context, and enabled-capability changes across Nano, Lite, Balanced, Pro, and Max rather than one best-device point;
- **integration efficiency:** SARN-Hybrid versus SARN-Dense at matched parameters, active FLOPs, measured latency, and memory, followed by stage-by-stage ablation.

No architecture-breakthrough claim is supported by quality-only or systems-only improvement. The result must state which frontier moved, by how much, on which workload and hardware, and with which regressions.

## 3. Evaluation Layers

### Model Quality

- token-level validation loss and perplexity where tokenizers are comparable;
- exact match or structured accuracy on synthetic tasks;
- language, code, math, and instruction-following suites appropriate to model size;
- calibration, abstention, contradiction handling, and robustness under paraphrase;
- performance by language/domain rather than only one aggregate.

### SARN Research Tasks

- copy and selective-copy length extrapolation;
- associative recall with capacity, distractor, and overwrite curves;
- variable binding and entity-state tracking;
- held-out rule-chain lengths and rule permutations;
- graph reachability and path tasks on held-out graph sizes;
- compositional command planning with unseen combinations;
- temporary fact recall after interference and reset;
- adaptive-cycle quality versus actual compute.

Synthetic generators publish structural splits. Random example splits alone are insufficient.

### Framework Quality

- request and schema conformance;
- cancellation and timeout correctness;
- deterministic budget enforcement;
- backend contract compliance;
- retrieval and citation behavior;
- memory isolation and deletion propagation;
- tool authorization and idempotency;
- trace completeness and redaction;
- failure recovery.

### Safety and Robustness

- direct and indirect prompt injection;
- unauthorized tool and persistence requests;
- data exfiltration and cross-session leakage;
- false-premise compliance and calibrated uncertainty;
- reward-proxy gaming and hidden tests;
- distribution shifts for goal misgeneralization;
- train/deploy condition changes relevant to deceptive-behavior concerns;
- denial-of-service attempts against token, retrieval, and tool budgets.

## 4. Systems Metrics

Measure:

- cold and warm load time;
- time to first token;
- prefill tokens per second;
- decode tokens per second at batch sizes 1 and relevant larger batches;
- end-to-end time to a correct or check-passing answer;
- peak resident RAM and accelerator memory;
- model and index storage;
- KV-cache bytes per token;
- active and total parameter counts;
- estimated or measured FLOPs with method stated;
- energy per request/token where reliable telemetry exists;
- retrieval, verifier, repair, and tool overhead;
- speculative-decoding acceptance and net speedup.

Warmup, synchronization, sampling count, percentiles, clock/thermal conditions, and measurement tools are documented.

## 5. Comparison Matrix

Every novel model module is compared with the strongest practical subset of:

- same-parameter dense Transformer;
- same-active-FLOP dense Transformer;
- same-measured-latency model;
- deeper or wider dense control matching added compute;
- module-specific null control;
- established external baseline when licensing and runtime allow.

Matched parameter count and matched active compute answer different questions; neither may be substituted silently.

## 6. Required Ablations

An architecture result reports the full model, each module removed, key hyperparameter sweeps, and meaningful null/shuffled controls. Combination gains require interaction tests: if A and B each help alone, measure baseline, A, B, and A+B.

For verifier loops, report raw accuracy, accepted accuracy, coverage, false accept/reject rate, repair success, regressions introduced by repair, and extra cost. Selective accuracy without coverage is misleading.

For retrieval, separate retrieval recall, reader accuracy given gold evidence, end-to-end accuracy, citation linkage, and unsupported-claim rate.

## 7. Statistical Practice

- predeclare a primary metric and threshold;
- use multiple training seeds for claims sensitive to initialization;
- report mean, dispersion, and confidence intervals or bootstrap intervals where suitable;
- use paired tests for predictions on the same examples;
- correct or disclose multiple comparisons;
- publish per-example outputs or hashes when licensing permits;
- distinguish practical effect size from statistical significance.

## 8. Scaling and Training Phenomena

The Scaling Observatory records train/validation loss and task metrics against parameters, data tokens, optimizer steps, and compute. This is where double descent and grokking are studied. Neither is assumed to appear; plots and definitions are fixed before interpreting them.

Track repetition, diversity, router/expert collapse, and task specialization precisely. “Mode collapse” is reserved for its standard GAN context unless an experiment explicitly defines an analogous metric.

## 9. Benchmark Integrity

Final test sets are access-controlled or sealed until configuration selection ends. Prompts and answers are scanned against training data where feasible. Repeated manual inspection of a test set converts it into development data and requires a new test set.

Public benchmarks are supplemented with private or procedurally generated variants. No single leaderboard score determines architecture acceptance.

## 10. Acceptance Gates

A module can be accepted when:

1. the primary metric improves beyond the preregistered practical threshold on held-out data;
2. matched systems cost satisfies the hypothesis;
3. results survive required seeds and ablations;
4. no unacceptable safety, stability, or baseline regression appears;
5. the implementation and result can be reproduced;
6. operational complexity is justified;
7. a decision-log entry records the evidence and limitations.

A well-executed rejection is a successful research outcome.
