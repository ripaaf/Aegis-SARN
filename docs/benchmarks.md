# Evaluation and Benchmarks

## 1. Evaluation Contract

Every reported result names the artifact and configuration digest, data and benchmark versions, prompting/evaluation code revision, hardware/backend, precision, context and generation limits, seed(s), sample count, uncertainty, and whether the result was exploratory or held out.

Scores without systems cost are incomplete for this project. Systems numbers without task quality are equally incomplete.

## 2. Evaluation Layers

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

## 3. Systems Metrics

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

## 4. Comparison Matrix

Every novel model module is compared with the strongest practical subset of:

- same-parameter dense Transformer;
- same-active-FLOP dense Transformer;
- same-measured-latency model;
- deeper or wider dense control matching added compute;
- module-specific null control;
- established external baseline when licensing and runtime allow.

Matched parameter count and matched active compute answer different questions; neither may be substituted silently.

## 5. Required Ablations

An architecture result reports the full model, each module removed, key hyperparameter sweeps, and meaningful null/shuffled controls. Combination gains require interaction tests: if A and B each help alone, measure baseline, A, B, and A+B.

For verifier loops, report raw accuracy, accepted accuracy, coverage, false accept/reject rate, repair success, regressions introduced by repair, and extra cost. Selective accuracy without coverage is misleading.

For retrieval, separate retrieval recall, reader accuracy given gold evidence, end-to-end accuracy, citation linkage, and unsupported-claim rate.

## 6. Statistical Practice

- predeclare a primary metric and threshold;
- use multiple training seeds for claims sensitive to initialization;
- report mean, dispersion, and confidence intervals or bootstrap intervals where suitable;
- use paired tests for predictions on the same examples;
- correct or disclose multiple comparisons;
- publish per-example outputs or hashes when licensing permits;
- distinguish practical effect size from statistical significance.

## 7. Scaling and Training Phenomena

The Scaling Observatory records train/validation loss and task metrics against parameters, data tokens, optimizer steps, and compute. This is where double descent and grokking are studied. Neither is assumed to appear; plots and definitions are fixed before interpreting them.

Track repetition, diversity, router/expert collapse, and task specialization precisely. “Mode collapse” is reserved for its standard GAN context unless an experiment explicitly defines an analogous metric.

## 8. Benchmark Integrity

Final test sets are access-controlled or sealed until configuration selection ends. Prompts and answers are scanned against training data where feasible. Repeated manual inspection of a test set converts it into development data and requires a new test set.

Public benchmarks are supplemented with private or procedurally generated variants. No single leaderboard score determines architecture acceptance.

## 9. Acceptance Gates

A module can be accepted when:

1. the primary metric improves beyond the preregistered practical threshold on held-out data;
2. matched systems cost satisfies the hypothesis;
3. results survive required seeds and ablations;
4. no unacceptable safety, stability, or baseline regression appears;
5. the implementation and result can be reproduced;
6. operational complexity is justified;
7. a decision-log entry records the evidence and limitations.

A well-executed rejection is a successful research outcome.
