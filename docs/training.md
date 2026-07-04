# Training and Experiment Plan

## 1. Training Philosophy

Training proceeds from executable invariants to language capability to isolated research hypotheses. A large pretraining run is not a substitute for proving that masking, packing, checkpointing, and evaluation are correct.

Every run starts from a frozen configuration and produces a run manifest, logs, checkpoints, validation metrics, systems telemetry, and terminal status.

## 2. Stage 0 — Correctness Harness

Before useful-scale training:

- overfit one batch and a tiny dataset;
- verify shifted labels and causal masking;
- compare loss with a simple reference implementation;
- test padding, packing, and document-boundary behavior;
- test gradient accumulation against an equivalent batch;
- interrupt and resume with matching optimizer, scheduler, data position, and RNG state;
- verify evaluation does not update parameters or data statistics;
- run NaN/Inf and gradient-norm guards;
- round-trip every checkpoint and tokenizer artifact.

Exit only when all invariant tests are automated.

## 3. Stage 1 — Dense Baseline

Train micro and tiny decoder-only Transformers on pipeline-validation and selected language data. Establish loss curves, scaling behavior, throughput, peak memory, generation behavior, and benchmark baselines.

Default objective:

```text
L_lm = -sum_t log p(token_t | token_<t)
```

Record tokens seen rather than only epochs. Validation is performed on immutable shards at fixed token intervals.

## 4. Stage 2 — Synthetic Generalization Baselines

Train and evaluate on controlled algorithmic generators. Split by structural dimensions, not random examples only. Plot train and validation performance over long enough schedules to observe delayed generalization where affordable.

Grokking is an observation if delayed generalization occurs. It is not forced into the roadmap as guaranteed. Weight decay, data size, model size, and training duration are controlled in a dedicated study.

## 5. Stage 3 — Single-Variable Architecture Experiments

Run independent tracks in this order unless evidence changes it:

1. MHA versus GQA;
2. latent workspace without graph edges;
3. graph/message-passing variants;
4. resettable working memory;
5. attention versus SSM and hybrid blocks;
6. dense FFN versus sparse MoE at a scale where routing is meaningful.

Do not combine winning-looking modules until individual effects and interactions are measured. Factorial experiments are used for combinations when budget permits.

## 6. Stage 4 — Language Pretraining

Only accepted or high-priority variants enter language pretraining. Compare against a dense baseline using the same tokenizer, data order or statistically equivalent shuffle, token budget, optimizer family, schedule, and evaluation suite.

The primary result includes validation loss versus tokens and compute, downstream quality, instability, throughput, energy where available, and total engineering complexity.

## 7. Stage 5 — Post-Training

### Supervised Fine-Tuning

Train on versioned instruction examples using prompt masking rules that prevent user text from becoming a target unless deliberately modeled. Preserve a base checkpoint for capability comparisons.

### Preference Optimization

DPO is the first planned preference method because it has a relatively simple training loop. Required inputs are chosen/rejected pairs, a frozen or defined reference policy, and a predeclared beta. Report preference accuracy, reward margins, KL-related behavior, capability regressions, calibration, and safety results.

Direct preference optimization is not “preference automation.” Human or process-generated preferences can be wrong, inconsistent, or gameable. Audit shortcut features such as response length and refusal style.

### Optional Tool/Structured-Output Tuning

Train schema-following separately and evaluate invalid-call rate, unnecessary-call rate, argument correctness, and refusal under missing permission. The model learns to propose calls; Aegis remains the authority.

## 8. Stage 6 — Distillation and Compression

Candidate students may learn from teacher logits, selected outputs, verifier-filtered solutions, or intermediate representations. Document teacher identity, sampling policy, temperature, loss weights, synthetic-data proportion, and rejected examples.

Quantization is evaluated after baseline accuracy is frozen. Calibration data is disjoint from final tests. Pruning studies compare unstructured sparsity, structured removal, and iterative retraining. Lottery-ticket-inspired rewinding is a research track, not assumed to produce a deployable winner.

## 9. Optimizers and Numerical Strategy

AdamW is the reference optimizer because it gives a familiar baseline. Optimizer alternatives—including Shampoo or distributed second-order approximations—are separate experiments justified at scales where convergence benefit can repay memory and implementation cost.

For every optimizer report parameter groups, betas or preconditioner settings, epsilon, weight decay exclusions, gradient clipping, learning-rate schedule, warmup, effective batch in tokens, numerical formats, loss scaling, and update count.

Mixed precision must preserve sensitive reductions and optimizer state as required. Numerical changes are validated against a high-precision short-run control.

## 10. Checkpointing and Recovery

A resumable checkpoint contains model weights, optimizer and scheduler state, scaler state, RNG states, data-loader position, training counters, configuration digest, code revision, tokenizer digest, and dataset-mixture digest. Write atomically through a temporary artifact and validate before marking complete.

Retention keeps recent recovery checkpoints and named milestone checkpoints. Deletion policy considers storage budget and reproducibility requirements.

## 11. Distributed Training

Distributed execution is deferred until one-device correctness. When added, topology, sharding strategy, collective library, world size, rank mapping, precision, gradient synchronization, expert parallelism, and fault behavior enter the run manifest.

Scaling efficiency is reported relative to the one-device or smallest practical control. MoE communication and token imbalance are measured explicitly.

## 12. Experiment Protocol

Every hypothesis has a short preregistration:

```text
question
mechanism and predicted effect
primary and secondary metrics
baseline and controlled variables
dataset/splits
resource budget
number of seeds
stopping rule
accept/reject threshold
known confounders
```

Exploratory results are labeled exploratory. Final configurations are evaluated once on held-out test data after selection.

## 13. Reproducibility Manifest

Minimum fields are run ID, parent runs, source revision and dirty flag, environment lock digest, hardware and drivers, full resolved configuration, seeds, artifact digests, command, start/end times, status, metrics files, and notes. Reports link to raw result artifacts rather than copying only the best number.

## 14. Failure Policy

OOMs, NaNs, preemption, corrupt data, divergence, and operator cancellation are distinct statuses. Failed runs remain indexed. Automatic recovery is bounded; it must not silently change batch size, precision, model shape, data, or hyperparameters and then claim to be the original run.
