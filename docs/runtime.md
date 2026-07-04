# Runtime, Hardware Profiles, and Deployment

## 1. Deployment Goal

“Run on anything” means one versioned Aegis contract across explicitly supported profiles. It does not mean one checkpoint, equal latency, equal context, or equal quality on all devices.

Profiles are selected from measured capacity and can be overridden. A run response discloses its profile, model, precision, disabled features, and relevant limits.

## 2. Profile Model

Initial names are semantic rather than fixed parameter promises:

| Profile | Typical environment | Expected behavior |
|---|---|---|
| Nano | constrained CPU/SBC | smallest quantized model, short context, no expensive search |
| Lite | modern CPU or small accelerator | local generation, bounded retrieval, light checks |
| Balanced | laptop/desktop accelerator | stronger checkpoint, larger context, verifier/repair budget |
| Pro | workstation/server accelerator | larger or sparse model, batching, deeper evaluations |
| Max | multi-accelerator research | distributed training/inference experiments, not edge parity |

Exact thresholds live in versioned profile files after measurement. Marketing labels never replace hardware facts.

## 3. Fit Estimation

Before loading, estimate:

```text
weight bytes
+ dequantization/runtime buffers
+ KV cache(context, batch, layers, KV heads, dtype)
+ activation/scratch workspace
+ backend overhead
+ retrieval/model auxiliary memory
+ safety margin
```

The planner rejects unsafe fits rather than relying on operating-system swapping by accident. It may propose a smaller model, shorter context, lower batch, different precision, CPU offload, or disabled optional module with an estimated impact.

## 4. Runtime Planning

The planner chooses:

- model/checkpoint and backend;
- weight, activation, and KV-cache precision;
- device placement and offload;
- context and output budget;
- batch and concurrency limit;
- attention implementation;
- reasoning/workspace cycle cap;
- retrieval and verifier budget;
- speculative decoding only when net beneficial.

Safety policy is never weakened to make a model fit. A minimal deterministic authorization path remains available even if optional critics are disabled.

## 5. Quantization

Quantization choices are artifact- and backend-specific. Evaluate weight-only integer formats, weight/activation formats, and KV-cache quantization separately. Report calibration corpus, algorithm, group size or granularity, kernel, file size, peak memory, latency, energy, and quality deltas by task.

The framework does not infer quality from bit width. Quantized checkpoints have distinct digests and cards. Sensitive layers may remain at higher precision when measurements justify it.

## 6. Speculative Decoding

The planner enables speculative decoding only if draft and target tokenization/algorithm are compatible and benchmarked on the current workload. Two resident models can increase memory and hurt constrained devices. Acceptance rate alone is not success; end-to-end latency and exact sampling behavior matter.

## 7. Packaging

A release bundle contains:

- framework package and locked dependency metadata;
- model manifest, weights, tokenizer, and generation defaults;
- profile compatibility and measured resource table;
- model/data/evaluation/safety cards;
- checksums and optional signatures;
- migration and rollback instructions;
- license and attribution files;
- a smoke-test command and expected result.

Framework and model can release independently. Compatibility is a range, not an assumption that “latest works.”

## 8. Deployment Shapes

### Embedded Library

Best for research and local applications. The host process owns lifecycle; tools still pass through Aegis policy.

### Local Service

Provides process isolation, a stable API, concurrency control, and shared model residency. Listen locally by default; remote access requires authentication, TLS, and a new network threat review.

### Research Cluster

Separates scheduler, workers, artifact store, and experiment tracker. Distributed capabilities are not exposed automatically through the local assistant API.

### Edge Device

Uses signed or checksum-verified artifacts, strict storage and thermal budgets, offline-first behavior, bounded logs, safe update/rollback, and a device-specific test matrix. Physical-actuator access is a separate adapter with independent interlocks.

## 9. Performance Method

Benchmark cold/warm startup, prompt lengths, output lengths, batch/concurrency levels, and representative workflows. Report median and tail latency after warmup; synchronize accelerators; control clocks/thermal state where possible; record OS, drivers, backend, threads, and power mode.

Regression thresholds apply to both quality and systems metrics. A faster release that crosses the quality floor is rejected.

## 10. Graceful Degradation Order

Suggested order, subject to task policy:

1. reduce concurrency or batch;
2. choose a memory-efficient kernel/backend;
3. reduce nonessential retrieval candidates or repair attempts;
4. shorten output and then context with disclosure;
5. choose an evaluated quantization;
6. choose a smaller evaluated checkpoint;
7. disable experimental cognitive modules;
8. refuse the run if required safety/correctness controls cannot fit.

The planner does not silently convert a high-assurance request into unchecked generation.

## 11. Release Validation

Each supported OS/hardware/backend matrix cell gets artifact integrity, load/unload, generation, cancellation, memory-pressure, long-context, concurrency, failure recovery, and security smoke tests. Unsupported combinations are labeled unsupported rather than implied by generic “CPU/GPU” language.
