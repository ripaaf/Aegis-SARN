# Vision, Scope, and Success

## Mission

Build an open, inspectable AI research stack that extracts as much **verified useful capability** as possible from a declared hardware budget, and that can scale from constrained local machines to larger accelerators through explicit profiles rather than hidden degradation.

The project develops both the surrounding cognitive runtime and the models it runs. The runtime should also host compatible third-party models, so framework progress is not blocked on successfully inventing a new neural architecture.

## The Two Deliverables

### Aegis Framework

Aegis owns system-level responsibilities that do not belong inside model weights:

- hardware discovery and budget enforcement;
- model loading and capability negotiation;
- conversation and context assembly;
- temporary and persistent memory policy;
- retrieval with provenance;
- tool registration, authorization, execution, and audit;
- candidate generation, checking, and bounded repair;
- safety policy and human approval gates;
- telemetry, experiment traces, evaluation, and packaging.

### SARN Model Family

SARN owns learned computation:

- a small, reproducible decoder-only Transformer baseline;
- efficient attention experiments such as RoPE and grouped-query attention;
- optional sparse expert routing;
- optional state-space or hybrid sequence blocks;
- an experimental latent concept and graph module;
- resettable working-memory experiments;
- model heads and adapters needed by accepted tasks;
- interpretability instrumentation and distilled deployment variants.

## North Star

For a workload `W` and resource budget `B`, maximize verified task utility:

```text
maximize   verified_quality(W)
subject to RAM, VRAM, latency, energy, storage, and safety constraints in B
```

This formulation matters. “Runs on anything” becomes a family of supported profiles with measured capability. “Near perfect” becomes an error budget per task. “No loss” is not promised: quantization, pruning, fewer reasoning passes, smaller checkpoints, and reduced context can all lower quality.

## Target Users

- researchers testing efficient reasoning and memory hypotheses;
- engineers building local-first assistants and agents;
- students who need an architecture they can understand end to end;
- edge-device developers who need explicit resource budgets;
- evaluators studying where modular verification helps or fails.

## Initial Use Cases

The first use cases are deliberately narrow:

1. train and evaluate a tiny causal language model;
2. solve synthetic recall, rule-chain, and graph tasks;
3. run compatible local checkpoints behind one runtime interface;
4. answer over local documents with source provenance;
5. propose code or structured actions, then check them in a sandbox or deterministic validator.

Robotics, unrestricted computer control, native audio, image generation, and broad multimodality are later programs. They must enter through adapters and capability-specific safety cases.

## Success Definition

The project is succeeding when it can demonstrate all of the following:

- a reproducible baseline trained from a pinned configuration;
- an end-to-end framework path with typed traces and no hidden side effects;
- per-tier latency, memory, energy where measurable, and quality reports;
- at least one novel SARN module that improves a predeclared metric at matched active compute, or a well-documented negative result;
- ablations that isolate which module caused an improvement;
- retrieval and memory tests that distinguish recall from unsupported generation;
- safety controls that fail closed for unauthorized tools and persistence;
- releases that include model cards, data cards, evaluation cards, and known limitations.

## Non-Goals and Honest Limits

This project does not promise:

- AGI, consciousness, perfect truthfulness, or guaranteed advanced reasoning;
- frontier-model capability on tiny hardware;
- identical output or quality on every hardware profile;
- that graph activations are human concepts or calibrated beliefs;
- safe autonomous self-improvement or online mutation of base weights;
- a reliable “deception detector”—no such general detector is established;
- that combining many fashionable techniques will make them additive;
- one checkpoint that is simultaneously optimal for text, vision, audio, robotics, and every device.

## Long-Term Direction

If the language and framework foundations pass their gates, Aegis can add modality adapters: a vision encoder or VLM, segmentation through a dedicated model, speech encoders and decoders, and constrained action models. These are components in a governed system, not acronym checkboxes forced into a single network.
