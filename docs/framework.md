# Aegis Framework Specification

## 1. Purpose

Aegis is the reusable runtime and control plane built to train, evaluate, govern, and deploy the SARN-Hybrid program. It also hosts SARN-Dense and compatible third-party models behind the same contracts, preserving a working system and comparison path while individual hybrid mechanisms are still being validated.

The framework is responsible for orchestration and effects, not for pretending that orchestration is part of model intelligence.

**Phase 1 implementation status:** validated run/config dataclasses, a synchronous session controller, token and wall-time budgets, fake and SARN-Dense backends, opt-in KV-cached decoding, ordered trace events, reproducibility manifests, toy evaluation, CPU microbenchmarking, and JSON CLI paths exist. Wall time is measured and reported around synchronous generation; cooperative cancellation/preemption is not yet implemented. General hardware profiling, an artifact registry, retrieval, persistent memory, tools, policy bundles, verification, cancellation, and service APIs remain specifications only.

## 2. Required Subsystems

### 2.1 Configuration

Configuration is hierarchical and immutable within a run:

```text
built-in safe defaults
  <- deployment configuration
  <- hardware profile
  <- model package constraints
  <- permitted request overrides
```

The resolved configuration is validated against a versioned schema, assigned a digest, and stored in the trace. Secrets are referenced by handles and never serialized into ordinary configuration or logs.

### 2.2 Hardware Profiler

Collects CPU architecture and features, logical and physical cores, system RAM, accelerator type and memory, supported numerical formats, available disk, and optional thermal or energy telemetry. Detection produces facts; a separate planner maps facts to a profile.

Selection must support an explicit override and a dry run explaining why a model fits or does not fit. Memory estimates include weights, runtime overhead, activations, KV cache, retrieval index, and safety margin.

### 2.3 Artifact Registry

Manages local and remote model packages, tokenizers, adapters, indexes, and evaluation bundles. Each artifact has:

- stable logical name and immutable content digest;
- format and schema version;
- size and checksums;
- source and license metadata;
- compatible backends and minimum resources;
- parent and transformation lineage;
- model, data, or evaluation card reference;
- trust status and signature when available.

Loading fails closed on checksum or compatibility mismatch.

### 2.4 Backend Adapter

Defines the narrow interface between Aegis and inference engines. The first backend should favor correctness and observability over breadth. Later adapters may target PyTorch, ONNX Runtime, llama.cpp-compatible formats, or vendor accelerators, but backend-specific objects do not leak into public APIs.

Required operations are `inspect`, `load`, `generate`, `embed` when supported, `cancel`, `health`, and `unload`. Streaming events must retain run and candidate identifiers.

### 2.5 Session Controller

Implements a finite state machine:

```text
created -> validated -> prepared -> running
running -> awaiting_approval -> running
running -> verifying -> repairing -> verifying
running -> completed | partial | failed | cancelled
```

State transitions are explicit events. Cancellation propagates to retrieval, model decoding, verifiers, and tools. Side-effecting retries require idempotency support.

### 2.6 Context Builder

Combines policy content, recent dialogue, selected summaries, approved memories, retrieval evidence, and tool results within a token budget. Allocation is deterministic for a fixed input/config and reports dropped items. It must resist instruction smuggling by preserving typed boundaries and source labels.

### 2.7 Memory Service

Exposes session and persistent stores through policy-checked read/write methods. Models can propose a memory operation but cannot commit it. Retention, encryption, deletion, export, conflict resolution, and user ownership live here. See [Memory and retrieval](memory.md).

### 2.8 Retrieval Service

Owns ingestion, parsing, chunking, lexical and vector indexes, query rewriting, hybrid retrieval, reranking, access control, deduplication, and provenance. Index versions are artifacts. Deleted or unauthorized source material must not remain silently retrievable.

### 2.9 Tool Registry and Executor

Every tool declares:

- versioned input and output schemas;
- read-only or side-effecting classification;
- required capabilities and approval policy;
- timeout, concurrency, and rate limits;
- sandbox and network requirements;
- redaction rules and audit fields;
- idempotency behavior and rollback support, if any.

The executor treats model arguments as untrusted. It validates paths, URLs, identifiers, and bounded values independently of the model.

### 2.10 Verification Service

Registers checkers by task and assurance class:

- **deterministic**: parser, schema validator, unit test, compiler, solver, checksum;
- **grounded**: citation entailment or source linkage, still limited by source quality;
- **statistical**: classifier, reward model, anomaly detector;
- **generative**: critic model or self-critique, lowest assurance without corroboration.

Findings are structured by severity, location, evidence, and checker version. A score alone is insufficient for high-impact decisions.

### 2.11 Policy Engine

Evaluates explicit rules over actor, capability, resource, data class, tool, arguments, and environment. Policy decisions include `allow`, `deny`, and `require_approval`, with a machine-readable reason. Safety defaults are independent of the selected language model.

### 2.12 Trace and Metrics Store

Emits append-only events with monotonic sequence numbers. Minimum metrics include queue time, time to first token, tokens per second, peak RAM/VRAM, prompt and generated tokens, retrieval and verifier latency, tool calls, budget use, stop reason, and error class.

Content logging is off or minimized by default for private deployments. Metrics and audit content have distinct retention controls.

## 3. Public Surfaces

The initial supported surfaces are:

1. a Python library for embedding and research;
2. a CLI for local use and diagnostics;
3. a local HTTP service after library contracts stabilize.

The CLI and server call the same application layer. They may not reimplement policy or bypass tracing.

## 4. Extension Points

- model backend;
- model package adapter;
- retriever and reranker;
- memory store;
- tool provider;
- verifier;
- policy bundle;
- hardware probe;
- trace exporter;
- modality encoder and renderer.

Extensions run with least privilege and declare compatibility versions. Third-party extension failures cannot corrupt the session controller state.

## 5. Framework Test Strategy

- unit tests for schemas, budgets, state transitions, and policy;
- contract tests shared by every backend, store, tool, and verifier;
- property tests for token/resource allocation and serialization round trips;
- fault injection for cancellation, timeout, corrupt artifacts, and partial streams;
- integration tests with a deterministic fake model before real inference;
- security tests for path traversal, prompt injection across trust zones, privilege escalation, secret leakage, and audit tampering;
- performance tests with pinned workload and environment metadata.

## 6. Minimum Viable Framework

The first vertical slice includes configuration, one hardware probe, a local artifact manifest, a fake deterministic backend, one real backend, the session state machine, token budgets, structured traces, a CLI, and tests. Retrieval, persistent memory, tools, repair loops, and remote services are deliberately not required for that first slice.

## 7. Definition of Done

A framework release is done only when its schema versions, compatibility range, migration notes, threat-model changes, benchmark results, known issues, and rollback instructions are published. “It runs on my machine” is a development observation, not a release gate.
