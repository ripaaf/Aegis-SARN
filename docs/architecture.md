# System Architecture

## 1. Architectural Boundary

Aegis-SARN is a layered system, not a single neural network:

```text
Clients
  |
  v
Aegis API and Session Controller
  |
  +--> Policy / budgets / capability registry
  +--> Context, memory, and retrieval
  +--> Model runtime --------------------+
  |       |                              |
  |       +--> SARN or compatible model  |
  |                                      |
  +--> Candidate checks and repair       |
  +--> Tool executor / sandbox           |
  +--> Trace and evaluation store        |
  |                                      |
  +<-------------------------------------+
  v
Typed response, action result, or refusal
```

The **framework control plane** decides what may run and records what happened. The **model data plane** computes proposals over tensors. A model never directly opens a file, calls a network, writes memory, or actuates hardware.

## 2. Layer Responsibilities

### Client Layer

CLI, Python API, local service, and later UI clients send a versioned `RunRequest`. Clients may request capabilities but cannot grant themselves permissions.

### Session Controller

Owns one run: validation, budget reservation, context construction, model invocation, checks, tool transitions, finalization, cancellation, and trace emission.

### Policy and Capability Layer

Intersects four things:

```text
requested capability
AND profile capability
AND configured policy
AND user/session authorization
```

The result is an immutable per-run capability set. Denied or unavailable capabilities are reported explicitly.

### Context and Knowledge Layer

Builds context from user content, conversation history, retrieved passages, and approved memories. Every item retains provenance, timestamp, trust class, and token cost. See [Memory and retrieval](memory.md).

### Model Runtime

Loads a model package through a backend adapter, negotiates supported features, applies generation limits, and returns candidates plus machine-readable telemetry. It can host SARN variants or compatible third-party checkpoints.

### Verification and Repair

Runs task-specific checks. Deterministic validators are preferred when available; learned critics are fallible evidence sources, not authorities. Repair is bounded by iteration and cost limits and preserves the original findings in the trace.

### Tool Executor

Validates a structured tool call against schema and policy, obtains approval when required, executes it in the narrowest available sandbox, and returns typed output. Plain generated text is never executed.

### Observability and Evaluation

Records structured events, timings, budgets, model and config identity, memory and retrieval provenance, tool decisions, verifier results, and final status. Sensitive content is redacted or omitted according to policy.

## 3. SARN Model Boundary

SARN owns learned tensor computation; Aegis owns orchestration and effects. The project has two explicit model paths.

### 3.1 Phase 1 Control Path

```text
request
  -> Aegis Runtime
  -> context and budget preparation
  -> SARN-Dense
       embedding + RoPE
       dense causal Transformer blocks
       decoder head
  -> candidate output
  -> Aegis policy and configured checks
  -> final output + trace
```

SARN-Dense establishes correctness, training behavior, model/runtime contracts, and matched baselines. It is not the final architectural identity.

### 3.2 Long-Term SARN-Hybrid Path

```text
Aegis Runtime
  -> trusted policy + typed context + resource budget
  -> SARN-Hybrid
       embedding + RoPE
       hybrid sequence engine
         GQA attention
         optional selective SSM blocks
       dense control or sparse expert capacity
       latent workspace slots
       graph message-passing cycles
       resettable working-memory read/update
       gated writeback to token states
       decoder head
  -> candidate + model telemetry + verifier hooks
  -> Aegis retrieval/tool/verification/repair controls
  -> checked output, qualified output, refusal, or authorized action
```

This diagram is the declared SARN-Hybrid construction target, not a claim of present implementation or success. Core stages and optional accelerators remain independently configurable and ablatable. A dense/attention-only route stays available for controls and hardware fallback. Full details are in [SARN model](model.md).

### 3.3 Ownership Boundary

SARN-Hybrid may emit logits, structured call proposals, latent-state summaries, uncertainty-related signals, and instrumentation hooks. It may update only bounded run/session neural state allocated by Aegis. It cannot commit persistent memory, execute tools, acquire resources, alter policy, or decide that a verifier passed.

Aegis selects artifacts and budgets, constructs context, authorizes capabilities, stores persistent memory, runs retrieval and tools, invokes verifiers, limits repair, and records the trace. These invariants remain true even if future training makes SARN-Hybrid more capable.

## 4. End-to-End Flows

### 4.1 Text Generation

1. Validate the request and reserve budgets.
2. Assemble trusted system policy, conversation, and user content.
3. Select model, backend, and profile.
4. Decode within token, time, and memory budgets.
5. Apply output policy and return the response plus disclosed limitations.

### 4.2 Retrieval-Augmented Answer

1. Classify whether retrieval is permitted and useful.
2. Query allowed collections and retrieve candidate chunks.
3. Rerank, deduplicate, and enforce a context budget.
4. Place passages in a clearly marked untrusted-evidence section.
5. Generate an answer with passage identifiers.
6. Check that citations refer to supplied passages and quoted claims have support.
7. Return the answer with provenance. Citation linkage is not proof of truth.

### 4.3 Tool Use

1. The model proposes a schema-valid tool call.
2. Aegis validates capability, arguments, side-effect class, and budget.
3. The framework asks for approval if policy requires it.
4. The executor runs in isolation and captures result, error, and side effects.
5. The model may interpret the typed result; it cannot alter the audit record.

### 4.4 Candidate–Verify–Repair

1. Generate one or more candidates.
2. Run deterministic and task-specific checks.
3. If checks fail and budget remains, provide findings to a repair pass.
4. Re-run checks from scratch.
5. Return a passing candidate, a qualified answer with failed checks, or an abstention.

Verifier loops always have a hard cap.

## 5. Trust Zones

| Zone | Examples | Default trust | May cause side effects? |
|---|---|---:|---:|
| Policy root | signed configuration, hard limits | high | defines permission only |
| Framework core | session controller, executors | high but tested | through controlled adapters |
| Model | generated tokens and calls | untrusted proposal | no |
| Retrieval | local docs, web pages, databases | evidence with source-specific trust | no |
| Session memory | conversation-derived state | untrusted or qualified | no |
| Verifier | tests, compilers, critics | varies by verifier | no direct side effects |
| Tool adapter | filesystem, shell, network, robot | privileged boundary | yes, with authorization |

## 6. Failure Semantics

Every run ends in one explicit state: `completed`, `partial`, `refused`, `budget_exhausted`, `cancelled`, `validation_error`, `model_error`, `tool_error`, or `internal_error`. Partial output must not be mislabeled as verified output. Retries are bounded and idempotency keys are required for side-effecting tools.

## 7. Cross-Cutting Invariants

- Base model weights are immutable during serving.
- Persistent memory and tool side effects are opt-in and audited.
- Resource limits are enforced outside model-generated text.
- Feature flags identify experimental paths in traces and artifacts.
- Model packages are content-addressed and include tokenizer and config lineage.
- Framework APIs are versioned independently from model checkpoints.
- Safety-critical decisions cannot rely only on learned self-critique.

## 8. Evolution Strategy

Text-only local inference comes first. Retrieval and deterministic verification follow stable contracts. Novel SARN modules are evaluated in the research package before runtime integration. Multimodal encoders and action models are later adapters; they do not destabilize the core request, policy, trace, and artifact contracts.
