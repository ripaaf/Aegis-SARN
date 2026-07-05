# Interfaces and Shared Schemas

This document defines the full conceptual contracts. Phase 1 implements a deliberately smaller typed subset in `aegis_sarn.config` and `aegis_sarn.aegis.schemas`; field names should remain stable unless an architecture decision records a replacement.

## Phase 1 Implemented Subset

- `RunRequest`: text prompt, request/session IDs, prompt/output token budgets, wall-time budget, seed, decoding strategy, temperature/top-k/top-p controls, optional stop token, opt-in KV cache, and schema version.
- `RunResult`: explicit status, text content, backend identity, usage, limitations, assurance placeholder, ordered trace events, and an optional manifest path.
- `TraceEvent`: run ID, monotonic sequence, event/component identity, UTC timestamp, typed JSON payload, and schema version.
- configuration dataclasses: model, training, runtime, decoding, seed, artifact, and run manifest.

Retrieval capabilities, multipart content envelopes, tools, verifier findings, persistent memory, and full model manifests remain future contracts. The Phase 1 implementation must not imply those capabilities exist.

## 1. Compatibility Rules

- Public schemas use explicit semantic versions.
- Producers may add optional fields in a compatible minor release.
- Consumers ignore unknown optional fields but reject unknown enum values when safety depends on them.
- Breaking field removal or semantic changes require a major version.
- Stored artifacts include the schema version used to create them.
- Framework API and model-package format version independently.

## 2. Run Request

```yaml
schema_version: aegis.run_request/v1
request_id: uuid
session_id: uuid-or-null
input:
  parts:
    - kind: text
      content: ...
requested_capabilities: [generate, retrieve]
generation:
  max_new_tokens: 256
  temperature: 0.0
  seed: 7
budget:
  wall_time_ms: 30000
  max_model_calls: 2
  max_tool_calls: 0
profile: auto
metadata: {}
```

Unknown capabilities never imply permission. Server-side limits can only tighten request budgets.

## 3. Run Response

```yaml
schema_version: aegis.run_response/v1
request_id: uuid
run_id: uuid
status: completed
output:
  parts:
    - kind: text
      content: ...
assurance:
  state: unchecked
  checks: []
provenance: []
usage:
  prompt_tokens: 0
  generated_tokens: 0
  wall_time_ms: 0
limitations: []
```

`assurance.state` is one of `unchecked`, `checks_passed`, `checks_failed`, or `mixed`; it never uses the vague value `verified` without naming checks.

## 4. Content Envelope

Every context item is wrapped with:

```yaml
content_id: digest-or-uuid
kind: text | image | structured | binary_reference
origin: user | policy | memory | retrieval | tool | model | verifier
trust: trusted_policy | authorized_data | untrusted_data | model_generated
content: ...
source_ref: optional-uri-or-artifact-id
created_at: timestamp
expires_at: optional-timestamp
labels: []
```

Trust is assigned by policy, not inferred from prose.

## 5. Model Manifest

```yaml
schema_version: aegis.model_manifest/v1
model_id: sarn-baseline-10m
artifact_digest: sha256:...
architecture: sarn_decoder
architecture_version: 1.0
tokenizer_artifact: sha256:...
weight_format: safetensors
parameter_count: 10000000
active_parameter_count: 10000000
context_limit: 2048
dtypes: [fp32, bf16]
capabilities: [generate, score]
experimental_features: []
minimum_runtime: 0.1.0
license: ...
cards:
  model: path-or-uri
  data: path-or-uri
```

The manifest contains measured memory by backend/profile once available; parameter count is not used as a memory estimator by itself.

## 6. Tool Contract

```yaml
tool_id: namespace.name
version: 1.0.0
side_effect: none | reversible | irreversible | external_communication
input_schema: {...}
output_schema: {...}
required_capabilities: [...]
approval: never | policy | always
timeout_ms: 10000
idempotency: unsupported | key_required | inherent
```

A tool request contains only structured arguments. The result distinguishes normal data, stderr or diagnostics, side-effect receipts, and execution errors.

## 7. Verifier Finding

```yaml
checker_id: python.unit_tests
checker_version: 1.2.0
assurance_class: deterministic
status: pass | fail | inconclusive | error
severity: info | warning | error | critical
subject_ref: candidate-id
location: optional-structured-location
message: ...
evidence_refs: []
```

An `error` or `inconclusive` check does not count as a pass.

## 8. Memory Record

```yaml
memory_id: uuid
owner_id: opaque-owner-reference
scope: run | session | user | workspace
content: {...}
source_refs: [...]
confidence: optional-calibrated-value
created_at: timestamp
expires_at: optional-timestamp
sensitivity: public | internal | private | restricted
write_authorization: {...}
supersedes: optional-memory-id
```

Memory records are append-oriented. Corrections supersede prior records; destructive deletion follows the store's privacy and retention contract.

## 9. Trace Event

Every event has `run_id`, `sequence`, `event_type`, timestamps, component and artifact versions, payload schema, and redaction status. Event payloads include IDs rather than duplicating sensitive content where possible. Traces support replay of control decisions, not necessarily bit-identical accelerator computation.
