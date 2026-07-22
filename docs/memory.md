# Memory and Retrieval

## 1. Memory Taxonomy

“Memory” is not one subsystem. Aegis distinguishes:

| Kind | Lifetime | Storage | Primary use | Writes controlled by |
|---|---|---|---|---|
| model weights | release | model artifact | learned general patterns | training pipeline |
| token context | one model call | RAM/VRAM | immediate prompt state | context builder |
| KV cache | one decoding call | RAM/VRAM | efficient autoregressive decoding | backend |
| neural working memory | run/session | runtime state | experimental associations | model runtime + policy |
| session memory | session | memory service | conversation continuity | Aegis |
| persistent user/workspace memory | cross-session | database | approved facts/preferences | Aegis + authorization |
| retrieval corpus | corpus version | index + source store | external evidence | ingestion pipeline |

These types have different correctness, privacy, deletion, and threat properties. No API should return an unqualified generic “memory.”

## 2. Context Management

The context builder receives items with provenance and allocates a fixed token budget. A recommended priority order is:

1. trusted system policy and tool contracts;
2. current user input;
3. required tool results and verifier findings;
4. recent relevant dialogue;
5. selected retrieval evidence;
6. approved memories;
7. older summaries.

Priority is configurable but policy cannot be displaced. Truncation happens at content boundaries where possible. The trace records selected, compressed, and omitted item IDs plus their token costs.

Summaries are model-generated derived data and retain links to source turns. They are not authoritative replacements for source content.

## 3. Neural Working Memory

Phase 7 implements a limited subset of this track: fixed-size neural slots behind an explicit feature flag, after the enabled latent workspace and optional graph. State is a caller-visible tensor carried in the generation cache, not a module-global value or stored record. A call without the tensor starts empty, so independent runs reset by construction. The implementation is disabled by default and does not mutate released weights during serving.

This prototype is not persistent or user memory, retrieval, self-learning, or a memory service. The broader SARN working-memory experiment requires state that is:

- allocated empty at run or session start;
- isolated across concurrent users and batch rows;
- bounded in slots, bytes, update count, and norm;
- optionally decayed;
- inspectable through summary statistics without exposing private vectors;
- explicitly reset;
- never serialized as long-term memory by default.

Phase 7 covers deterministic key/value, distractor, overwrite, tiny capacity, delayed-copy, batch-row, and independent-call reset/isolation checks. The complete workstream must later add recency curves, poisoned-write evaluation, stronger information-leakage tests, and comparisons with the same facts in token context and an Aegis key/value store. Those later tests and services are not implemented by Phase 7.

## 4. Persistent Memory

Persistent memory is a framework feature with an explicit write protocol:

```text
candidate fact
 -> classify sensitivity and owner
 -> attach source and confidence
 -> detect conflict/duplication
 -> policy/approval decision
 -> append record or reject
 -> emit audit event
```

Preferred records are concise facts or preferences, not raw hidden prompts. Users can inspect, correct, export, and delete their records. Expiration and retention are first-class. Corrections create lineage through `supersedes` rather than silently rewriting history.

Never persist secrets, authentication material, health/legal/financial details, third-party personal data, inferred protected traits, or unsafe instructions without a specific product policy and consent model.

## 5. Retrieval Pipeline

### Ingestion

1. authorize and identify the source;
2. preserve immutable source bytes or a source digest;
3. parse content and record parser version;
4. normalize without losing location mapping;
5. classify sensitivity and access control;
6. chunk with stable chunk IDs;
7. compute lexical representation and optional embeddings;
8. build a versioned index;
9. run quality checks and publish atomically.

### Query

1. derive a search query without treating retrieved text as policy;
2. filter by caller access before ranking;
3. retrieve lexical and/or vector candidates;
4. fuse and rerank;
5. deduplicate near-identical chunks;
6. diversify sources when useful;
7. enforce context and latency budgets;
8. return content envelopes with provenance.

### Generation and Attribution

Retrieved passages appear as untrusted evidence, delimited from instructions. Generated citations reference chunk IDs that can be resolved to source locations. Citation validation checks existence and textual support. It does not certify that a source is accurate, current, unbiased, or complete.

## 6. Retrieval Evaluation

Report retrieval separately from answer generation:

- recall@k, precision@k, MRR or nDCG as appropriate;
- answer-bearing-chunk recall;
- reranker improvement;
- citation precision and recall;
- attributed-answer correctness;
- abstention when evidence is absent;
- freshness and deletion propagation;
- cross-user and access-control isolation;
- prompt-injection resistance;
- latency, index size, and embedding cost.

## 7. Poisoning and Prompt Injection

Retrieved documents and memories can contain malicious instructions. Defenses include typed trust boundaries, instruction hierarchy, source access controls, ingestion scanning, query-time filtering, output checks, and tool permission isolation. No textual prompt alone is considered a complete defense.

Memory poisoning tests attempt to store false identity claims, override policy, smuggle tool commands, create self-reinforcing summaries, and transfer data across users. Aegis fails persistent writes closed when owner, source, or authorization is ambiguous.

## 8. Deletion and Backup Semantics

Deletion behavior must state whether it covers primary records, indexes, caches, traces, backups, and derived embeddings; how long propagation takes; and what legal or operational retention exceptions apply. Tombstones prevent deleted material from reappearing during an index rollback. Backup restore procedures replay deletions before reopening retrieval.
