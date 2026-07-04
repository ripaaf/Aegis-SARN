# Development Workflow

## 1. Definition of a Change

A change is complete when code, tests, configuration/schema updates, documentation, compatibility notes, and relevant benchmark evidence land together. Experimental code can be incomplete in capability but not ambiguous in status.

## 2. Environment

Use a locked Python environment once implementation begins. The supported Python and accelerator matrix will be chosen in Phase 1 and recorded in `pyproject.toml` and CI. Optional backend dependencies live in extras so a CPU-only framework installation does not require every accelerator stack.

Commands are exposed through a small documented task interface rather than developer-specific shell history.

## 3. Code Standards

- typed public interfaces and validated configuration;
- small modules with explicit ownership of state;
- no import-time model loading, network calls, or environment mutation;
- structured errors and logs, not string matching for control flow;
- injectable clocks, stores, backends, and executors for tests;
- explicit device and dtype movement in tensor code;
- reference kernels retained until optimized implementations prove parity;
- comments explain invariants and tradeoffs, not obvious syntax.

## 4. Test Pyramid

### Unit

Tensor shapes, masks, numerical parity, config validation, budgeting, state transitions, and pure policy logic.

### Contract

Reusable suites every model backend, memory store, retriever, verifier, and tool adapter must pass.

### Integration

Tiny model train/save/load/generate; framework request through fake and real backend; retrieval attribution; tool approval; cancellation and recovery.

### Security

Prompt injection across trust zones, path and URL validation, secret redaction, cross-session isolation, corrupt/untrusted artifacts, unauthorized persistence, and audit integrity.

### Performance

Pinned microbenchmarks and representative end-to-end workloads. Performance tests do not replace correctness assertions.

## 5. Tensor-Specific Tests

- exact shapes and dtypes;
- finite forward/backward values;
- gradient presence or intentional absence;
- causal invariance to changed future tokens;
- padding and packing equivalence;
- KV-cache versus full-prefix logits;
- CPU reference versus accelerated kernel tolerance;
- save/load output parity;
- deterministic behavior under supported seeded mode;
- batch-row isolation for working memory;
- sparse router capacity and token conservation.

## 6. Pull Request Expectations

A change summary answers: what problem, why this layer, which contract changed, how tested, measured cost, safety/privacy impact, artifact/config migrations, and rollback. Research PRs link their preregistration and include all seeds, not only the best run.

Generated or AI-assisted code is reviewed under the same standard. The author owns correctness and licensing.

## 7. Experiment Lifecycle

1. write a hypothesis and comparison plan;
2. assign an experiment ID and freeze configs;
3. run smoke tests;
4. execute the budgeted runs;
5. validate artifacts and aggregate without hiding failures;
6. write a report with raw artifact links;
7. accept, reject, or continue with a stated reason;
8. record architectural consequences in the decision log.

Exploratory notebooks are allowed, but final preprocessing and metrics move to tested scripts/modules before supporting a claim.

## 8. Documentation Maintenance

Normative behavior uses “must,” recommendations use “should,” and possibilities use “may.” Documents state whether a component exists. Cross-links are checked in CI. Architecture diagrams, schemas, and examples are updated with interface changes.

## 9. Versioning and Releases

Use semantic versions for public framework APIs and artifact schemas. Pre-1.0 changes may move quickly but still include migrations. Model artifacts use stable IDs plus immutable digests. Releases are built from a clean revision and include an SBOM or dependency inventory when distribution begins.

## 10. Security Hygiene

Never commit secrets, private data, raw user traces, licensed datasets, or unreviewed model files. Avoid unsafe pickle-style loading for untrusted artifacts. Pin and scan dependencies. Security-sensitive defaults change only with an explicit review and tests.

## 11. First Coding Issue

The recommended first implementation issue is the dense micro-model vertical slice: validated model/training config, RoPE causal attention, one decoder block, full model, tiny generated dataset, one-batch overfit, checkpoint round trip, greedy generation, and an Aegis fake/real backend contract. This creates a spine that future research can extend without pretending the entire roadmap exists.
