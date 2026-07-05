# Engineering and Research Principles

These principles are normative. Architecture decisions should cite the relevant principle when tradeoffs are disputed.

## 1. Evidence Before Complexity

Start with the smallest credible baseline. Add one mechanism, measure it against matched baselines, and retain it only if the evidence justifies its maintenance and runtime cost.

## 2. Breakthrough Target, Evidence-Gated Path

The project is allowed to pursue a new architecture, not merely optimize existing ones. However, ambition does not bypass measurement: a mechanism becomes part of the default path only when it beats matched baselines under declared resource, quality, robustness, and safety gates. SARN-Hybrid is the target; its final design is determined by evidence.

## 3. Separate Mechanism, Claim, and Metric

A graph update is a mechanism. “It reasons” is a claim. Exact-match accuracy on held-out compositional chains is one possible metric. Documentation and experiments must not substitute one for another.

## 4. Framework and Model Are Separate Products

Retrieval, permissions, audit logs, device selection, and deterministic verification belong in Aegis. Token prediction and learned latent transformations belong in a model. Boundaries may be researched, but side effects always remain framework-controlled.

## 5. Explicit Budgets

Every run declares resource limits: model bytes, peak RAM/VRAM, context tokens, generated tokens, wall time, reasoning iterations, retrieval count, tool calls, and energy where measurable. “Efficient” without a budget and comparator is not a result.

## 6. Graceful Degradation, Not Impossible Equivalence

Hardware profiles preserve API semantics and safety invariants. They do not promise equal intelligence. The runtime reports disabled capabilities and quality-affecting changes instead of silently reducing them.

## 7. Typed Information Boundaries

User input, retrieved text, memory, tool output, model proposals, verifier findings, and final answers retain origin and trust metadata. Untrusted text never becomes an instruction merely because it was retrieved.

## 8. No Hidden Mutation

Inference cannot modify base weights. Temporary state is scoped and resettable. Persistent writes require policy validation, provenance, and, where configured, user confirmation. Every external side effect passes through an auditable executor.

## 9. Reproducibility Is a Feature

Configurations, code revision, data manifests, random seeds, environment, tokenizer, checkpoint lineage, and evaluation versions are recorded. A result that cannot be reconstructed is a lead, not evidence.

## 10. Negative Results Stay Valuable

Rejected ideas and failed experiments remain in the experiment registry with conditions and metrics. This prevents the team from cycling through the same attractive failure.

## 11. Safety Is a System Property

Preference tuning is not a complete safety solution. Safety combines training, permissions, isolation, budgets, deterministic checks, monitoring, evaluations, incident response, and human authority.

## 12. Interpretability Does Not Equal Control

An SAE feature or probe correlation is a hypothesis about a representation. It requires causal intervention and robustness testing before it can support a control or safety claim.

## 13. Optimize the End-to-End System

Parameter count alone is not capability or cost. Measure active FLOPs, memory bandwidth, KV cache, routing communication, retrieval latency, verification overhead, and total time-to-correct-answer.
