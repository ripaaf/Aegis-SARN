# Aegis-X Grand Plan

## 1. Purpose

**Aegis-X** is the long-term plan to create and evaluate a new advanced hybrid AI architecture and governed runtime that can scale from constrained local hardware to powerful servers while maximizing verified intelligence per active compute, memory, energy, and latency budget.

It is the complete-system research hypothesis above the two model lines:

```text
Aegis-X
  = SARN-Hybrid learned model core
  + Aegis governed runtime
  + external memory, retrieval, tools, and verification
  + safety, interpretability, and deployment systems
  + future multimodal and action adapters
```

SARN-Dense remains the control used to determine whether Aegis-X mechanisms provide real value. Aegis-X does not replace the Aegis-SARN project name; it names the farthest system architecture this research program intends to test.

**Current status:** specification only. No Aegis-X model or runtime implementation exists yet.

## 2. What It Is Not

Aegis-X is:

- not a finished algorithm;
- not a claim of AGI, consciousness, or universal reasoning;
- not a promise of perfection or absence of failure modes;
- not permission for uncontrolled self-learning or base-weight mutation;
- not a claim that weak hardware can equal large server models without loss;
- not a monolithic network that forces every modality and mechanism into every run;
- not evidence that graph nodes are human concepts or that model activations are calibrated beliefs;
- not safe merely because a model can critique its own output.

“Near-perfect” behavior is allowed only as an asymptotic research direction: continually reduce measured error while documenting what remains unsolved.

## 3. What It Tries to Become

Aegis-X aims to become a hardware-adaptive cognitive system combining:

- SARN-Hybrid as the learned generative and reasoning core;
- Aegis as the control plane for budgets, policy, effects, and evidence;
- efficient RoPE/GQA sequence computation;
- optional local-attention and selective-SSM blocks;
- sparse expert capacity where routing is beneficial;
- a latent graph workspace with bounded message-passing cycles;
- resettable working memory without serving-time base-weight mutation;
- governed retrieval and persistent memory with provenance;
- deterministic and learned verification with bounded repair;
- interpretability hooks, sparse feature dictionaries, and causal analysis;
- permissioned tools and constrained action planning;
- evaluated compression and hardware profiles;
- future text, image, audio, file, sensor, and structured-input adapters.

The architectural claim is not that more modules automatically create more intelligence. The claim to test is that separating these responsibilities and activating them conditionally may produce a better capability-cost-safety frontier than a matched dense Transformer-only system.

## 4. Core Algorithm Hypothesis

A dense causal Transformer performs sequence understanding, factual recall, temporary state tracking, implicit planning, and output generation through repeated dense token transformations. SARN-Hybrid tests a different decomposition:

```text
sequence understanding
  -> sparse latent concept/state representation
  -> bounded graph computation
  -> temporary memory read/write
  -> conditional expert transformation
  -> gated integration into token state
  -> final decoding
```

The hypothesis has five parts:

1. **Separation:** token sequence processing, latent relational computation, and temporary memory can use different state structures.
2. **Conditionality:** not all parameters, experts, memory slots, graph nodes, or reasoning cycles need to activate for every token or request.
3. **Persistence without mutation:** bounded run/session state may preserve temporary bindings without changing released base weights.
4. **Governed cognition:** retrieval, tools, persistent memory, and verification are more controllable when orchestrated by a typed runtime rather than hidden inside generation.
5. **Hardware adaptation:** model size, numerical format, context, active experts, graph cycles, and optional services can vary by measured profile while safety invariants remain fixed.

Every part can fail independently. SARN-Hybrid therefore retains dense, no-memory, no-graph, fixed-depth, and no-tool controls.

## 5. Target System Architecture

### 5.1 Input and Cognitive Flow

```text
Input
  text / image / audio / file / sensor / typed tool result
  -> Aegis input router and content envelope
  -> authorized modality encoder or parser
  -> token and/or modality features
  -> SARN-Hybrid
       embeddings
       RoPE/GQA efficient sequence engine
       optional local-attention or SSM blocks
       dense or sparse-expert FFN capacity
       latent workspace slots
       optional SAE observation/feature bridge
       graph message-passing cycles
       resettable working-memory read/write
       gated writeback and adaptive decoder
  -> candidate text / structured call / action plan
  -> Aegis retrieval, tool, verifier, critic, or simulator hooks
  -> policy, permission, provenance, and safety gates
  -> checked answer / code / authorized action / refusal
```

Retrieval and tool calls can create additional bounded model passes. They are not differentiable model layers and cannot be invoked without Aegis authorization. A verifier finding is evidence, not an automatic truth oracle.

### 5.2 Ownership Boundary

**SARN-Hybrid owns:** learned tensor computation, sequence modeling, optional expert routing, latent workspace updates, bounded neural working-memory state, gated writeback, candidate logits, structured proposals, and instrumentation hooks.

**Aegis owns:** hardware budgets, model selection, context, retrieval, persistent memory, tool permissions and execution, verification, repair limits, policy, user approvals, tracing, artifact integrity, deployment, and safety controls.

The model can propose an effect; only Aegis can authorize and perform one.

### 5.3 Modality Strategy

Text comes first. Future modalities use replaceable adapters and typed content envelopes. Vision-language models, SAM-like segmentation, speech models, and sensor/robotics adapters remain independently testable artifacts. Aegis-X is a unified governed system, not necessarily one universal checkpoint.

## 6. Hardware Profiles

Profile names describe evaluated operating envelopes, not fixed parameter counts or equal capability promises.

### Nano

- smallest evaluated SARN-Dense or SARN-Hybrid-derived model;
- int4/int8 or another measured quantization;
- few or no optional experts and few graph cycles;
- short context and strict generation budget;
- retrieval-heavy when an authorized local index is available;
- deterministic safety and permission gates remain enabled.

### Lite

- small local model on a modern CPU or modest accelerator;
- limited conditional experts when they show net benefit;
- resettable working memory enabled if it passes isolation tests;
- bounded retrieval and inexpensive verification;
- moderate context and concurrency.

### Balanced

- local GPU or capable accelerator;
- accepted graph/workspace path and working memory;
- longer context and larger retrieval budget;
- deterministic verifier plus selected learned checks;
- optional draft decoding and multimodal input adapters.

### Pro

- larger dense or sparse model;
- more accepted experts and bounded reasoning cycles;
- deeper candidate/check/repair budgets;
- multimodal support and stronger retrieval/tool integration;
- detailed telemetry and interpretability sampling.

### Max

- server-grade or distributed research deployment;
- full accepted Aegis-X stack rather than every proposed experiment;
- extensive retrieval, verification, and simulation;
- highest supported context, concurrency, and experiment instrumentation;
- distributed behavior and failure containment evaluated separately.

Stronger hardware can increase optional computation. It cannot loosen permission, privacy, audit, or high-risk approval rules.

## 7. Acceptance Gates

A module must improve at least one predeclared target without crossing declared regression limits in:

- verified task quality;
- structural and distribution-shift robustness;
- latency and time-to-verified-correct-answer;
- peak RAM/VRAM and artifact storage;
- active compute and energy;
- interpretability or causal inspectability;
- safety and misuse resistance;
- deployability and operational complexity.

Minimum evidence includes a matched SARN-Dense or simpler control, several seeds where training variance matters, null and removal ablations, systems measurements on target hardware, held-out tests, failure analysis, and an ADR. Integrated Aegis-X configurations also require interaction ablations: components that win alone may interfere together.

No improvement is accepted when quality rises but latency, memory, energy, safety, interpretability, or maintenance cost exceeds the declared envelope without an explicit profile-specific tradeoff decision.

## 8. Long-Term Success Criteria

Aegis-X is a successful research direction if, under matched and reproducible conditions, it demonstrates meaningful improvement over dense Transformer controls in several of the following:

- verified reasoning or task quality per active compute;
- long-context performance under fixed memory limits;
- temporary binding, overwrite, conflict, and reset behavior;
- safe and correct tool/action proposals under permission constraints;
- lower unsupported-claim rate after retrieval and verification;
- graceful quality-cost scaling across hardware profiles;
- interpretable and causally validated workspace, routing, or memory features;
- robustness under structural and goal-related distribution shift;
- lower total time or energy to a checked correct answer.

Success does not require every proposed mechanism to survive. Aegis-X may ultimately use dense FFNs instead of MoE, omit SSM blocks on some profiles, or keep SAEs entirely offline. The successful architecture is the smallest evidence-supported integrated system, not the longest feature list.

## 9. Governance and Stop Conditions

Architecture ambition remains subordinate to human authority and risk controls. Research pauses or narrows when:

- data provenance or licensing cannot support the run or release;
- a module creates unexplained safety regressions;
- persistent or cross-user state cannot be isolated;
- tool or action boundaries cannot fail closed;
- results cannot be reproduced;
- the resource cost exceeds the approved budget;
- claimed gains disappear under matched controls or contamination review.

Negative results are retained. Aegis-X is a measured breakthrough path: it earns its final shape one module and one system interaction at a time.
