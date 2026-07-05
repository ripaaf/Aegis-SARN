# SARN Model Family Specification

## 1. Research Position

SARN is an evidence-gated architecture program with a declared destination. **SARN-Dense** is the baseline and control. **SARN-Hybrid** is the target architecture: one coherent learned-computation spine assembled from efficient sequence processing, sparse conditional capacity, latent graph computation, and resettable working memory.

The canonical first model is a small GPT-style model: a decoder-only Transformer trained causally to predict the next token. GPT models are Transformer models; “GPT versus Transformer” is a category mistake.

SARN-Hybrid is not predeclared successful or frozen in every detail. The target supplies direction; matched experiments determine which proposed mechanisms survive. A rejected mechanism changes the final hybrid design without changing the research thesis.

## 2. SARN-Hybrid Target Algorithm

### 2.1 Canonical Forward Path

```text
token IDs and/or authorized modality features
 -> token and/or modality embeddings
 -> hybrid sequence engine
      MHA/GQA attention with RoPE on queries/keys
      optional local/sliding attention
      optional selective SSM/Mamba-style blocks
 -> conditional-capacity stage
      dense FFN control or sparse expert FFN
 -> latent workspace slots
 -> optional sparse-autoencoder feature extraction/observation
 -> sparse graph message-passing cycles
 -> resettable fast/working-memory read and update
 -> gated writeback to token states
 -> verifier and instrumentation hooks
 -> final normalization
 -> tied language-model head
 -> next-token logits
```

At the system boundary, Aegis can attach retrieval evidence before a model call and consume verifier hooks after candidate generation. Retrieval, persistent memory, tool execution, policy, and the decision to repair remain framework operations; they are not silently folded into model weights.

### 2.2 Component Responsibilities

- **RoPE and GQA:** RoPE supplies relative-position-sensitive query/key transformations; GQA reduces KV-cache heads and memory traffic compared with full multi-head key/value projections. Neither guarantees unlimited context.
- **Local attention:** optionally limits attention span or supplies structured locality when it improves the target workload.
- **Selective SSM blocks:** provide an optional long-context/state-propagation path whose real kernel speed and quality must be measured against attention.
- **Dense or sparse-expert FFN:** retains a dense control while testing whether conditional routing can activate useful capacity without proportional active compute. Total storage and communication remain part of the cost.
- **Latent workspace:** provides persistent learned slots for bounded internal state. Slots are not called human concepts until causal evidence supports that interpretation.
- **Sparse feature extraction:** SAEs initially observe frozen activations for interpretability. An online feature bridge is optional, read-only by default, and requires a separate causal acceptance test before influencing computation.
- **Graph message passing:** updates relations among workspace slots for a bounded number of cycles. It is an explicit relational computation mechanism, not proof of logical reasoning.
- **Resettable working memory:** stores temporary associations during a run or session without mutating released base weights; capacity, conflict, reset, isolation, and poisoning are tested.
- **Gated writeback:** controls how workspace and memory state re-enters token representations, limiting unstable or irrelevant latent state from dominating generation.
- **Verifier hooks:** expose candidates and typed instrumentation to Aegis checkers. The model may provide signals or self-critique, but it cannot declare itself correct or authorize an effect.
- **Final normalization and tied head:** convert the final token representation into vocabulary logits using a reproducible baseline-compatible output path.

### 2.3 Architectural Thesis

SARN-Hybrid tests whether useful capability can scale across several controlled dimensions:

- **sequence depth and attention** for content-addressed token interaction;
- **recurrent state-space computation** for efficient sequence-state propagation where it wins;
- **conditional parameters** for capacity without activating every expert on every token;
- **latent workspace cycles** for bounded computation over persistent internal slots;
- **graph structure** for sparse, inspectable message routes among slots;
- **working-memory state** for temporary bindings without modifying base weights;
- **adaptive runtime budgets** for changing active computation across hardware tiers.

The intended novelty is the integrated computation path and its learned interfaces, not the claim that any one ingredient was invented here. SARN-Hybrid is not an acronym checklist: it is one end-to-end architecture hypothesis whose components must work together under a shared training and evaluation contract.

### 2.4 Module Contracts

Every hybrid stage preserves a typed tensor contract and can be replaced by a control:

| Stage | Input/Output responsibility | Required control |
|---|---|---|
| sequence engine | contextual token states to contextual token states | attention-only SARN-Dense block |
| conditional capacity | token states to transformed token states | dense FFN at matched active cost |
| workspace router | token states to initial/update signals for `K` slots | pooled dense state or no workspace |
| sparse feature observer | selected activations to sparse diagnostic features | observer disabled and reconstruction-only control |
| graph cycles | workspace state to updated workspace state | independent slots, shuffled/frozen edges |
| working memory | bounded read/update of run-scoped state | no memory and explicit key/value memory |
| gated writeback | workspace/memory state to token-state residual | no writeback and equal-parameter MLP |
| verifier hooks | candidate/telemetry to typed Aegis interface | candidate-only interface with no learned signals |

The model configuration declares block order, shared versus unshared cycle weights, slot count, graph sparsity, memory capacity, cycle count, expert routing, and all gates. The resolved configuration and feature flags are part of the artifact identity.

### 2.5 Core Versus Optional Mechanisms

The target's core research spine is efficient RoPE/GQA attention, latent workspace slots, graph message passing, resettable working memory, and gated token-state writeback. Selective SSM blocks, local/linear attention variants, and MoE routing are optional accelerators: they enter the integrated architecture only when they improve the relevant quality-cost frontier. A dense FFN and full-attention path always remain available as controls.

### 2.6 Integration Rule

Individual wins are not enough to declare SARN-Hybrid successful. The integrated path must be trained end to end, compared with a dense model at matched parameters and active compute, and ablated stage by stage. Interactions are reported explicitly; two helpful modules can interfere when combined.

## 3. Baseline: SARN-Dense

### 3.1 Forward Path

```text
token IDs
 -> token embeddings
 -> repeated pre-normalized decoder blocks
      RMSNorm
      causal self-attention with RoPE
      residual connection
      RMSNorm
      gated feed-forward network
      residual connection
 -> final RMSNorm
 -> tied token projection
 -> next-token logits
```

The initial implementation should support standard multi-head attention. GQA is added as a configuration and compared rather than silently made mandatory.

### 3.2 Suggested Research Sizes

These are experiment classes, not capability promises:

| Class | Approximate parameters | Purpose |
|---|---:|---|
| micro | 1–5M | unit tests, overfit tests, synthetic tasks |
| tiny | 10–50M | baseline language and reasoning experiments |
| small | 50–300M | only after data/runtime pipeline is stable |

Exact dimensions are generated from configuration and recorded. Comparisons match parameter count, active FLOPs, tokens trained, and wall-clock cost where possible.

### 3.3 Baseline Requirements

- causal masking with tests that future tokens cannot leak;
- padding and packed-sequence behavior defined explicitly;
- deterministic initialization and seeded data order when supported;
- tokenizer/config/checkpoint compatibility checks;
- tied versus untied embeddings configurable and reported;
- loss in full enough precision for numerical stability;
- KV-cache and no-cache generation equivalence within tolerance;
- readable reference implementation before fused kernels.

## 4. Attention Efficiency Track

### RoPE

Rotary position embedding is the baseline positional method. Long-context extrapolation is not assumed. Any scaling method must be evaluated for original-context regression, long-context retrieval, numerical stability, and fine-tuning requirements.

### Grouped-Query Attention

GQA uses more query heads than key/value heads, reducing KV-cache size and decode memory traffic. It may preserve quality better than a single key/value head, but the tradeoff is empirical. Record query heads, key/value heads, cache bytes per token, prefill speed, decode speed, and quality.

### Local or Sliding Attention

This is a later option for bounded attention cost. It requires deliberate global-token or memory mechanisms and should be tested on tasks needing distant dependencies.

## 5. Efficient Sequence Track

Mamba-style selective state-space blocks are an **alternative experiment**, not an automatic upgrade. Candidate variants are:

- attention-only control;
- SSM-only model at matched scale;
- interleaved attention/SSM blocks;
- parallel attention and SSM branches with a learned merge.

Measure training throughput, prefill and recurrent decode behavior, memory, long-context recall, language loss, and hardware kernel maturity. A theoretically favorable complexity does not guarantee faster execution on target hardware.

## 6. Sparse Expert Track

The proposed MoE replacement applies to selected feed-forward sublayers:

```text
hidden state -> router -> top-k experts -> weighted combine -> residual
```

Required instrumentation includes expert assignment histograms, token drop or overflow rate, routing entropy, per-expert gradient norms, load-balance auxiliary loss, router z-loss if used, communication cost, total and active parameters, and expert collapse indicators.

MoE is not the first edge optimization: all expert weights still require storage and often memory residency, while small batches and CPUs can suffer routing overhead. MoE enters only after dense baselines and primarily targets accelerator research until measurements show otherwise.

Expert names such as “math” or “code” are not hard-coded claims. Specialization must be demonstrated from behavior and interventions.

## 7. Latent Workspace and Graph Track

### Hypothesis

A small persistent latent workspace updated for several internal cycles may improve systematic multi-step tasks or temporary state tracking compared with spending the same compute only on deeper token transformations.

### Proposed Mechanism

Let token states be `H` and `K` workspace nodes be `Z`. A cycle can perform:

```text
Z0 = top-k or soft routing from pooled/token states H
Z(t+1) = normalize(Z(t) + message_pass(Z(t), edges) + read_tokens(Z(t), H))
H' = H + gated_writeback(H, Z(T))
```

Edges may be dense at tiny scale, sparse top-k, or dynamically generated. The implementation must define whether routing is differentiable, how inactive nodes behave, whether cycles share weights, and how gradients flow through selection.

### Naming Discipline

Nodes are latent slots until interpretability evidence supports more. Scores are not “beliefs” unless calibrated. Message passing is not evidence of logical reasoning. Top-k competition can improve sparsity but can also destroy gradient flow or useful uncertainty.

### Required Ablations

- equal-compute deeper Transformer;
- equal-parameter dense MLP replacement;
- workspace with no edges;
- shuffled or frozen edges;
- one versus multiple cycles;
- soft versus hard routing;
- no writeback;
- performance by reasoning-chain length and distribution shift.

The module is accepted only if gains survive several seeds and cannot be explained by additional parameters, training tokens, or test contamination.

## 8. Working-Memory Track

Working memory is separate from persistent user memory and from the Transformer KV cache. Candidate mechanisms include:

- explicit key/value slots with learned read/write gates;
- differentiable fast-weight matrix;
- graph-edge state scoped to a run;
- non-differentiable associative cache managed by Aegis.

A Hebbian-style candidate may update a temporary matrix:

```text
M(t+1) = decay * M(t) + learning_rate * outer(key_t, value_t)
```

This simple rule can saturate, overwrite, leak information, or destabilize scale. Experiments must include normalization, bounded updates, reset behavior, distractors, conflicting facts, deletion, adversarial writes, and batch isolation. Serving never persists this state automatically and never applies it to base weights.

## 9. Sparse Autoencoder Track

SAEs are initially offline interpretability artifacts trained on frozen activations and are not assumed to solve polysemanticity. The target diagram permits an optional instrumentation path that extracts features without writing back into model state; it is disabled in the default computational path. If SAE features later influence workspace or token computation, that becomes a separate causal experiment measuring reconstruction loss, sparsity, dead features, downstream quality, feature stability, and intervention effects.

Superposition is treated as a representation phenomenon, not a switch to enable. Compact models may use superposed features; SAEs may offer one approximate decomposition.

## 10. Adaptive Computation

Reasoning depth may later vary by input using halting scores or a framework-set cycle budget. Safe rules are:

- a hard maximum is enforced by Aegis;
- the model cannot purchase more tool or wall-time budget;
- halting is trained and calibrated against actual task benefit;
- easy-task regressions and difficult-task underthinking are reported;
- comparisons use total compute, not only final accuracy.

Test-time search, self-consistency, and verifier repair are framework strategies and must not be conflated with one model forward pass.

## 11. Alignment and Post-Training

After pretraining, compatible checkpoints may undergo supervised instruction tuning and then preference tuning such as DPO. Preference data must encode helpfulness, honesty, uncertainty, instruction hierarchy, privacy, tool boundaries, and safe refusal without rewarding verbosity or superficial agreement.

DPO optimizes preferences present in the dataset. It cannot guarantee factuality, eliminate reward hacking, or establish safe goals. Held-out capability and safety evaluations are required after every post-training stage.

## 12. Speculative Decoding

Speculative decoding is a runtime optimization using a cheaper draft model and a target model that accepts or rejects draft tokens. It is not speculative “encoding” and does not improve the target distribution by itself. Evaluate acceptance rate, end-to-end latency, batch size, memory cost of both models, tokenizer compatibility, and exactness of the chosen algorithm. It is enabled only when faster on the actual profile.

## 13. Compression Track

Compression candidates are weight-only and weight/activation quantization, distillation, structured pruning, and research into iterative magnitude pruning inspired by lottery-ticket work. Every compressed artifact receives its own evaluation card. The project never reports size reduction without task deltas, peak memory, real latency, and calibration method.

Finding a sparse subnetwork does not imply it trains cheaply from scratch or is universally transferable. “Lottery ticket” is a hypothesis and experiment family, not a deployment guarantee.

## 14. Multimodal and Action Extensions

Text is first. Later:

- a vision-language adapter can project a frozen or jointly trained vision encoder into the language model;
- segmentation can call a dedicated segmentation model through Aegis;
- masked-language encoders may support retrieval or classification but are not required inside the causal decoder;
- an action model emits typed plans, never direct side effects;
- speech uses explicit encoder/decoder adapters.

Each modality needs its own data, metrics, threat model, artifact card, and hardware budget. “LLM + MLM + VLM + SAM + LAM + SLM” is a system inventory, not a scientifically meaningful architecture by itself.

## 15. Variant Naming

Use compositional identifiers, for example:

```text
sarn-dense-25m-rope-mha
sarn-dense-25m-rope-gqa
sarn-hybrid-25m-gqa-ws32-graph4-mem16
sarn-workspace-25m-k32-cycles4
sarn-ssmhybrid-25m-a2s2
sarn-moe-100m-active25m-top2
```

The human-readable name never replaces the full configuration digest.

## 16. Acceptance Rule

SARN-Hybrid is not a random collection of buzzwords. It is one integrated algorithm hypothesis with separately configurable and ablatable mechanisms. No component becomes the default because it is novel. Acceptance requires a pre-registered hypothesis, at least one matched baseline, multi-seed results, systems measurements, ablations, robustness and safety tests, documented failure cases, and a decision-log entry. A negative result can complete the research milestone without entering the production model.
