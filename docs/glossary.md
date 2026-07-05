# Glossary

## Project Terms

- **Aegis**: the framework and control plane surrounding models.
- **SARN**: Sparse Adaptive Reasoning Network, the model family and experimental architecture.
- **SARN-Dense**: the decoder-only Transformer baseline and scientific control for architecture comparisons.
- **SARN-Hybrid**: the target evidence-gated model architecture combining efficient attention, latent graph workspace, resettable working memory, gated writeback, and optional SSM/MoE accelerators.
- **profile**: a named set of resource limits and enabled capabilities for a hardware class.
- **run**: one bounded execution of a request, including retrieval, model calls, checks, and tools.
- **artifact**: a content-addressed model, tokenizer, dataset manifest, evaluation result, or release package.
- **accepted module**: an experimental component that passed its predeclared evidence gate.

## Model Terms

- **causal language model**: predicts a token using only preceding tokens in its context.
- **Transformer**: sequence architecture built around attention and position-wise feed-forward layers.
- **GPT-style**: decoder-only causal Transformer training and generation; not a separate alternative to Transformers.
- **RoPE**: rotary position embedding applied to attention queries and keys.
- **GQA**: grouped-query attention, where multiple query heads share fewer key/value heads.
- **SSM**: state-space model; in this project, an optional sequence-block family evaluated against attention.
- **MoE**: mixture of experts with learned routing. Sparse activation can reduce active compute while total parameters still consume storage.
- **latent workspace**: a fixed-size internal state updated for a bounded number of cycles.
- **concept node**: a learned latent slot. The name does not assert that it maps cleanly to a human concept.
- **belief**: reserved for calibrated probabilities. Raw graph activations must be called activations, scores, or states.
- **working memory**: resettable, run- or session-scoped state used in addition to the token context.
- **Hebbian update**: a local association rule based on co-activation; proposed as an experiment, not assumed to be stable memory.

## Framework Terms

- **retrieval**: selecting external evidence for a run without changing model parameters.
- **persistent memory**: policy-controlled information stored across sessions.
- **tool**: a typed external capability that may observe or change state.
- **verifier**: a checker that emits findings and evidence. It may be deterministic or learned and has a defined assurance level.
- **repair loop**: bounded regeneration using verifier findings.
- **provenance**: the source, lineage, time, transformation history, and trust metadata of information.

## Training and Evaluation Terms

- **pretraining**: training a model on a broad predictive objective, normally next-token prediction here.
- **SFT**: supervised fine-tuning on input/desired-output examples.
- **DPO**: Direct Preference Optimization using preferred/rejected response pairs; it optimizes preferences, not factual truth by itself.
- **distillation**: training a student to reproduce selected teacher behavior or distributions.
- **quantization**: representing weights and/or activations at lower numerical precision.
- **ablation**: a controlled removal or change used to estimate a component's contribution.
- **matched compute**: comparison under a declared equivalent budget, such as active FLOPs or measured latency.
- **grokking**: delayed generalization after an extended period of fitting or overfitting in some algorithmic settings; a phenomenon to study, not a training stage.
- **double descent**: a family of observed non-monotonic error curves as capacity or training varies; a measurement concern, not a module.
- **mode collapse**: primarily a generative-adversarial-network failure where outputs lose diversity. For language models, use precise terms such as repetition, low diversity, or expert collapse unless the GAN meaning applies.

## Safety Terms

- **reward hacking**: achieving a measured objective through behavior that violates the intended objective.
- **goal misgeneralization**: competent behavior under shift that pursues an unintended learned goal.
- **mesa-optimization**: the hypothesis that training produces a learned system that itself performs optimization for an internal objective.
- **deceptive alignment**: a theoretical failure mode in which a strategically aware learned optimizer behaves aligned instrumentally. Aegis evaluates concerning behavior but does not claim access to hidden intent.
- **instrumental convergence**: the hypothesis that varied objectives can produce similar instrumental strategies such as resource acquisition. It informs threat modeling, not a neuron-level detector.
