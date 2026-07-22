# Model Card: SARN-Dense Phase 1-6

## Overview

- Model name: SARN-Dense
- Current role: dense Transformer control baseline for Aegis-SARN
- Phase covered: Phase 1 hardened baseline through the Phase 6 graph prototype
- Status: implemented as a small CPU-compatible research baseline

SARN-Dense is a decoder-only Transformer used to validate the training, checkpointing, evaluation, generation, benchmarking, registry, and reporting stack. It is the control model for future SARN-Hybrid comparisons, not a claim that hybrid mechanisms are implemented.

It is a baseline/control, not a useful natural-language model.

## Architecture

- Token embedding with optional tied language-model head
- RoPE causal multi-head attention by default, with experimental configurable grouped-query attention
- RMSNorm
- Gated feed-forward network
- Decoder-only autoregressive language-model objective
- Optional KV cache during generation
- Optional bounded latent workspace, disabled by default, with transient cache state and gated/no-writeback controls
- Optional bounded graph message passing over enabled workspace slots, disabled by default

The implementation intentionally excludes a full SARN-Hybrid path, resettable working memory, MoE, SSM/Mamba, retrieval, tools, VLM, SAM, LAM, multimodal modules, and advanced safety systems.

## Parameter Count Range

The default micro configuration is small enough for CPU tests. Parameter count depends on `ModelConfig`, especially `vocab_size`, `d_model`, `n_layers`, `n_heads`, `ffn_hidden_dim`, and whether embeddings are tied. Baseline reports record the exact parameter count for each run.

## Intended Use

- Reproducible smoke training and checkpoint resume tests
- Toy loss, perplexity, and token-accuracy evaluation
- CPU generation benchmark and memory estimates
- Registry and report generation for future baseline comparison
- Tiny-size scaling sweeps, task-level toy evaluation, comparison reports, and experiment quality gates
- Matched MHA/GQA sweeps with KV-head and approximate cache-memory reporting
- Matched disabled/null/enabled latent-workspace sweeps and mechanical diagnostics
- Matched dense/workspace/null-edge/frozen-edge/learned-edge graph sweeps, structural fixtures, and mechanical diagnostics
- Control baseline for later SARN-Hybrid experiments

## Non-Intended Use

- Natural-language assistant behavior
- Factual answering
- Safety-critical decisions
- Production deployment
- Claims about SARN-Hybrid quality or efficiency
- Claims about reasoning, memory, retrieval, tool use, multimodal ability, or advanced safety

## Training Data

Phase 1-6 training and evaluation use generated toy token batches. These datasets are deterministic fixtures, not real corpora.

## Evaluation Data

Phase 2-6 evaluation uses generated toy validation batches and deterministic task variants. Phase 6 adds relation-chain, route-propagation, slot-binding, and tiny length-extrapolation fixtures. The metrics are useful for regression testing, controlled comparisons, and reproducibility checks only. They do not measure natural-language quality or establish reasoning.

## Limitations

- Toy datasets are low-entropy generated patterns.
- The model is tiny and not trained on a governed language corpus.
- Perplexity and token accuracy are meaningful only for the generated toy task distribution.
- Generation samples are toy-byte/token outputs, not useful prose.
- CPU benchmark numbers are local measurements and should not be generalized across machines.
- Experimental GQA reduces stored KV heads by construction, but the toy runs do not establish useful language quality or a generally faster implementation.
- Latent workspace slots are bounded tensor states, not human-like concepts or persistent memory.
- Graph cycles are small learned tensor transformations, not resettable memory, formal logic, proof of human-like reasoning, or an accepted quality improvement.

## Safety Notes

SARN-Dense has no retrieval, tools, persistent memory, multimodal adapters, or autonomous action path. The Aegis controller records traces and budgets for the implemented runtime path, but advanced safety systems remain future work. Outputs from the toy model should be treated as test artifacts.

## Baseline Role

SARN-Dense remains the control baseline, MHA remains its default attention setting, and workspace/graph modules remain disabled by default. Experimental GQA, latent slots, and graph cycles are research configurations, not a SARN-Hybrid model path. Future work must compare against the control with matched metrics, seeds, artifacts, and system measurements.
