# Model Card: SARN-Dense Phase 1/2

## Overview

- Model name: SARN-Dense
- Current role: dense Transformer control baseline for Aegis-SARN
- Phase covered: Phase 1 hardened baseline plus Phase 2 reproducible baseline evaluation lab
- Status: implemented as a small CPU-compatible research baseline

SARN-Dense is a decoder-only Transformer used to validate the training, checkpointing, evaluation, generation, benchmarking, registry, and reporting stack. It is the control model for future SARN-Hybrid comparisons, not a claim that hybrid mechanisms are implemented.

It is a baseline/control, not a useful natural-language model.

## Architecture

- Token embedding with optional tied language-model head
- RoPE causal multi-head attention
- RMSNorm
- Gated feed-forward network
- Decoder-only autoregressive language-model objective
- Optional KV cache during generation

The implemented baseline intentionally excludes SARN-Hybrid modules, MoE, graph workspace, resettable working memory, SSM/Mamba, retrieval, tools, VLM, SAM, LAM, multimodal modules, and advanced safety systems.

## Parameter Count Range

The default micro configuration is small enough for CPU tests. Parameter count depends on `ModelConfig`, especially `vocab_size`, `d_model`, `n_layers`, `n_heads`, `ffn_hidden_dim`, and whether embeddings are tied. Baseline reports record the exact parameter count for each run.

## Intended Use

- Reproducible smoke training and checkpoint resume tests
- Toy loss, perplexity, and token-accuracy evaluation
- CPU generation benchmark and memory estimates
- Registry and report generation for future baseline comparison
- Control baseline for later SARN-Hybrid experiments

## Non-Intended Use

- Natural-language assistant behavior
- Factual answering
- Safety-critical decisions
- Production deployment
- Claims about SARN-Hybrid quality or efficiency
- Claims about reasoning, memory, retrieval, tool use, multimodal ability, or advanced safety

## Training Data

Phase 1/2 smoke training uses generated toy token batches such as `toy/repeated_pattern`. These datasets are deterministic fixtures, not real corpora.

## Evaluation Data

Phase 2 toy evaluation uses generated toy validation batches. The metrics are useful for regression testing and reproducibility checks only. They do not measure natural-language quality.

## Limitations

- Toy datasets are low-entropy generated patterns.
- The model is tiny and not trained on a governed language corpus.
- Perplexity and token accuracy are meaningful only for the generated toy task distribution.
- Generation samples are toy-byte/token outputs, not useful prose.
- CPU benchmark numbers are local measurements and should not be generalized across machines.

## Safety Notes

SARN-Dense has no retrieval, tools, persistent memory, multimodal adapters, or autonomous action path. The Aegis controller records traces and budgets for the implemented runtime path, but advanced safety systems remain future work. Outputs from the toy model should be treated as test artifacts.

## Baseline Role

SARN-Dense remains the control baseline. Future SARN-Hybrid work must compare against it with matched metrics, seeds, artifacts, and system measurements. Hybrid ideas remain documented research targets until separately implemented and accepted behind evidence gates.
