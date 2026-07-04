# Interpretability and Representation Research

## 1. Purpose

Interpretability work has three goals: understand failures, test SARN's representation hypotheses, and discover candidate monitoring signals. It does not promise a transparent mind or a general safety detector.

## 2. Representation Questions

- Do latent workspace slots acquire stable functions across examples and seeds?
- Are graph edges causally used or merely correlated with outputs?
- Does sparse routing create specialization, collapse, or arbitrary partitions?
- How does temporary memory store and overwrite associations?
- Which features change during delayed generalization?
- What information is lost under quantization, pruning, or distillation?

## 3. Instrumentation

Models expose opt-in hooks for token embeddings, attention outputs, FFN activations, router logits and assignments, workspace states/edges, memory reads/writes, residual streams, and output logits. Hooks are disabled or sampled in ordinary private serving because activations can contain sensitive information and incur substantial cost.

Instrumentation version and tensor semantics are part of the model manifest.

## 4. Sparse Autoencoders

An SAE learns a reconstruction of an activation `x` from sparse features `f`:

```text
f = sparse_activation(W_enc x + b_enc)
x_hat = W_dec f + b_dec
loss = reconstruction_error(x, x_hat) + sparsity_penalty(f)
```

The project records layer and hook point, activation sample, dictionary expansion, sparsity method and coefficient, reconstruction metrics, feature density, dead features, decoder norms, and training stability.

Feature interpretation uses top-activating examples, counterexamples, automated labels, independent human review, and cross-dataset analysis. Labels remain hypotheses.

## 5. Superposition and Polysemanticity

Superposition offers an account of models representing more sparse features than available dimensions, with interference. Polysemantic neurons are one possible symptom. SARN does not aim to prohibit superposition globally; doing so may reduce capacity. Instead it measures feature interference and tests whether workspace sparsity or routing changes it.

SAEs provide one learned dictionary and may miss, split, merge, or fabricate apparent features due to training choices. Reconstruction quality and downstream faithfulness are always reported.

## 6. Causal Validation

Correlational probes are followed, where feasible, by:

- activation patching between matched examples;
- ablation and mean replacement;
- feature steering at multiple strengths;
- edge deletion or shuffling;
- router intervention;
- memory-write suppression;
- counterfactual inputs and causal mediation analysis.

An interpretation is stronger if a targeted intervention changes the predicted behavior without broad unrelated damage and replicates across examples. Steering can create off-distribution artifacts, so results require controls.

## 7. Concept-Graph Validation

To call a node a candidate concept, require selectivity across a defined dataset, robustness to paraphrase, sensitivity to counterfactual changes, causal relevance to output, stability across checkpoints or an explicit alignment mapping, and known failure cases.

Graph visualizations show activation magnitude, edge type, cycle, and uncertainty. They must never be presented as literal natural-language chain of thought or calibrated confidence without validation.

## 8. Router and Expert Analysis

Track utilization, entropy, mutual information with domains/tasks, co-routing, overflow, expert ablation, expert swapping, and stability across seeds. Human names for experts follow analysis rather than precede it. High specialization is not automatically good if it harms robustness or load balance.

## 9. Generalization Studies

For grokking experiments, save checkpoints across memorization and generalization phases. Compare weight norms, margins, representations, SAE features, workspace structure, and causal circuits over time. Because many runs may not grok, selection criteria and checkpoint cadence are fixed in advance.

The double-descent study similarly plots outcomes rather than using the phrase as a post-hoc explanation for any regression.

## 10. Safety Use

Interpretability signals may rank examples for review or support research hypotheses. They cannot alone authorize tools, persistence, release, or denial of user access. Any monitor reports calibrated performance, false positives/negatives, distribution limits, adaptive robustness, and what happens when it fails.

## 11. Artifact and Privacy Policy

Activation datasets, SAE weights, feature dashboards, and labels are versioned artifacts linked to a base checkpoint and data manifest. Before sharing, assess whether examples or features leak copyrighted, private, or memorized training content. Interpretability access follows the same least-privilege principle as model access.
