# Research References and Project Interpretation

This is a curated starting bibliography, not a claim that cited results automatically transfer to SARN. Prefer primary papers and official project pages. Architecture choices still require local reproduction under Aegis budgets.

## Language and Sequence Architectures

1. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017). Introduced the Transformer. This is the foundation for the SARN-Dense control.
2. Radford et al., [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (2018). Establishes the early GPT generative-pretraining/fine-tuning approach. It supports the terminology correction that GPT is Transformer-based.
3. Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021). Basis for the reference RoPE implementation; it does not by itself guarantee unlimited context extrapolation.
4. Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (2023). Motivates fewer key/value heads to reduce decode cost while retaining multiple query heads.
5. Gu and Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (2023). Motivates a selective-SSM comparison track, not an assumed replacement for attention.
6. Dao and Gu, [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) (2024). Mamba-2/SSD reference for later sequence experiments and the relation between structured SSMs and attention-like computation.
7. Kosowski et al., [The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain](https://arxiv.org/abs/2509.26507) (2025 preprint). Relevant to locally interacting graph dynamics, sparse positive activations, and inference-time Hebbian state. Its broad biological, interpretability, scaling, and generalization claims need independent evaluation; SARN does not copy or treat them as established facts.

## Sparse Computation, Memory, and Retrieval

8. Fedus, Zoph, and Shazeer, [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) (2021). Reference for sparse expert routing, load balancing, and training instability. It also reinforces that sparse compute introduces routing/communication complexity.
9. Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (2020). Foundation for separating parametric generation from retrieved evidence. Aegis adds provenance, access control, and citation checks as system requirements.
10. Frankle and Carbin, [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) (2018). Motivates controlled pruning/rewinding research; it is not evidence that any deployed SARN model contains a cheaply discoverable universal subnetwork.

## Post-Training and Inference

11. Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (2023). Basis for the planned DPO post-training track. The method learns from preference pairs and does not guarantee factuality or alignment.
12. Leviathan, Kalman, and Matias, [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022). Basis for exact draft-and-verify decoding experiments. The project measures net latency and memory rather than assuming speedup.

## Interpretability and Representation

13. Elhage et al., [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (2022). Frames superposition as sparse features represented in fewer dimensions with interference tradeoffs.
14. Bricken et al., [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (2023). Primary basis for offline SAE experiments and the need to treat extracted features as a learned decomposition.
15. Templeton et al., [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (2024). Relevant to scaling SAE feature analysis and its substantial interpretation challenges.

## Generalization and Optimization

16. Power et al., [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) (2022). Motivates long training curves on controlled algorithmic data; grokking is an observed regime, not a guaranteed phase.
17. Nakkiran et al., [Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/abs/1912.02292) (2019). Motivates tracking performance over capacity, data, and training rather than assuming monotonic improvement.
18. Anil et al., [Scalable Second Order Optimization for Deep Learning](https://arxiv.org/abs/2002.09018) (2020). Reference for scalable Shampoo-style optimizer experiments; memory, communication, and wall-clock costs require direct comparison with AdamW.

## Safety and Alignment Threat Models

19. Hubinger et al., [Risks from Learned Optimization in Advanced Machine Learning Systems](https://arxiv.org/abs/1906.01820) (2019). Source for mesa-optimization and deceptive-alignment threat models. Aegis treats these as unresolved research risks, not detectable labels.
20. Langosco et al., [Goal Misgeneralization in Deep Reinforcement Learning](https://arxiv.org/abs/2105.14111) (2021). Empirical reference for capable policies pursuing unintended goals under distribution shift.
21. Shah et al., [Goal Misgeneralization: Why Correct Specifications Aren't Enough For Correct Goals](https://arxiv.org/abs/2210.01790) (2022). Broader framing for distinguishing specification failures from learned-goal failures.
22. Leike et al., [AI Safety Gridworlds](https://arxiv.org/abs/1711.09883) (2017). Motivates hidden performance functions and controlled tests for reward gaming and other safety problems.
23. Omohundro, [The Basic AI Drives](https://selfawaresystems.com/wp-content/uploads/2008/01/ai_drives_final.pdf) (2008). Historical argument for convergent instrumental strategies. It informs threat modeling but is not an empirical law that every model exhibits these drives.

## Future Modalities

24. Kirillov et al., [Segment Anything](https://arxiv.org/abs/2304.02643) (2023). Relevant if Aegis later integrates promptable segmentation. SAM is a separate vision component, not a layer required inside the text model.

## Important Corrections to Earlier Discussion

- GPT is a decoder-only Transformer family/training approach, not a different alternative algorithm.
- Sparse MoE reduces active computation but does not automatically reduce total parameter storage or edge-device memory.
- Mamba/SSM has favorable sequence-scaling properties, but realized speed and quality depend on kernels, workload, and hardware.
- RAG retrieves evidence; it does not guarantee that sources or generated claims are correct.
- SAEs can expose useful learned features but do not prove monosemanticity, faithful reasoning traces, or safety.
- DPO means Direct Preference Optimization, not “direct preference automation.”
- The relevant inference technique is speculative **decoding**, not speculative encoding.
- Goal generalization, grokking, double descent, superposition, reward hacking, mesa-optimization, deceptive alignment, and mode collapse are phenomena, objectives, or risks—not architecture blocks to stack indiscriminately.
- A general “deceptive alignment monitor” is not an established solved technology. The project uses behavioral evaluations and hard capability controls instead.
- “Belief propagation” is reserved for a defined probabilistic or graphical procedure. Generic latent graph updates are message passing, and their activations are not calibrated beliefs.

## Reference Maintenance

When a reference changes an implementation decision, add an ADR that states the specific claim, reproduction status, target workload, and consequences. Newer papers are not automatically better evidence, and preprints are labeled as such when relevant.
