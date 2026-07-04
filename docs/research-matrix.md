# Research Traceability Matrix

This matrix prevents ideas from disappearing or quietly becoming commitments. “Type” distinguishes a component to build from a technique to test, a phenomenon to measure, and a risk to defend against.

| Idea | Type | Place in Aegis-SARN | Earliest phase | Primary evidence required | Default status |
|---|---|---|---:|---|---|
| GPT-style causal modeling | baseline | SARN-Dense next-token model | 1 | loss, generation, correctness invariants | accepted baseline |
| Transformer | baseline architecture | attention + FFN decoder blocks | 1 | reproducible control results | accepted baseline |
| RoPE | model technique | attention query/key position encoding | 1 | implementation tests; context behavior | accepted baseline |
| grouped-query attention | model experiment | lower-KV-head attention option | 4 | cache, latency, quality Pareto | proposed |
| Mamba/selective SSM | model experiment | attention alternative/hybrid | 7 | workload-specific quality, memory, speed | proposed |
| MoE/Switch-style routing | model experiment | selected FFN replacements | 8 | load balance, quality per active cost, total memory | proposed, later-scale |
| latent concept model | research hypothesis | latent workspace slots | 5 | structural generalization and causal semantics | proposed; neutral slot naming |
| sparse concept graph | research hypothesis | message passing among workspace slots | 5 | equal-compute/null-edge ablations | proposed |
| top-k competition/inhibition | model experiment | workspace/expert sparsity | 5/8 | stability, sparsity, quality, gradients | proposed |
| belief propagation | terminology/method | only if a defined probabilistic graphical model exists | unscheduled | calibration and correct probabilistic semantics | not currently claimed |
| Hebbian/fast-weight memory | model experiment | resettable neural working memory | 6 | capacity, conflict, reset, leakage, baseline gain | proposed |
| RAG | framework technique | retrieval service + typed evidence | 9 | retrieval, answer, citation, injection metrics | planned |
| persistent user memory | framework component | governed cross-session records | 9 | authorization, correction, deletion, isolation | planned |
| verifier/critic | framework component | assurance-class checker registry | 10 | false pass/reject, coverage, repair benefit | planned |
| simulator | tool/verifier specialization | domain-specific deterministic or learned checker | 10+ | domain validity and containment | future |
| adaptive reasoning depth | model/runtime experiment | bounded workspace/search budget | 5+ | quality per total compute and halting calibration | proposed |
| hardware profiles | framework component | model/precision/context planner | 12 | on-device fit, latency, quality-floor tests | planned |
| quantization | deployment technique | distinct model artifacts | 12/13 | memory/latency/quality by task | planned |
| distillation | training technique | deployable student variants | 13 | student Pareto improvement and safety regression | proposed |
| speculative decoding | inference technique | compatible draft + target backend path | 13 | exactness, acceptance, net latency and memory | proposed |
| superposition | representation phenomenon | measured in SARN activations | 14 | interference and causal feature studies | research topic |
| sparse autoencoders | interpretability technique | offline feature dictionaries first | 14 | reconstruction, sparsity, stability, interventions | proposed offline tool |
| polysemanticity | interpretation concern | representation analysis | 14 | feature selectivity and causal tests | research topic |
| grokking | training phenomenon | synthetic generalization study | 3/14 | predeclared delayed-generalization curves | observe, do not assume |
| double descent | scaling phenomenon | scaling observatory | 3 | error versus capacity/data/training plots | observe, do not assume |
| lottery-ticket hypothesis | compression research | iterative pruning/rewinding | 13 | retraining and deployment Pareto evidence | proposed experiment |
| Shampoo/second-order optimization | optimizer experiment | larger-run training lab | 13+ | wall-clock/convergence versus AdamW, memory cost | deferred |
| DPO | post-training technique | preference tuning after SFT | 11 | held-out preferences, shortcuts, capability/safety | planned experiment |
| goal generalization | evaluation objective | OOD instruction/constraint adherence | 3 onward | structural shifts and hidden constraints | required evaluation |
| goal misgeneralization | safety failure | threat/evaluation suite | 3 onward | proxy-breaking distribution shifts | required risk test |
| reward hacking | safety failure | hidden-performance and tool-task tests | 3 onward | reward/performance gap | required risk test |
| instrumental convergence | threat model | resource/autonomy capability limits | all | observable behavior tests + hard permissions | threat-model input |
| mesa-optimization | threat model | research evaluations, not a module | later capable systems | behavioral/interpretability evidence with caveats | unresolved risk |
| deceptive alignment | threat model | oversight-shift tests, hard containment | later capable systems | no universal detector claim | unresolved risk |
| mode collapse | mostly GAN phenomenon | use precise repetition/diversity/expert-collapse metrics | 3 onward | metric-specific evidence | terminology correction |
| Dragon Hatchling/BDH | external research architecture | inspiration/comparator for graph + Hebbian work | 5–7 | independent reproduction and matched comparison | reference, not dependency |
| LLM | model category | SARN text generator or hosted models | 1 onward | model card and task metrics | core category |
| SLM | deployment category | smaller checkpoint/profile, not separate intelligence type | 12/13 | per-profile quality/cost | planned family tier |
| MLM | training/model category | optional encoder for retrieval/classification | 9+ | task need versus simpler embedding model | future adapter |
| VLM | multimodal category | later vision-language adapter/model | 15 | multimodal data, metrics, safety case | future |
| SAM/segmentation | dedicated external model | Aegis vision tool/adapter | 15 | segmentation task and license/runtime fit | future |
| LAM/action model | model/system category | typed plan proposer behind Aegis policy | 15 | action correctness and containment | future high-risk |

## Interpretation Rule

Only rows marked accepted baseline are default model commitments today. “Planned” means the architecture reserves a clean interface and roadmap gate. “Proposed,” “future,” and “research topic” do not imply implementation or expected success.

When an experiment concludes, update this matrix and link its decision record. Do not erase rejected rows; change their status and preserve the evidence.
