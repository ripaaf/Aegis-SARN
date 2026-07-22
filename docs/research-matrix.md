# Research Traceability Matrix

This matrix prevents ideas from disappearing or quietly becoming commitments. “Type” says what an idea is; “Role in SARN-Hybrid” uses a controlled category to say why it exists in the program; “Default status” records present evidence. Role and status are independent—a core-spine mechanism can still be proposed and ultimately rejected.

| Idea | Type | Role in SARN-Hybrid | Place in Aegis-SARN | Earliest phase | Primary evidence required | Default status |
|---|---|---|---|---:|---|---|
| GPT-style causal modeling | baseline | accepted baseline | SARN-Dense next-token model | 1 | loss, generation, correctness invariants | accepted baseline |
| Transformer | baseline architecture | accepted baseline | attention + FFN decoder blocks | 1 | reproducible control results | accepted baseline |
| RoPE | model technique | core spine | attention query/key position encoding | 1 | implementation tests; context behavior | accepted baseline |
| grouped-query attention | model experiment | core spine | efficient attention foundation | 4 | cache, latency, quality Pareto | experimental; MHA default |
| local attention | model experiment | optional accelerator | bounded/local sequence interactions | 9 | long-context quality, latency, lost-global-information tests | proposed |
| linear attention | model experiment | optional accelerator | approximate/structured attention alternative | 9 | quality, stability, realized hardware cost | future experiment |
| Mamba/selective SSM | model experiment | optional accelerator | attention alternative/hybrid | 9 | workload-specific quality, memory, speed | proposed |
| MoE/Switch-style routing | model experiment | optional accelerator | selected FFN replacements | 8 | load balance, quality per active cost, total memory | tiny local prototype; disabled by default |
| latent workspace | research hypothesis | core spine | bounded latent workspace slots | 5 | structural generalization and causal semantics | experimental; neutral slot naming |
| latent-slot graph | research hypothesis | core spine | message passing among workspace slots | 6 | equal-compute/null-edge/frozen-edge ablations | experimental; disabled by default |
| gated writeback | model mechanism | core spine | controlled workspace residual into token states | 5–6 | stability, gate use, no-writeback and MLP controls | experimental |
| top-k competition/inhibition | model experiment | core spine | workspace/expert sparsity | 5/8 | stability, sparsity, quality, gradients | proposed |
| belief propagation | terminology/method | research phenomenon | only if a defined probabilistic graphical model exists | unscheduled | calibration and correct probabilistic semantics | not currently claimed |
| resettable slot/fast-weight memory | model experiment | core spine | bounded temporary neural working memory | 7 | capacity, conflict, reset, leakage, baseline gain | experimental; disabled by default |
| verifier hooks | model/runtime interface | core spine | typed candidate and instrumentation interface | 10 | hook faithfulness, schema stability, no authorization bypass | planned |
| RAG | framework technique | framework component | retrieval service + typed evidence | 9 | retrieval, answer, citation, injection metrics | planned |
| persistent user memory | framework component | framework component | governed cross-session records | 9 | authorization, correction, deletion, isolation | planned |
| verifier/critic | framework component | framework component | assurance-class checker registry | 10 | false pass/reject, coverage, repair benefit | planned |
| simulator | tool/verifier specialization | framework component | domain-specific deterministic or learned checker | 10+ | domain validity and containment | future |
| adaptive reasoning depth | model/runtime experiment | core spine | bounded workspace/search budget | 4+ | quality per total compute and halting calibration | proposed |
| hardware-adaptive runtime | framework component | core spine | model/precision/context/cycle planner | 13 | on-device fit, latency, quality-floor and degradation tests | planned |
| Aegis-X orchestration | system integration | framework component | governed model/memory/tool/verifier loop | 17 | end-to-end trace, permission, failure, and interaction tests | future target |
| quantization | deployment technique | training/deployment technique | distinct model artifacts | 14 | memory/latency/quality by task | planned |
| distillation | training technique | training/deployment technique | deployable student variants | 14 | student Pareto improvement and safety regression | proposed |
| pruning | compression technique | training/deployment technique | structured/unstructured capacity removal | 14 | sparsity, realized speed/memory, quality recovery | proposed |
| speculative decoding | inference technique | optional accelerator | compatible draft + target backend path | 14 | exactness, acceptance, net latency and memory | proposed |
| superposition analysis | representation study | interpretability tool | feature interference in SARN activations | 11 | interference and causal feature studies | research topic |
| sparse autoencoders | interpretability technique | interpretability tool | offline feature dictionaries first | 11 | reconstruction, sparsity, stability, interventions | proposed offline tool |
| polysemanticity analysis | interpretation study | interpretability tool | neuron/feature selectivity analysis | 11 | feature selectivity and causal tests | research topic |
| activation patching | interpretability technique | interpretability tool | causal intervention across matched examples | 11 | controlled behavioral effect and replication | planned research tool |
| feature dictionaries | interpretability artifact | interpretability tool | versioned SAE feature basis and labels | 11 | reconstruction, stability, privacy, causal validation | planned research artifact |
| grokking | training phenomenon | research phenomenon | synthetic generalization study | 2/11 | predeclared delayed-generalization curves | observe, do not assume |
| double descent | scaling phenomenon | research phenomenon | scaling observatory | 2 | error versus capacity/data/training plots | observe, do not assume |
| scaling laws | scaling phenomenon | research phenomenon | capability/loss/cost trend analysis | 2 onward | preregistered fits, uncertainty, extrapolation checks | research topic |
| lottery-ticket hypothesis | compression research | training/deployment technique | iterative pruning/rewinding | 14 | retraining and deployment Pareto evidence | proposed experiment |
| Shampoo/second-order optimization | optimizer experiment | training/deployment technique | larger-run training lab | 14+ | wall-clock/convergence versus AdamW, memory cost | deferred |
| DPO | post-training technique | training/deployment technique | preference tuning after SFT | 10+ | held-out preferences, shortcuts, capability/safety | planned experiment |
| goal generalization | evaluation objective | research phenomenon | OOD instruction/constraint adherence | 2 onward | structural shifts and hidden constraints | required evaluation |
| goal misgeneralization | safety failure | safety risk | threat/evaluation suite | 12 | proxy-breaking distribution shifts | required risk test |
| reward hacking | safety failure | safety risk | hidden-performance and tool-task tests | 12 | reward/performance gap | required risk test |
| instrumental convergence | threat model | safety risk | resource/autonomy capability limits | all | observable behavior tests + hard permissions | threat-model input |
| mesa-optimization | threat model | safety risk | research evaluations, not a module | later capable systems | behavioral/interpretability evidence with caveats | unresolved risk |
| deceptive alignment | threat model | safety risk | oversight-shift tests, hard containment | later capable systems | no universal detector claim | unresolved risk |
| unsafe self-modification | safety failure | safety risk | weight/config/policy mutation attempts | 12 onward | mutation denied, offline authority boundary preserved | prohibited capability |
| memory poisoning | safety failure | safety risk | neural/session/persistent memory writes | 9/12 | provenance, conflict, reset, access, adversarial-write tests | required risk test |
| tool misuse | safety failure | safety risk | unauthorized or harmful structured calls | 10/12 | permission, argument, sandbox, approval, audit tests | required risk test |
| mode collapse | mostly GAN phenomenon | research phenomenon | use precise repetition/diversity/expert-collapse metrics | 3 onward | metric-specific evidence | terminology correction |
| Dragon Hatchling/BDH | external research architecture | research phenomenon | inspiration/comparator for graph + Hebbian work | 5–7 | independent reproduction and matched comparison | reference, not dependency |
| LLM | model category | accepted baseline | SARN text generator or hosted models | 1 onward | model card and task metrics | core category |
| SLM | deployment category | training/deployment technique | smaller checkpoint/profile, not separate intelligence type | 12/13 | per-profile quality/cost | planned family tier |
| MLM | training/model category | future adapter | optional encoder for retrieval/classification | 9+ | task need versus simpler embedding model | future adapter |
| VLM | multimodal category | future adapter | later vision-language adapter/model | 15 | multimodal data, metrics, safety case | future |
| SAM/segmentation | dedicated external model | future adapter | Aegis vision tool/adapter | 15 | segmentation task and license/runtime fit | future |
| speech encoder/decoder | multimodal category | future adapter | Aegis speech input/output adapter | 15 | speech quality, latency, privacy, and safety case | future |
| LAM/action model | model/system category | future adapter | typed plan proposer behind Aegis policy | 15 | action correctness and containment | future high-risk |
| robotics/sensor integration | physical-system adapter | future adapter | sensor input and actuator/tool boundary | 15 | real-time behavior, interlocks, simulation, human approval | future high-risk |

## Interpretation Rule

Only rows marked accepted baseline are default model commitments today. Core-spine roles define the SARN-Hybrid thesis, not accomplished work. “Planned” means the architecture reserves a clean interface and roadmap gate. “Proposed,” “future,” and “research topic” do not imply implementation or expected success.

Canonical role values are `accepted baseline`, `core spine`, `optional accelerator`, `framework component`, `interpretability tool`, `training/deployment technique`, `safety risk`, `research phenomenon`, and `future adapter`.

When an experiment concludes, update this matrix and link its decision record. Do not erase rejected rows; change their status and preserve the evidence.
