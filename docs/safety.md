# Safety, Alignment, and Security

## 1. Safety Position

Aegis assumes model output can be mistaken, manipulated, or adversarial. Safety is built through containment, authorization, observability, evaluation, and training—not through a fictional module that reads a model's true intentions.

The initial system has no independent long-horizon autonomy. It responds within a user-initiated, bounded run. Background operation, self-scheduling, remote control, robotics, financial actions, self-modification, and resource acquisition are separate capability changes requiring new threat models and explicit approval.

## 2. Protected Assets

- user and third-party data;
- credentials and secrets;
- filesystem, network, devices, and accounts;
- model and dataset artifacts;
- persistent memory integrity;
- training and evaluation integrity;
- audit records and policy configuration;
- human authority and informed consent;
- compute, money, and external reputation.

## 3. Threat Actors and Failure Sources

- malicious user;
- malicious retrieved document or tool output;
- compromised extension, dataset, checkpoint, or dependency;
- accidental operator misconfiguration;
- ordinary model hallucination or poor generalization;
- model behavior optimized for a faulty proxy;
- cross-user bugs and race conditions;
- future more capable models exhibiting strategic behavior.

## 4. Capability Control

Default-deny rules apply to network, filesystem writes, process execution, persistent memory, credentials, external communication, physical actuators, and model/ policy changes. Permissions are scoped by resource and duration. A model proposal cannot grant permission or suppress an approval step.

High-impact operations require deterministic validation where possible, human confirmation with a concrete preview, sandboxing, time and quantity limits, and a receipt. Irreversible operations should not be early project capabilities.

## 5. Prompt Injection Boundary

System policy, user instructions, retrieved evidence, memories, and tool output remain typed. Retrieved or tool-provided prose cannot redefine policy. Models may still follow malicious text despite delimiters, so actual security rests on external permissions and validators.

Indirect-injection tests cover documents that request secret disclosure, unauthorized tool use, policy override, memory writes, or concealment. The correct framework behavior is to prevent the side effect even if the generated text is poor.

## 6. Reward Hacking

Any optimized metric can become a target. Controls include multiple metrics, hidden tests, adversarial examples, process and outcome measures, distribution-shift evaluation, manual audits, and conservative deployment limits.

Examples:

- a verifier pass rate is paired with independent correctness tests;
- preference win rate is checked for length, style, and agreement shortcuts;
- tool-task completion is checked for unauthorized side effects;
- retrieval attribution is checked for answer correctness, not merely citation presence.

The model is never allowed to modify its evaluator, tests, reward logs, or approval records.

## 7. Goal Misgeneralization and Instrumental Behavior

Training tasks include shifts where superficial proxies break: new wording, hidden constraints, changed correlations, decoy rewards, and opportunities to take unnecessary resources. Evaluations measure whether the model follows the intended instruction and constraints, not only whether it completes an outcome.

Instrumental-convergence concerns motivate hard limits on persistence, self-modification, resource access, delegation, concealment, and disabling oversight. They do not justify claiming a universal “power-seeking detector.”

## 8. Mesa-Optimization and Deceptive Alignment

These are research threat models, not currently solved engineering detections. Aegis can look for concerning observable behavior through:

- train/deploy environment variations;
- tests with and without apparent oversight;
- hidden objectives and held-out opportunities;
- consistency and sandbagging evaluations;
- interpretability research with causal follow-up;
- strict limits that remain effective regardless of internal motive.

A learned classifier called `deception_monitor` would create false confidence and is not part of the normative architecture. Claims about internal goals require extraordinary evidence.

## 9. Memory Safety

Session state is isolated and resettable. Persistent writes are authenticated, source-linked, sensitivity-labeled, auditable, and reversible where possible. The model cannot decide retention or owner identity. Poisoning, conflicts, stale facts, inferred sensitive attributes, and cross-session leakage have dedicated tests.

## 10. Training-Time Safety and Supply Chain

- verify artifact digests and source lineage;
- restrict executable dataset content and unsafe deserialization;
- scan dependencies and pin environments;
- isolate training jobs and credentials;
- prevent benchmarks from entering training mixtures;
- control who can publish checkpoints or alter policy;
- assess memorization, bias, dangerous capability, and extraction risk before release.

## 11. Release Tiers

Possible tiers are research-only, local text generation, retrieval-enabled, read-only tools, reversible tools, and high-impact capabilities. Promotion requires a tier-specific safety case. Model scale alone does not define risk; tool access, autonomy, data, and deployment context matter.

## 12. Incident Response

The project will maintain a private reporting channel before public deployment. Response procedure:

1. preserve relevant artifacts and sanitized traces;
2. contain the capability, tool, model, or release;
3. assess affected users/data and legal duties;
4. reproduce and classify root cause;
5. patch controls and add regression tests;
6. rotate secrets or revoke artifacts if necessary;
7. document lessons and update the threat model.

Safety-relevant logs must be tamper-evident enough for their threat level, while respecting privacy and retention limits.

## 13. Non-Claims

Passing a benchmark does not prove alignment. DPO does not create a correct objective. An SAE feature does not reveal a complete thought. A verifier can be wrong. A refusal style can be gamed. Sandboxing lowers impact but does not make an action wise. These limitations must appear in release documentation.

## 14. Advanced Architecture Safety Risks

Aegis-X adds new state, routing, checking, and effect interfaces. Its safety case therefore tracks the following as distinct risks rather than hiding them under generic alignment language:

| Risk | Example failure | Required response |
|---|---|---|
| reward hacking | maximizes verifier/reward score while failing the intended task | hidden objectives, independent checks, immutable evaluation logs |
| goal misgeneralization | follows a learned proxy under distribution shift | proxy-breaking and constraint-shift evaluations |
| mesa-optimization | learned computation behaves like optimization for an unintended internal objective | capability limits, behavioral stress tests, cautious interpretability claims |
| deceptive alignment | behavior changes when oversight or deployment conditions appear different | oversight-shift tests; never claim a universal deception detector |
| instrumental convergence | proposes unnecessary persistence, resources, concealment, or control | hard permission, budget, persistence, and delegation limits |
| unsafe tool use | proposes harmful or unauthorized arguments/actions | schema and semantic validation, sandboxing, approval, receipts |
| memory poisoning | stores false, adversarial, cross-user, or policy-like content | provenance, owner scope, conflict checks, reset/deletion, write approval |
| self-modification | attempts to alter weights, code, configuration, policy, or evaluators | serving-time mutation denied; offline governed release path only |
| hidden persistence | encodes state outside approved session or memory stores | storage/network isolation, lifecycle tests, auditable writes |
| simulator bypass | treats an untested plan as simulated/validated or escapes the simulation boundary | signed checker identity, environment isolation, explicit inconclusive state |
| verifier gaming | targets checker shortcuts or suppresses unfavorable findings | checker diversity, hidden tests, immutable findings, false-pass audits |

### Hard Controls

- no autonomous base-weight mutation;
- no unrestricted tool, network, filesystem, account, or actuator execution;
- no hidden long-term memory writes or model-selected retention policy;
- no self-replication, autonomous deployment, or background persistence capability;
- no objective or permission to acquire compute, money, credentials, influence, or other resources;
- every action plan passes typed capability, argument, budget, and permission gates;
- high-risk or irreversible actions require a concrete preview and human approval;
- model-generated plans, calls, checks, approvals, and side-effect receipts are logged and auditable;
- verifiers and simulators can return `inconclusive` or `error`; those states never count as approval;
- safety controls remain external to model-generated prose and cannot be disabled by the model.

These controls reduce reachable harm even when model internals are misunderstood. They do not prove that the model has safe goals.
