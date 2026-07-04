# Repository and Package Layout

## 1. Intended Tree

The repository begins documentation-only. Implementation should grow into this shape rather than placing everything in one package:

```text
Aegis-SARN/
  README.md
  LICENSE
  pyproject.toml
  docs/
  configs/
    models/
    training/
    runtime/
    evaluation/
    policy/
  src/
    aegis/
      api/
      application/
      artifacts/
      backends/
      context/
      hardware/
      memory/
      policy/
      retrieval/
      runtime/
      tools/
      tracing/
      verification/
    sarn/
      config/
      data/
      modules/
      models/
      training/
      generation/
      interpretability/
  tests/
    unit/
    contracts/
    integration/
    security/
    performance/
  research/
    experiments/
    generators/
    analyses/
  scripts/
  artifacts/          # ignored local cache; manifests may be tracked
  reports/
    decisions/
    evaluations/
    model-cards/
    data-cards/
```

Directories are created only when their first real file is implemented; empty architecture theater is avoided.

## 2. Dependency Direction

```text
clients -> aegis.api -> aegis.application
application -> framework ports/interfaces
adapters -> ports/interfaces
aegis backend adapter -> sarn public inference interface
sarn modules -> no aegis implementation imports
research -> may import public aegis/sarn APIs
```

`sarn` cannot import policy, tools, persistent stores, or application services. The model package must train independently. Aegis cannot import internal SARN layers except through an explicit research adapter.

## 3. Package Responsibilities

### `src/aegis`

Production-oriented orchestration and capability control. Core logic depends on protocols; adapters contain backend and infrastructure details. The application layer is testable with fake models, stores, and executors.

### `src/sarn`

Tensor code, architecture configuration, model serialization, training/generation primitives, and instrumentation. Experimental layers are configuration-gated and unit-tested against reference implementations.

### `research`

Experiment definitions, synthetic generators, exploratory analysis, and preregistrations. Reusable stable code graduates into `src`; notebooks are not the sole copy of an algorithm or result.

### `configs`

Human-readable, schema-validated inputs. Resolved run configs are artifacts and must not be edited in place after a run.

### `reports`

Reviewed conclusions and artifact links. Large checkpoints, raw datasets, and bulky traces belong in an artifact store rather than Git.

## 4. Configuration Naming

Names express intent while digests provide identity:

```text
configs/models/sarn_dense_micro.yaml
configs/training/synthetic_smoke.yaml
configs/runtime/nano_cpu.yaml
configs/evaluation/reasoning_v1.yaml
```

Configuration inheritance is shallow and its resolved form is always logged. Environment variables are limited to secrets and deployment paths, not hidden model hyperparameters.

## 5. Artifact Locations

Local `artifacts/`, caches, datasets, checkpoints, secrets, generated indexes, and private traces are ignored by Git. Small manifests, checksums, schemas, and evaluation summaries are tracked. Artifact URIs must be replaceable so local filesystem and remote stores share the same logical registry interface.

## 6. Initial Implementation Order

The first code commit should create only:

```text
pyproject.toml
src/sarn/config/
src/sarn/modules/
src/sarn/models/
src/sarn/training/
src/aegis/api/
src/aegis/application/
src/aegis/backends/
tests/unit/
tests/integration/
configs/models/
configs/training/
```

It should deliver the Phase 1 vertical slice rather than scaffold every future subsystem.

## 7. Ownership and Review

Changes crossing `aegis` and `sarn` boundaries require an interface review. Model-format changes require backward-compatibility and migration review. Policy/tool changes require security review. Dataset and model releases require license and card review. Decision authority can begin with maintainers and later be formalized in governance documents.
