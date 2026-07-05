# Repository and Package Layout

## 1. Phase 1 Tree and Future Growth

The implemented Phase 1 package uses one distribution namespace with explicit internal model/runtime boundaries:

```text
Aegis-SARN/
  README.md
  LICENSE
  pyproject.toml
  docs/
  src/
    aegis_sarn/
      config.py
      cli.py
      aegis/
        schemas.py
        trace.py
        backends.py
        controller.py
      sarn/
        layers.py
        model.py
        data.py
        generation.py
        training.py
        checkpoint.py
      eval/
      utils/
  tests/
  artifacts/          # ignored local generated checkpoints/manifests
```

Future `configs/`, `research/`, and `reports/` trees are created only with their first real artifact; empty architecture theater remains prohibited.

## 2. Dependency Direction

```text
clients -> aegis_sarn.cli -> aegis_sarn.aegis.SessionController
controller -> GenerationBackend protocol
SARNBackend -> aegis_sarn.sarn public model/generation interface
aegis_sarn.sarn -> config/eval/utils only; no aegis runtime imports
future research -> public aegis_sarn APIs
```

`aegis_sarn.sarn` cannot import policy, tools, persistent stores, or application services. The model package trains independently. Aegis uses the public SARN model/generation interface through `SARNBackend`; direct imports of individual internal layers are not runtime contracts.

## 3. Package Responsibilities

### `src/aegis_sarn/aegis`

Production-oriented orchestration and capability control. Core logic depends on protocols; adapters contain backend and infrastructure details. The application layer is testable with fake models, stores, and executors.

### `src/aegis_sarn/sarn`

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

## 6. Initial Implementation Status

The Phase 1 minimum slice now contains:

```text
pyproject.toml
src/aegis_sarn/config.py
src/aegis_sarn/sarn/
src/aegis_sarn/aegis/
src/aegis_sarn/eval/
src/aegis_sarn/utils/
tests/
```

This is intentionally a vertical slice rather than scaffolding future subsystems. New directories require implemented behavior and tests.

## 7. Ownership and Review

Changes crossing `aegis` and `sarn` boundaries require an interface review. Model-format changes require backward-compatibility and migration review. Policy/tool changes require security review. Dataset and model releases require license and card review. Decision authority can begin with maintainers and later be formalized in governance documents.
