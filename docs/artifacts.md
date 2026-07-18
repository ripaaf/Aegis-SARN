# Artifact Policy

Generated artifacts are reproducible outputs, not the source of truth. Source code, tests, and documentation define the experiment; artifacts record a local run of that experiment.

## Locations

- Checkpoints: `artifacts/<run>/train/sarn-dense-smoke.pt`
- Run manifests: next to the command output that created them, such as `artifacts/<run>/eval/*.json`
- Registries: `artifacts/<run>/runs/registry.json` or `runs/registry.json`
- Phase 2 reports: `artifacts/reports/sarn-dense-phase2-baseline.md` and `.json`
- Phase 3 sweeps: `artifacts/phase3-sweep/sweep-summary.md` and `.json`
- Phase 3 comparisons: `artifacts/reports/baseline-comparison.md` and `.json`

- Phase 4 attention sweeps: artifacts/phase4-attention/attention-sweep-summary.md and attention-sweep-summary.json
- Phase 4 attention comparisons: artifacts/reports/attention-comparison.md and attention-comparison.json

## Git Policy

Commit source files, tests, and documentation. Do not commit generated checkpoints, registries, reports, run directories, caches, virtual environments, or bytecode unless a maintainer explicitly promotes a small artifact into tracked documentation.

The default ignored outputs include:

- `artifacts/`
- `runs/`
- `*.pt`
- `__pycache__/`
- `.pytest_cache/`
- `.venv/`
- `*.egg-info/`

## Cleaning

Generated outputs can be removed by deleting the relevant ignored directory, for example `artifacts/phase3-sweep` or `runs`. On some Windows workspaces, delete or atomic replace may be denied by the backing filesystem. In that case, create a new output directory for the next run or clean the directory from a shell with sufficient permissions.

## Windows Write Note

The code prefers deterministic JSON/checkpoint content. On Windows workspaces where atomic `os.replace` is denied, direct deterministic writes may be used for JSON and checkpoint artifacts so CPU reproduction remains usable. This affects file replacement behavior only; it does not make generated artifacts authoritative source files.

## Scope

Artifact reports may contain local CPU timings and toy-task metrics. They are suitable for regression checks and baseline comparison. They are not language capability claims and do not imply that SARN-Hybrid, retrieval, tools, memory systems, or multimodal systems are implemented.
