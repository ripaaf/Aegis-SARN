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
- Phase 5 workspace sweeps: artifacts/phase5-workspace/workspace-sweep-summary.md and workspace-sweep-summary.json
- Phase 5 workspace comparisons: artifacts/reports/workspace-comparison.md and workspace-comparison.json
- Phase 6 graph sweeps: artifacts/phase6-graph/graph-sweep-summary.md and graph-sweep-summary.json
- Phase 6 graph comparisons: artifacts/reports/graph-comparison.md and graph-comparison.json
- Phase 7 memory sweeps: `artifacts/phase7-memory/memory-sweep-summary.md` and `memory-sweep-summary.json`
- Phase 7 memory comparisons: `artifacts/reports/memory-comparison.md` and `memory-comparison.json`
- Phase 8 expert sweeps: `artifacts/phase8-experts/expert-sweep-summary.md` and `expert-sweep-summary.json`
- Phase 8 expert comparisons: `artifacts/reports/expert-comparison.md` and `expert-comparison.json`

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

Artifact reports may contain local CPU timings and toy/structural/memory-task metrics. They are suitable for regression checks and controlled comparison. Phase 7 artifacts do not establish useful or human-like memory; Phase 8 routing artifacts do not establish expert specialization or full MoE scaling. Reports do not imply that SARN-Hybrid, persistent memory, retrieval, tools, or multimodal systems are implemented.
