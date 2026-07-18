'''Baseline report synthesis for the SARN-Dense control model.'''

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from aegis_sarn.config import ModelConfig
from aegis_sarn.registry import record_entry
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.utils import config_hash, git_commit, package_version, write_json

REPORT_BASENAME = 'sarn-dense-phase2-baseline'
REPORT_SCHEMA_VERSION = 'aegis.baseline_report/v1'


@dataclass(slots=True)
class BaselineReportResult:
    run_id: str
    markdown_path: Path
    json_path: Path
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'run_id': self.run_id,
            'markdown_path': str(self.markdown_path),
            'json_path': str(self.json_path),
            'summary': self.summary,
        }


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get('schema_version') != 'aegis.run_manifest/v1':
        return None
    if 'run_id' not in payload or 'command' not in payload:
        return None
    payload = dict(payload)
    payload['_manifest_path'] = str(path)
    return payload


def _discover_manifests(run_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    if not run_dir.exists():
        raise ValueError(f'run directory does not exist: {run_dir}')
    for path in run_dir.rglob('*.json'):
        if path.name == 'registry.json':
            continue
        manifest = _load_manifest(path)
        if manifest is not None:
            manifests.append(manifest)
    manifests.sort(key=lambda item: (item.get('created_at') or '', item['run_id']))
    return manifests


def _latest(manifests: list[dict[str, Any]], command: str) -> dict[str, Any] | None:
    candidates = [manifest for manifest in manifests if manifest.get('command') == command]
    return candidates[-1] if candidates else None


def _first_model_config(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    for manifest in manifests:
        config = manifest.get('model_config') or {}
        if config:
            return config
    return {}


def _parameter_count(model_config: dict[str, Any], bench: dict[str, Any] | None) -> int | None:
    if bench is not None:
        count = (bench.get('metrics') or {}).get('parameter_count')
        if count is not None:
            return int(count)
    if not model_config:
        return None
    return SARNDense(ModelConfig.from_dict(model_config)).count_parameters()


def _checkpoint_path(*manifests: dict[str, Any] | None) -> str | None:
    for manifest in manifests:
        if manifest is None:
            continue
        artifacts = manifest.get('artifacts') or {}
        checkpoint = artifacts.get('checkpoint') or artifacts.get('checkpoint_path')
        if checkpoint:
            return str(checkpoint)
    return None


def _seed(*manifests: dict[str, Any] | None) -> int | None:
    for manifest in manifests:
        if manifest is None:
            continue
        seed = (manifest.get('seed_config') or {}).get('seed')
        if seed is not None:
            return int(seed)
    return None


def _device(*manifests: dict[str, Any] | None) -> str | None:
    for manifest in manifests:
        if manifest is None:
            continue
        runtime_device = (manifest.get('runtime_config') or {}).get('device')
        if runtime_device:
            return str(runtime_device)
        info_device = (manifest.get('device_info') or {}).get('device')
        if info_device:
            return str(info_device)
    return None


def _source_paths(manifests: list[dict[str, Any]]) -> list[str]:
    return [str(manifest['_manifest_path']) for manifest in manifests]


def _format_value(value: Any) -> str:
    if value is None:
        return 'not recorded'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def _markdown_table(mapping: dict[str, Any]) -> str:
    lines = ['| Field | Value |', '|---|---|']
    for key, value in mapping.items():
        lines.append(f'| {key} | {_format_value(value)} |')
    return '\n'.join(lines)


def _render_markdown(summary: dict[str, Any]) -> str:
    model_config = summary['model_config']
    training = summary['training']
    evaluation = summary['evaluation']
    benchmark = summary['benchmark']
    memory = summary['memory_estimates']
    reproducibility = summary['reproducibility']
    limitations = '\n'.join(f'- {item}' for item in summary['known_limitations'])
    config_lines = '\n'.join(
        f'- `{key}`: `{value}`' for key, value in sorted(model_config.items())
    )
    generation_ids = evaluation.get('generation_token_ids') or []
    return f'''# SARN-Dense Phase 2 Baseline Report

Generated at: {summary['generated_at']}

SARN-Dense is the control baseline for this repository. This report does not claim that SARN-Hybrid, MoE, graph workspace, working memory, SSM, retrieval, tools, VLM, SAM, LAM, or multimodal modules are implemented.

## Model

- Model family: SARN-Dense
- Role: {summary['role']}
- Parameter count: {_format_value(summary['parameter_count'])}

### Model Config

{config_lines or '- not recorded'}

## Dataset And Task

{_markdown_table(summary['dataset'])}

## Training

{_markdown_table(training)}

## Evaluation

{_markdown_table(evaluation)}

Generation sample token IDs: `{generation_ids}`

## Benchmark

{_markdown_table(benchmark)}

## Memory Estimates

{_markdown_table(memory)}

## Checkpoint

`{_format_value(summary['checkpoint_path'])}`

## Reproducibility

{_markdown_table(reproducibility)}

Source manifests:

{chr(10).join(f'- `{path}`' for path in summary['source_manifests'])}

## Known Limitations

{limitations}
'''


def build_baseline_summary(run_dir: Path) -> dict[str, Any]:
    manifests = _discover_manifests(run_dir)
    if not manifests:
        raise ValueError(f'no run manifests found under {run_dir}')

    train = _latest(manifests, 'train-smoke')
    eval_manifest = _latest(manifests, 'eval-toy')
    bench = _latest(manifests, 'bench')
    model_config = _first_model_config(manifests)
    train_metrics = {} if train is None else train.get('metrics') or {}
    eval_metrics = {} if eval_manifest is None else eval_manifest.get('metrics') or {}
    bench_metrics = {} if bench is None else bench.get('metrics') or {}
    run_id = str(uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()
    dataset_name = (
        eval_metrics.get('dataset_name')
        or train_metrics.get('dataset_name')
        or 'toy/repeated_pattern'
    )
    task = eval_metrics.get('task') or train_metrics.get('task') or 'repeated_pattern'
    summary: dict[str, Any] = {
        'schema_version': REPORT_SCHEMA_VERSION,
        'run_id': run_id,
        'report_name': REPORT_BASENAME,
        'generated_at': generated_at,
        'source_run_dir': str(run_dir),
        'model_family': 'SARN-Dense',
        'role': 'control baseline for future SARN-Hybrid comparisons',
        'model_config': model_config,
        'parameter_count': _parameter_count(model_config, bench),
        'dataset': {
            'dataset_name': dataset_name,
            'task': task,
            'split': eval_metrics.get('split') or 'toy validation',
            'examples': eval_metrics.get('examples') or train_metrics.get('examples'),
            'sequence_length': (
                eval_metrics.get('sequence_length') or train_metrics.get('sequence_length')
            ),
            'benchmark_scope': 'generated toy bytes, not a real language benchmark',
        },
        'training': {
            'train_loss_start': train_metrics.get('initial_loss'),
            'train_loss_end': train_metrics.get('final_loss'),
            'training_evaluation_loss': train_metrics.get('evaluation_loss'),
            'completed_step': train_metrics.get('completed_step'),
        },
        'evaluation': {
            'eval_loss': eval_metrics.get('validation_loss'),
            'perplexity': eval_metrics.get('perplexity'),
            'token_accuracy': eval_metrics.get('token_accuracy'),
            'generation_sample': eval_metrics.get('generation_sample'),
            'generation_token_ids': eval_metrics.get('generation_token_ids'),
        },
        'benchmark': {
            'tokens_per_second': bench_metrics.get('tokens_per_second'),
            'runtime_duration_ms': bench_metrics.get('runtime_duration_ms'),
            'prompt_length': bench_metrics.get('prompt_length'),
            'generated_tokens': bench_metrics.get('generated_tokens'),
            'repeats': bench_metrics.get('repeats'),
        },
        'memory_estimates': {
            'parameter_memory_bytes': bench_metrics.get('parameter_memory_bytes'),
            'approximate_kv_cache_bytes': bench_metrics.get('approximate_kv_cache_bytes'),
            'active_parameter_count': bench_metrics.get('active_parameter_count'),
            'total_parameter_count': bench_metrics.get('parameter_count'),
        },
        'checkpoint_path': _checkpoint_path(train, eval_manifest, bench),
        'reproducibility': {
            'git_commit': git_commit(),
            'package_version': package_version(),
            'device': _device(train, eval_manifest, bench),
            'seed': _seed(train, eval_manifest, bench),
            'source_manifest_count': len(manifests),
            'source_config_hashes': [
                manifest.get('config_hash') for manifest in manifests if manifest.get('config_hash')
            ],
        },
        'source_manifests': _source_paths(manifests),
        'known_limitations': [
            'The toy datasets are generated patterns and byte streams, not natural-language benchmarks.',
            'Loss, perplexity, and token accuracy measure only the tiny generated task distribution.',
            'Generation samples are toy byte/token outputs and should not be treated as useful prose.',
            'CPU benchmark numbers are local systems measurements and are not cross-machine claims.',
            'SARN-Hybrid and other future mechanisms are not implemented in this baseline report.',
        ],
    }
    summary['command'] = 'report-baseline'
    summary['timestamp'] = generated_at
    summary['git_commit_hash'] = summary['reproducibility']['git_commit']
    summary['package_version'] = summary['reproducibility']['package_version']
    summary['seed'] = summary['reproducibility']['seed']
    summary['device'] = summary['reproducibility']['device']
    summary['status'] = 'completed'
    summary['config'] = {
        'model': model_config,
        'source_run_dir': str(run_dir),
        'source_config_hashes': summary['reproducibility']['source_config_hashes'],
    }
    summary['metrics'] = {
        'eval_loss': summary['evaluation']['eval_loss'],
        'perplexity': summary['evaluation']['perplexity'],
        'token_accuracy': summary['evaluation']['token_accuracy'],
        'tokens_per_second': summary['benchmark']['tokens_per_second'],
        'parameter_count': summary['parameter_count'],
    }
    summary['artifacts'] = {
        'checkpoint': summary['checkpoint_path'],
        'source_manifests': summary['source_manifests'],
    }
    summary['limitations'] = summary['known_limitations']
    summary['reproducibility']['report_config_hash'] = config_hash(summary)
    summary['config_hash'] = summary['reproducibility']['report_config_hash']
    return summary


def write_baseline_report(
    run_dir: Path,
    output_dir: Path,
    registry_path: Path | None = None,
) -> BaselineReportResult:
    summary = build_baseline_summary(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f'{REPORT_BASENAME}.md'
    json_path = output_dir / f'{REPORT_BASENAME}.json'
    markdown_path.write_text(_render_markdown(summary), encoding='utf-8', newline='\n')
    write_json(json_path, summary)
    if registry_path is not None:
        record_entry(
            registry_path,
            {
                'run_id': summary['run_id'],
                'command_name': 'report-baseline',
                'timestamp': summary['generated_at'],
                'git_commit': summary['reproducibility']['git_commit'],
                'package_version': summary['reproducibility']['package_version'],
                'config_hash': summary['reproducibility']['report_config_hash'],
                'checkpoint_path': summary['checkpoint_path'],
                'manifest_path': str(json_path),
                'metrics_summary': {
                    'train_loss_start': summary['training']['train_loss_start'],
                    'train_loss_end': summary['training']['train_loss_end'],
                    'eval_loss': summary['evaluation']['eval_loss'],
                    'perplexity': summary['evaluation']['perplexity'],
                    'token_accuracy': summary['evaluation']['token_accuracy'],
                    'tokens_per_second': summary['benchmark']['tokens_per_second'],
                },
                'device': summary['reproducibility']['device'],
                'seed': summary['reproducibility']['seed'],
                'status': 'completed',
            },
        )
    return BaselineReportResult(
        run_id=summary['run_id'],
        markdown_path=markdown_path,
        json_path=json_path,
        summary=summary,
    )
