'''Phase 3 baseline scaling, comparison, and quality gates.'''

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from aegis_sarn.config import (
    ArtifactConfig,
    DecodingConfig,
    ModelConfig,
    SeedConfig,
    TrainingConfig,
)
from aegis_sarn.eval import benchmark_generation, evaluate_toy
from aegis_sarn.registry import record_entry, record_manifest
from aegis_sarn.sarn.checkpoint import load_checkpoint
from aegis_sarn.sarn.training import run_smoke_training
from aegis_sarn.utils import config_hash, git_commit, package_version, write_json

SWEEP_SCHEMA_VERSION = 'aegis.baseline_sweep/v1'
COMPARISON_SCHEMA_VERSION = 'aegis.baseline_comparison/v1'
GATE_SCHEMA_VERSION = 'aegis.quality_gates/v1'
COMMON_MANIFEST_FIELDS = frozenset(
    {
        'schema_version',
        'run_id',
        'command',
        'timestamp',
        'git_commit_hash',
        'package_version',
        'config',
        'config_hash',
        'seed',
        'device',
        'status',
        'metrics',
        'artifacts',
        'limitations',
    }
)


@dataclass(frozen=True, slots=True)
class SweepConfig:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    context_length: int
    train_steps: int

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=256,
            max_seq_len=max(16, self.context_length),
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            ffn_hidden_dim=self.d_ff,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'n_layers': self.n_layers,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'context_length': self.context_length,
            'train_steps': self.train_steps,
        }


def default_sweep_configs(train_steps: int | None = None) -> list[SweepConfig]:
    return [
        SweepConfig('micro', 1, 16, 2, 32, 16, train_steps or 8),
        SweepConfig('small', 1, 24, 3, 72, 24, train_steps or 8),
        SweepConfig('medium-tiny', 2, 32, 4, 96, 32, train_steps or 8),
    ]


def build_common_summary(
    command: str,
    run_id: str,
    created_at: str,
    seed: int,
    device: str,
    configuration: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    schema_version: str,
    status: str = 'completed',
) -> dict[str, Any]:
    return {
        'schema_version': schema_version,
        'run_id': run_id,
        'command': command,
        'timestamp': created_at,
        'created_at': created_at,
        'git_commit_hash': git_commit(),
        'git_commit': git_commit(),
        'package_version': package_version(),
        'config': configuration,
        'config_hash': config_hash(configuration),
        'seed': seed,
        'device': device,
        'status': status,
        'metrics': metrics,
        'artifacts': artifacts,
        'limitations': [
            'SARN-Dense is the only implemented model path.',
            'Toy generated tasks are for baseline measurement, not language capability claims.',
            'SARN-Hybrid and advanced modules remain future work.',
        ],
    }


def _sweep_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# SARN-Dense Phase 3 Scaling Sweep',
        '',
        f"Generated at: {summary['timestamp']}",
        '',
        'SARN-Dense is the only implemented model path. This sweep does not implement or validate SARN-Hybrid.',
        '',
        '| Config | Eval Loss | Perplexity | Token Accuracy | Tokens/Sec | Params | Param Bytes | KV Cache Bytes | Status |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for item in summary['results']:
        lines.append(
            '| {config_name} | {eval_loss:.6g} | {perplexity:.6g} | {token_accuracy:.4f} | '
            '{tokens_per_second:.3f} | {parameter_count} | {parameter_memory_bytes} | '
            '{approximate_kv_cache_bytes} | {status} |'.format(**item)
        )
    lines.extend(
        [
            '',
            '## Notes',
            '',
            '- Quality/resource tradeoffs are local CPU measurements on generated toy tasks.',
            '- Higher token accuracy and lower loss/perplexity are better for the toy task only.',
            '- Future hybrid modules must run through the same comparison path before claims are made.',
        ]
    )
    return '\n'.join(lines) + '\n'


def run_baseline_sweep(
    output_dir: Path,
    device: str = 'cpu',
    seed: int = 123,
    train_steps: int | None = None,
    batch_size: int = 2,
    max_new_tokens: int = 2,
    bench_repeats: int = 1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / 'runs' / 'registry.json'
    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []

    for sweep_config in default_sweep_configs(train_steps):
        config_started = time.perf_counter()
        config_dir = output_dir / sweep_config.name
        train_dir = config_dir / 'train'
        eval_dir = config_dir / 'eval'
        bench_dir = config_dir / 'bench'
        model_config = sweep_config.model_config()
        training_config = TrainingConfig(
            learning_rate=1.0e-2,
            batch_size=batch_size,
            sequence_length=min(sweep_config.context_length, model_config.max_seq_len),
            max_steps=sweep_config.train_steps,
            device=device,
        )
        seed_config = SeedConfig(seed=seed)
        command_args = {
            'command': 'sweep-baseline',
            'config_name': sweep_config.name,
            'seed': seed,
            'device': device,
        }
        train_result = run_smoke_training(
            model_config=model_config,
            training_config=training_config,
            seed_config=seed_config,
            artifact_config=ArtifactConfig(output_dir=train_dir),
            command_args=command_args,
        )
        record_manifest(registry_path, train_result.manifest_path)

        checkpoint = train_result.checkpoint_path
        artifacts = {'checkpoint': str(checkpoint)}
        model = load_checkpoint(checkpoint, map_location=device).model
        decoding_config = DecodingConfig(
            strategy='greedy',
            max_new_tokens=max_new_tokens,
            use_kv_cache=True,
            seed=seed,
        )
        eval_result = evaluate_toy(
            model=model,
            output_dir=eval_dir,
            seed_config=seed_config,
            decoding_config=decoding_config,
            device=device,
            batch_size=batch_size,
            sequence_length=training_config.sequence_length,
            command_args=command_args,
            artifacts=artifacts,
        )
        record_manifest(registry_path, eval_result.manifest_path)

        model = load_checkpoint(checkpoint, map_location=device).model
        bench_result = benchmark_generation(
            model=model,
            output_dir=bench_dir,
            seed_config=seed_config,
            decoding_config=decoding_config,
            device=device,
            prompt_length=min(8, training_config.sequence_length),
            repeats=bench_repeats,
            command_args=command_args,
            artifacts=artifacts,
        )
        record_manifest(registry_path, bench_result.manifest_path)

        eval_metrics = eval_result.metrics
        bench_metrics = bench_result.metrics
        results.append(
            {
                'config_name': sweep_config.name,
                'model_config': model_config.to_dict(),
                'sweep_config': sweep_config.to_dict(),
                'train_loss_start': train_result.initial_loss,
                'train_loss_end': train_result.final_loss,
                'eval_loss': float(eval_metrics['validation_loss']),
                'perplexity': float(eval_metrics['perplexity']),
                'token_accuracy': float(eval_metrics['token_accuracy']),
                'tokens_per_second': float(bench_metrics['tokens_per_second']),
                'parameter_count': int(bench_metrics['parameter_count']),
                'active_parameter_count': int(bench_metrics['active_parameter_count']),
                'parameter_memory_bytes': int(bench_metrics['parameter_memory_bytes']),
                'approximate_kv_cache_bytes': int(bench_metrics['approximate_kv_cache_bytes']),
                'runtime_duration_ms': (time.perf_counter() - config_started) * 1000.0,
                'seed': seed,
                'config_hash': config_hash(sweep_config.to_dict()),
                'checkpoint_path': str(checkpoint),
                'manifest_path': str(eval_result.manifest_path),
                'manifest_paths': {
                    'train': str(train_result.manifest_path),
                    'eval': str(eval_result.manifest_path),
                    'bench': str(bench_result.manifest_path),
                },
                'device': device,
                'status': 'completed',
            }
        )

    metrics = {
        'config_count': len(results),
        'completed_count': sum(1 for item in results if item['status'] == 'completed'),
        'best_eval_loss': min(item['eval_loss'] for item in results),
        'best_token_accuracy': max(item['token_accuracy'] for item in results),
        'best_tokens_per_second': max(item['tokens_per_second'] for item in results),
    }
    artifacts = {
        'summary_json': str(output_dir / 'sweep-summary.json'),
        'summary_markdown': str(output_dir / 'sweep-summary.md'),
        'registry': str(registry_path),
    }
    summary = build_common_summary(
        command='sweep-baseline',
        run_id=run_id,
        created_at=created_at,
        seed=seed,
        device=device,
        configuration={
            'configs': [config.to_dict() for config in default_sweep_configs(train_steps)],
            'batch_size': batch_size,
            'max_new_tokens': max_new_tokens,
            'bench_repeats': bench_repeats,
        },
        metrics=metrics,
        artifacts=artifacts,
        schema_version=SWEEP_SCHEMA_VERSION,
    )
    summary['results'] = results
    write_json(output_dir / 'sweep-summary.json', summary)
    (output_dir / 'sweep-summary.md').write_text(
        _sweep_markdown(summary), encoding='utf-8', newline='\n'
    )
    record_entry(
        registry_path,
        {
            'run_id': run_id,
            'command_name': 'sweep-baseline',
            'timestamp': created_at,
            'git_commit': summary['git_commit_hash'],
            'package_version': summary['package_version'],
            'config_hash': summary['config_hash'],
            'checkpoint_path': None,
            'manifest_path': str(output_dir / 'sweep-summary.json'),
            'metrics_summary': metrics,
            'device': device,
            'seed': seed,
            'status': 'completed',
        },
    )
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _score_inverse_loss(item: dict[str, Any]) -> float:
    return 1.0 / max(float(item['eval_loss']), 1.0e-12)


def _best_by(results: list[dict[str, Any]], key: str, reverse: bool = False) -> dict[str, Any]:
    return sorted(results, key=lambda item: float(item[key]), reverse=reverse)[0]


def _best_name(item: dict[str, Any]) -> str:
    return str(item['config_name'])


def _comparison_markdown(summary: dict[str, Any]) -> str:
    winners = summary['winners']
    lines = [
        '# SARN-Dense Baseline Comparison',
        '',
        f"Generated at: {summary['timestamp']}",
        '',
        'This comparison ranks SARN-Dense sweep outputs only. It does not implement SARN-Hybrid.',
        '',
        '| Criterion | Config | Score |',
        '|---|---|---:|',
    ]
    for name, item in winners.items():
        lines.append(f"| {name} | {item['config_name']} | {item['score']:.6g} |")
    lines.extend(
        [
            '',
            '## Notes',
            '',
            '- Best quality per parameter uses token accuracy divided by parameter count.',
            '- Best quality per MB uses token accuracy divided by parameter memory in MiB.',
            '- Best quality per second uses token accuracy multiplied by measured tokens/sec.',
            '- Balanced config normalizes toy accuracy, inverse loss, speed, and memory efficiency.',
        ]
    )
    return '\n'.join(lines) + '\n'


def compare_baselines(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    sweep = _read_json(input_dir / 'sweep-summary.json')
    results = sweep.get('results') or []
    if not results:
        raise ValueError('sweep summary does not contain results')
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = str(uuid4())

    best_loss = _best_by(results, 'eval_loss')
    best_perplexity = _best_by(results, 'perplexity')
    best_accuracy = _best_by(results, 'token_accuracy', reverse=True)
    best_speed = _best_by(results, 'tokens_per_second', reverse=True)
    best_quality_per_param = max(
        results,
        key=lambda item: float(item['token_accuracy']) / max(float(item['parameter_count']), 1.0),
    )
    best_quality_per_mb = max(
        results,
        key=lambda item: float(item['token_accuracy'])
        / max(float(item['parameter_memory_bytes']) / (1024.0 * 1024.0), 1.0e-12),
    )
    best_quality_per_second = max(
        results,
        key=lambda item: float(item['token_accuracy']) * float(item['tokens_per_second']),
    )

    max_speed = max(float(item['tokens_per_second']) for item in results)
    max_accuracy = max(float(item['token_accuracy']) for item in results)
    max_inverse_loss = max(_score_inverse_loss(item) for item in results)
    max_memory_efficiency = max(1.0 / float(item['parameter_memory_bytes']) for item in results)

    def balanced_score(item: dict[str, Any]) -> float:
        return (
            (float(item['token_accuracy']) / max(max_accuracy, 1.0e-12))
            + (_score_inverse_loss(item) / max(max_inverse_loss, 1.0e-12))
            + (float(item['tokens_per_second']) / max(max_speed, 1.0e-12))
            + ((1.0 / float(item['parameter_memory_bytes'])) / max_memory_efficiency)
        ) / 4.0

    balanced = max(results, key=balanced_score)
    winners = {
        'best_eval_loss': {'config_name': _best_name(best_loss), 'score': float(best_loss['eval_loss'])},
        'best_perplexity': {'config_name': _best_name(best_perplexity), 'score': float(best_perplexity['perplexity'])},
        'best_token_accuracy': {'config_name': _best_name(best_accuracy), 'score': float(best_accuracy['token_accuracy'])},
        'best_tokens_per_second': {'config_name': _best_name(best_speed), 'score': float(best_speed['tokens_per_second'])},
        'best_quality_per_parameter': {
            'config_name': _best_name(best_quality_per_param),
            'score': float(best_quality_per_param['token_accuracy']) / float(best_quality_per_param['parameter_count']),
        },
        'best_quality_per_mb': {
            'config_name': _best_name(best_quality_per_mb),
            'score': float(best_quality_per_mb['token_accuracy'])
            / (float(best_quality_per_mb['parameter_memory_bytes']) / (1024.0 * 1024.0)),
        },
        'best_quality_per_second': {
            'config_name': _best_name(best_quality_per_second),
            'score': float(best_quality_per_second['token_accuracy'])
            * float(best_quality_per_second['tokens_per_second']),
        },
        'best_balanced_config': {
            'config_name': _best_name(balanced),
            'score': balanced_score(balanced),
        },
    }

    summary = build_common_summary(
        command='compare-baselines',
        run_id=run_id,
        created_at=created_at,
        seed=int(sweep.get('seed') or 0),
        device=str(sweep.get('device') or 'unknown'),
        configuration={'input': str(input_dir), 'source_config_hash': sweep.get('config_hash')},
        metrics={'config_count': len(results)},
        artifacts={
            'comparison_json': str(output_dir / 'baseline-comparison.json'),
            'comparison_markdown': str(output_dir / 'baseline-comparison.md'),
        },
        schema_version=COMPARISON_SCHEMA_VERSION,
    )
    summary['winners'] = winners
    summary['results'] = results
    summary['notes'] = [
        'Quality/speed/memory rankings are based on tiny generated toy tasks.',
        'Balanced score is a simple normalized heuristic, not a scientific claim.',
        'Future hybrid modules should use this path for matched comparisons.',
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'baseline-comparison.json', summary)
    (output_dir / 'baseline-comparison.md').write_text(
        _comparison_markdown(summary), encoding='utf-8', newline='\n'
    )
    return summary


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _load_manifest(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _required_manifest_fields_present(path: Path) -> tuple[bool, list[str]]:
    payload = _load_manifest(path)
    missing = sorted(COMMON_MANIFEST_FIELDS - set(payload))
    return not missing, missing


def _declared_artifact_paths(value: Any, prefix: str = 'artifacts') -> list[tuple[str, Path]]:
    if isinstance(value, dict):
        paths: list[tuple[str, Path]] = []
        for key, child in value.items():
            paths.extend(_declared_artifact_paths(child, f'{prefix}.{key}'))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_declared_artifact_paths(child, f'{prefix}[{index}]'))
        return paths
    if isinstance(value, str) and value:
        return [(prefix, Path(value))]
    return []


def _gate_items(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(summary.get('results'), list):
        return list(summary['results'])
    if 'evaluation' in summary or 'benchmark' in summary:
        return [
            {
                'config_name': summary.get('report_name', 'baseline-report'),
                'eval_loss': (summary.get('evaluation') or {}).get('eval_loss'),
                'token_accuracy': (summary.get('evaluation') or {}).get('token_accuracy'),
                'perplexity': (summary.get('evaluation') or {}).get('perplexity'),
                'tokens_per_second': (summary.get('benchmark') or {}).get('tokens_per_second'),
                'parameter_count': summary.get('parameter_count')
                or (summary.get('memory_estimates') or {}).get('total_parameter_count'),
                'checkpoint_path': summary.get('checkpoint_path'),
                'manifest_path': None,
                'status': 'completed',
            }
        ]
    if 'metrics' in summary:
        metrics = summary['metrics']
        return [
            {
                'config_name': summary.get('command', 'metrics'),
                'eval_loss': metrics.get('eval_loss') or metrics.get('validation_loss'),
                'token_accuracy': metrics.get('token_accuracy'),
                'perplexity': metrics.get('perplexity'),
                'tokens_per_second': metrics.get('tokens_per_second'),
                'parameter_count': metrics.get('parameter_count'),
                'checkpoint_path': (summary.get('artifacts') or {}).get('checkpoint'),
                'manifest_path': summary.get('manifest_path'),
                'status': summary.get('status', 'completed'),
            }
        ]
    return []


def check_gates(
    summary_path: Path,
    max_eval_loss: float | None = None,
    min_token_accuracy: float | None = None,
    min_tokens_per_second: float | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({'name': name, 'passed': passed, 'detail': detail})

    add('summary_file_exists', summary_path.exists(), str(summary_path))
    if not summary_path.exists():
        return {
            'schema_version': GATE_SCHEMA_VERSION,
            'summary_path': str(summary_path),
            'passed': False,
            'checks': checks,
        }

    summary = _read_json(summary_path)
    missing_summary_fields = sorted(COMMON_MANIFEST_FIELDS - set(summary))
    add(
        'summary_required_fields',
        not missing_summary_fields,
        'ok' if not missing_summary_fields else ', '.join(missing_summary_fields),
    )
    for artifact_name, artifact_path in _declared_artifact_paths(summary.get('artifacts')):
        add(f'{artifact_name}_exists', artifact_path.exists(), str(artifact_path))

    items = _gate_items(summary)
    add('summary_has_items', bool(items), f'{len(items)} item(s)')

    for item in items:
        name = str(item.get('config_name') or item.get('run_id') or 'item')
        add(f'{name}:status_completed', item.get('status') == 'completed', str(item.get('status')))
        parameter_count = item.get('parameter_count')
        add(
            f'{name}:parameter_count_positive',
            _is_finite_number(parameter_count) and float(parameter_count) > 0,
            str(parameter_count),
        )
        checkpoint_path = item.get('checkpoint_path')
        if checkpoint_path:
            add(f'{name}:checkpoint_exists', Path(str(checkpoint_path)).exists(), str(checkpoint_path))
        else:
            add(f'{name}:checkpoint_path_present', False, 'missing checkpoint_path')

        for metric_name in ('eval_loss', 'token_accuracy', 'perplexity', 'tokens_per_second'):
            value = item.get(metric_name)
            add(f'{name}:{metric_name}_present', value is not None, str(value))
            add(f'{name}:{metric_name}_finite', _is_finite_number(value), str(value))

        if max_eval_loss is not None and item.get('eval_loss') is not None:
            add(
                f'{name}:eval_loss_threshold',
                float(item['eval_loss']) <= max_eval_loss,
                f"{item['eval_loss']} <= {max_eval_loss}",
            )
        if min_token_accuracy is not None and item.get('token_accuracy') is not None:
            add(
                f'{name}:token_accuracy_threshold',
                float(item['token_accuracy']) >= min_token_accuracy,
                f"{item['token_accuracy']} >= {min_token_accuracy}",
            )
        if min_tokens_per_second is not None and item.get('tokens_per_second') is not None:
            add(
                f'{name}:tokens_per_second_threshold',
                float(item['tokens_per_second']) >= min_tokens_per_second,
                f"{item['tokens_per_second']} >= {min_tokens_per_second}",
            )

        manifest_paths = item.get('manifest_paths') or {}
        if item.get('manifest_path'):
            manifest_paths = {'primary': item['manifest_path'], **manifest_paths}
        for manifest_name, manifest_path in manifest_paths.items():
            path = Path(str(manifest_path))
            add(f'{name}:{manifest_name}_manifest_exists', path.exists(), str(path))
            if path.exists() and path.name.endswith('.json'):
                try:
                    present, missing = _required_manifest_fields_present(path)
                except (OSError, json.JSONDecodeError):
                    present, missing = False, ['unreadable']
                add(
                    f'{name}:{manifest_name}_manifest_required_fields',
                    present,
                    'ok' if present else ', '.join(missing),
                )

    if summary.get('command') == 'sweep-attention':
        completed = [item for item in items if item.get('status') == 'completed']
        mha_items = [
            item for item in completed if item.get('attention_type') == 'mha'
        ]
        gqa_items = [
            item for item in completed if item.get('attention_type') == 'gqa'
        ]
        add(
            'attention:mha_baseline_present',
            bool(mha_items),
            f'{len(mha_items)} completed MHA variant(s)',
        )
        add(
            'attention:gqa_variant_present',
            bool(gqa_items),
            f'{len(gqa_items)} completed GQA variant(s)',
        )
        for item in completed:
            name = str(item.get('config_name') or 'attention')
            for field in ('n_heads', 'n_kv_heads', 'kv_group_size'):
                value = item.get(field)
                add(
                    f'{name}:{field}_positive',
                    _is_finite_number(value) and float(value) > 0,
                    str(value),
                )
            cache_bytes = item.get('approximate_kv_cache_bytes')
            add(
                f'{name}:kv_cache_bytes_positive',
                _is_finite_number(cache_bytes) and float(cache_bytes) > 0,
                str(cache_bytes),
            )
        lower_cache = False
        detail = 'MHA or GQA result missing or has invalid cache bytes'
        mha_cache_values = [
            float(item['approximate_kv_cache_bytes'])
            for item in mha_items
            if _is_finite_number(item.get('approximate_kv_cache_bytes'))
        ]
        gqa_cache_values = [
            float(item['approximate_kv_cache_bytes'])
            for item in gqa_items
            if _is_finite_number(item.get('approximate_kv_cache_bytes'))
        ]
        if mha_cache_values and gqa_cache_values:
            mha_cache = min(mha_cache_values)
            best_gqa_cache = min(gqa_cache_values)
            lower_cache = best_gqa_cache < mha_cache
            detail = f'best GQA {best_gqa_cache:g} < MHA {mha_cache:g}'
        add('attention:gqa_reduces_kv_cache', lower_cache, detail)

    passed = all(check['passed'] for check in checks)
    return {
        'schema_version': GATE_SCHEMA_VERSION,
        'summary_path': str(summary_path),
        'passed': passed,
        'checks': checks,
    }
