'''Lightweight local run registry for reproducible baseline artifacts.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aegis_sarn.utils import normalize_json, write_json

REGISTRY_SCHEMA_VERSION = 'aegis.run_registry/v1'

_SUMMARY_KEYS = (
    'initial_loss',
    'final_loss',
    'evaluation_loss',
    'validation_loss',
    'perplexity',
    'token_accuracy',
    'tokens_per_second',
    'parameter_count',
    'parameter_memory_bytes',
    'approximate_kv_cache_bytes',
    'generated_tokens',
    'mean_validation_loss',
    'std_validation_loss',
    'mean_token_accuracy',
    'std_token_accuracy',
    'mean_perplexity',
    'std_perplexity',
    'config_count',
    'completed_count',
    'best_eval_loss',
    'best_token_accuracy',
    'best_tokens_per_second',
    'aggregate_validation_loss',
    'aggregate_token_accuracy',
    'aggregate_perplexity',
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in _SUMMARY_KEYS if key in metrics}


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'schema_version': REGISTRY_SCHEMA_VERSION, 'runs': {}}
    payload = _read_json(path)
    runs = payload.get('runs', {})
    if isinstance(runs, list):
        runs = {str(entry['run_id']): entry for entry in runs}
    if not isinstance(runs, dict):
        raise ValueError('registry runs must be an object keyed by run_id')
    return {
        'schema_version': payload.get('schema_version', REGISTRY_SCHEMA_VERSION),
        'runs': runs,
    }


def registry_entry_from_manifest(
    manifest: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    artifacts = manifest.get('artifacts') or {}
    runtime_config = manifest.get('runtime_config') or {}
    seed_config = manifest.get('seed_config') or {}
    device_info = manifest.get('device_info') or {}
    metrics = manifest.get('metrics') or {}
    checkpoint_path = artifacts.get('checkpoint') or artifacts.get('checkpoint_path')
    return {
        'run_id': str(manifest['run_id']),
        'command_name': str(manifest.get('command') or manifest.get('run_name') or ''),
        'timestamp': str(manifest.get('created_at') or ''),
        'git_commit': str(manifest.get('git_commit') or 'unavailable'),
        'package_version': str(manifest.get('package_version') or 'unknown'),
        'config_hash': str(manifest.get('config_hash') or ''),
        'checkpoint_path': checkpoint_path,
        'manifest_path': str(manifest_path),
        'metrics_summary': _metrics_summary(metrics),
        'device': runtime_config.get('device') or metrics.get('device') or device_info.get('device'),
        'seed': seed_config.get('seed'),
        'status': str(manifest.get('status') or 'completed'),
    }


def record_entry(registry_path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    registry = load_registry(registry_path)
    run_id = str(entry['run_id'])
    registry['schema_version'] = REGISTRY_SCHEMA_VERSION
    registry['runs'][run_id] = normalize_json(entry)
    write_json(registry_path, registry)
    return registry['runs'][run_id]


def record_manifest(
    registry_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _read_json(manifest_path) if manifest is None else manifest
    return record_entry(
        registry_path,
        registry_entry_from_manifest(payload, manifest_path),
    )


def registry_entries(registry_path: Path) -> list[dict[str, Any]]:
    registry = load_registry(registry_path)
    entries = list(registry['runs'].values())
    return sorted(entries, key=lambda entry: (entry.get('timestamp') or '', entry['run_id']))
