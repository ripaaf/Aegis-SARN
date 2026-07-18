'''Phase 4 matched MHA/GQA experiments for the SARN-Dense control model.'''

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from aegis_sarn.config import (
    ArtifactConfig,
    DecodingConfig,
    ModelConfig,
    SeedConfig,
    TrainingConfig,
)
from aegis_sarn.eval import benchmark_generation, evaluate_toy
from aegis_sarn.phase3 import build_common_summary
from aegis_sarn.registry import record_entry, record_manifest
from aegis_sarn.sarn.checkpoint import load_checkpoint
from aegis_sarn.sarn.training import run_smoke_training
from aegis_sarn.utils import config_hash, write_json

ATTENTION_SWEEP_SCHEMA_VERSION = 'aegis.attention_sweep/v1'
ATTENTION_COMPARISON_SCHEMA_VERSION = 'aegis.attention_comparison/v1'


@dataclass(frozen=True, slots=True)
class AttentionVariant:
    name: str
    attention_type: Literal['mha', 'gqa']
    n_kv_heads: int

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=256,
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            attention_type=self.attention_type,
            n_kv_heads=self.n_kv_heads,
            ffn_hidden_dim=96,
        )

    def to_dict(self) -> dict[str, Any]:
        config = self.model_config()
        return {
            'name': self.name,
            'attention_type': config.attention_type,
            'n_heads': config.n_heads,
            'n_kv_heads': config.resolved_n_kv_heads,
            'kv_group_size': config.kv_group_size,
        }


def default_attention_variants() -> list[AttentionVariant]:
    return [
        AttentionVariant('mha', 'mha', 4),
        AttentionVariant('gqa-kv2', 'gqa', 2),
        AttentionVariant('gqa-kv1', 'gqa', 1),
    ]


def _attention_sweep_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# SARN-Dense Phase 4 Attention Sweep',
        '',
        f"Generated at: {summary['timestamp']}",
        '',
        'Matched tiny SARN-Dense runs compare MHA with experimental GQA. '
        'This does not implement SARN-Hybrid.',
        '',
        '| Variant | Type | Q Heads | KV Heads | Group | Eval Loss | Token Accuracy | '
        'Tokens/Sec | Params | KV Cache Bytes | Status |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for item in summary['results']:
        lines.append(
            '| {config_name} | {attention_type} | {n_heads} | {n_kv_heads} | '
            '{kv_group_size} | {eval_loss:.6g} | {token_accuracy:.4f} | '
            '{tokens_per_second:.3f} | {parameter_count} | '
            '{approximate_kv_cache_bytes} | {status} |'.format(**item)
        )
    lines.extend(
        [
            '',
            '## Limits',
            '',
            '- Measurements are local CPU results on generated toy data.',
            '- GQA is an experimental attention option within SARN-Dense, not SARN-Hybrid.',
            '- Speed can be noisy; cache-size differences follow the stored KV tensor shapes.',
        ]
    )
    return '\n'.join(lines) + '\n'


def run_attention_sweep(
    output_dir: Path,
    device: str = 'cpu',
    seed: int = 123,
    train_steps: int = 8,
    batch_size: int = 2,
    sequence_length: int = 16,
    max_new_tokens: int = 2,
    bench_repeats: int = 1,
) -> dict[str, Any]:
    if train_steps <= 0:
        raise ValueError('train_steps must be positive')
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / 'runs' / 'registry.json'
    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    variants = default_attention_variants()
    results: list[dict[str, Any]] = []

    for variant in variants:
        variant_started = time.perf_counter()
        variant_dir = output_dir / variant.name
        model_config = variant.model_config()
        if sequence_length > model_config.max_seq_len:
            raise ValueError('sequence_length exceeds the fixed attention sweep context')
        training_config = TrainingConfig(
            learning_rate=1.0e-2,
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_steps=train_steps,
            device=device,
        )
        seed_config = SeedConfig(seed=seed)
        command_args = {
            'parent_command': 'sweep-attention',
            'attention_variant': variant.name,
            'attention_type': model_config.attention_type,
            'n_heads': model_config.n_heads,
            'n_kv_heads': model_config.resolved_n_kv_heads,
            'kv_group_size': model_config.kv_group_size,
            'seed': seed,
            'device': device,
        }
        train_result = run_smoke_training(
            model_config=model_config,
            training_config=training_config,
            seed_config=seed_config,
            artifact_config=ArtifactConfig(output_dir=variant_dir / 'train'),
            command_args=command_args,
        )
        record_manifest(registry_path, train_result.manifest_path)
        checkpoint = train_result.checkpoint_path
        checkpoint_artifacts = {'checkpoint': str(checkpoint)}
        decoding_config = DecodingConfig(
            strategy='greedy',
            max_new_tokens=max_new_tokens,
            use_kv_cache=True,
            seed=seed,
        )

        model = load_checkpoint(checkpoint, map_location=device).model
        eval_result = evaluate_toy(
            model=model,
            output_dir=variant_dir / 'eval',
            seed_config=seed_config,
            decoding_config=decoding_config,
            device=device,
            batch_size=batch_size,
            sequence_length=sequence_length,
            command_args=command_args,
            artifacts=checkpoint_artifacts,
        )
        record_manifest(registry_path, eval_result.manifest_path)

        model = load_checkpoint(checkpoint, map_location=device).model
        bench_result = benchmark_generation(
            model=model,
            output_dir=variant_dir / 'bench',
            seed_config=seed_config,
            decoding_config=decoding_config,
            device=device,
            prompt_length=min(8, sequence_length),
            repeats=bench_repeats,
            command_args=command_args,
            artifacts=checkpoint_artifacts,
        )
        record_manifest(registry_path, bench_result.manifest_path)

        eval_metrics = eval_result.metrics
        bench_metrics = bench_result.metrics
        results.append(
            {
                'config_name': variant.name,
                'attention_type': model_config.attention_type,
                'n_heads': model_config.n_heads,
                'n_kv_heads': model_config.resolved_n_kv_heads,
                'kv_group_size': model_config.kv_group_size,
                'model_config': model_config.to_dict(),
                'train_loss_start': train_result.initial_loss,
                'train_loss_end': train_result.final_loss,
                'eval_loss': float(eval_metrics['validation_loss']),
                'perplexity': float(eval_metrics['perplexity']),
                'token_accuracy': float(eval_metrics['token_accuracy']),
                'tokens_per_second': float(bench_metrics['tokens_per_second']),
                'parameter_count': int(bench_metrics['parameter_count']),
                'active_parameter_count': int(bench_metrics['active_parameter_count']),
                'parameter_memory_bytes': int(bench_metrics['parameter_memory_bytes']),
                'approximate_kv_cache_bytes': int(
                    bench_metrics['approximate_kv_cache_bytes']
                ),
                'kv_cache_bytes_per_token': int(
                    bench_metrics['kv_cache_bytes_per_token']
                ),
                'runtime_duration_ms': (time.perf_counter() - variant_started) * 1000.0,
                'seed': seed,
                'config_hash': config_hash(model_config.to_dict()),
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
        'completed_count': sum(item['status'] == 'completed' for item in results),
        'mha_count': sum(item['attention_type'] == 'mha' for item in results),
        'gqa_count': sum(item['attention_type'] == 'gqa' for item in results),
        'minimum_kv_cache_bytes': min(
            item['approximate_kv_cache_bytes'] for item in results
        ),
    }
    artifacts = {
        'summary_json': str(output_dir / 'attention-sweep-summary.json'),
        'summary_markdown': str(output_dir / 'attention-sweep-summary.md'),
        'registry': str(registry_path),
    }
    summary = build_common_summary(
        command='sweep-attention',
        run_id=run_id,
        created_at=created_at,
        seed=seed,
        device=device,
        configuration={
            'variants': [variant.to_dict() for variant in variants],
            'train_steps': train_steps,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'max_new_tokens': max_new_tokens,
            'bench_repeats': bench_repeats,
        },
        metrics=metrics,
        artifacts=artifacts,
        schema_version=ATTENTION_SWEEP_SCHEMA_VERSION,
    )
    summary['results'] = results
    write_json(output_dir / 'attention-sweep-summary.json', summary)
    (output_dir / 'attention-sweep-summary.md').write_text(
        _attention_sweep_markdown(summary), encoding='utf-8', newline='\n'
    )
    record_entry(
        registry_path,
        {
            'run_id': run_id,
            'command_name': 'sweep-attention',
            'timestamp': created_at,
            'git_commit': summary['git_commit_hash'],
            'package_version': summary['package_version'],
            'config_hash': summary['config_hash'],
            'checkpoint_path': None,
            'manifest_path': str(output_dir / 'attention-sweep-summary.json'),
            'metrics_summary': metrics,
            'device': device,
            'seed': seed,
            'status': 'completed',
        },
    )
    return summary


def _best_by(
    results: list[dict[str, Any]], key: str, reverse: bool = False
) -> dict[str, Any]:
    return sorted(results, key=lambda item: float(item[key]), reverse=reverse)[0]


def _attention_comparison_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# SARN-Dense Attention Comparison',
        '',
        f"Generated at: {summary['timestamp']}",
        '',
        'This report compares matched MHA and experimental GQA SARN-Dense variants.',
        '',
        '| Criterion | Variant | Score |',
        '|---|---|---:|',
    ]
    for criterion, winner in summary['winners'].items():
        lines.append(
            f"| {criterion} | {winner['config_name']} | {winner['score']:.6g} |"
        )
    lines.extend(
        [
            '',
            '- Rankings describe tiny generated tasks and local benchmark runs only.',
            '- The balanced score is a transparent heuristic, not a capability claim.',
            '- No SARN-Hybrid path is implemented or evaluated.',
        ]
    )
    return '\n'.join(lines) + '\n'


def compare_attention(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    sweep_path = input_dir / 'attention-sweep-summary.json'
    sweep = json.loads(sweep_path.read_text(encoding='utf-8'))
    results = sweep.get('results') or []
    if not results:
        raise ValueError('attention summary does not contain results')

    best_loss = _best_by(results, 'eval_loss')
    best_accuracy = _best_by(results, 'token_accuracy', reverse=True)
    best_speed = _best_by(results, 'tokens_per_second', reverse=True)
    best_cache = _best_by(results, 'approximate_kv_cache_bytes')
    best_quality_per_cache = max(
        results,
        key=lambda item: float(item['token_accuracy'])
        / float(item['approximate_kv_cache_bytes']),
    )
    max_accuracy = max(float(item['token_accuracy']) for item in results)
    max_inverse_loss = max(1.0 / max(float(item['eval_loss']), 1.0e-12) for item in results)
    max_speed = max(float(item['tokens_per_second']) for item in results)
    max_cache_efficiency = max(
        1.0 / float(item['approximate_kv_cache_bytes']) for item in results
    )

    def balanced_score(item: dict[str, Any]) -> float:
        return (
            float(item['token_accuracy']) / max(max_accuracy, 1.0e-12)
            + (1.0 / max(float(item['eval_loss']), 1.0e-12)) / max_inverse_loss
            + float(item['tokens_per_second']) / max(max_speed, 1.0e-12)
            + (1.0 / float(item['approximate_kv_cache_bytes']))
            / max_cache_efficiency
        ) / 4.0

    balanced = max(results, key=balanced_score)
    winners = {
        'best_eval_loss': {
            'config_name': best_loss['config_name'],
            'score': float(best_loss['eval_loss']),
        },
        'best_token_accuracy': {
            'config_name': best_accuracy['config_name'],
            'score': float(best_accuracy['token_accuracy']),
        },
        'best_tokens_per_second': {
            'config_name': best_speed['config_name'],
            'score': float(best_speed['tokens_per_second']),
        },
        'lowest_kv_cache_bytes': {
            'config_name': best_cache['config_name'],
            'score': float(best_cache['approximate_kv_cache_bytes']),
        },
        'best_quality_per_kv_cache_byte': {
            'config_name': best_quality_per_cache['config_name'],
            'score': float(best_quality_per_cache['token_accuracy'])
            / float(best_quality_per_cache['approximate_kv_cache_bytes']),
        },
        'best_balanced_attention': {
            'config_name': balanced['config_name'],
            'score': balanced_score(balanced),
        },
    }
    created_at = datetime.now(timezone.utc).isoformat()
    artifacts = {
        'comparison_json': str(output_dir / 'attention-comparison.json'),
        'comparison_markdown': str(output_dir / 'attention-comparison.md'),
    }
    summary = build_common_summary(
        command='compare-attention',
        run_id=str(uuid4()),
        created_at=created_at,
        seed=int(sweep.get('seed') or 0),
        device=str(sweep.get('device') or 'unknown'),
        configuration={
            'input': str(input_dir),
            'source_config_hash': sweep.get('config_hash'),
        },
        metrics={'config_count': len(results)},
        artifacts=artifacts,
        schema_version=ATTENTION_COMPARISON_SCHEMA_VERSION,
    )
    summary['winners'] = winners
    summary['results'] = results
    summary['notes'] = [
        'All variants share model width, depth, FFN size, context, data, and seed.',
        'Only attention type and KV head count differ.',
        'Quality per KV-cache byte is toy token accuracy divided by estimated cache bytes.',
        'SARN-Hybrid and later modules remain unimplemented.',
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'attention-comparison.json', summary)
    (output_dir / 'attention-comparison.md').write_text(
        _attention_comparison_markdown(summary), encoding='utf-8', newline='\n'
    )
    return summary
