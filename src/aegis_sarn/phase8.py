'''Phase 8 matched sparse-expert routing experiments in SARN-Dense.'''

from __future__ import annotations

import json
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
from aegis_sarn.eval import benchmark_generation, evaluate_tasks, evaluate_toy
from aegis_sarn.phase3 import build_common_summary
from aegis_sarn.registry import record_entry, record_manifest
from aegis_sarn.sarn.checkpoint import load_checkpoint
from aegis_sarn.sarn.data import TOY_TASK_NAMES
from aegis_sarn.sarn.training import run_smoke_training
from aegis_sarn.utils import config_hash, write_json

EXPERT_SWEEP_SCHEMA_VERSION = 'aegis.expert_sweep/v1'
EXPERT_COMPARISON_SCHEMA_VERSION = 'aegis.expert_comparison/v1'


@dataclass(frozen=True, slots=True)
class ExpertVariant:
    name: str
    experts_enabled: bool
    num_experts: int = 0
    top_k: int = 1
    replaces_ffn: bool = True

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=256,
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_hidden_dim=96,
            experts_enabled=self.experts_enabled,
            expert_num_experts=self.num_experts,
            expert_top_k=self.top_k,
            expert_capacity_factor=1.0,
            expert_hidden_dim=96 if self.experts_enabled else None,
            expert_router_noise=0.0,
            expert_load_balance_weight=(
                0.01
                if self.experts_enabled and self.replaces_ffn
                else 0.0
            ),
            expert_use_shared_expert=False,
            expert_layer_frequency=1,
            expert_replaces_ffn=self.replaces_ffn,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'experts_enabled': self.experts_enabled,
            'expert_num_experts': self.num_experts if self.experts_enabled else 0,
            'expert_top_k': self.top_k if self.experts_enabled else 0,
            'expert_capacity_factor': 1.0 if self.experts_enabled else 0.0,
            'expert_hidden_dim': 96 if self.experts_enabled else 0,
            'expert_router_noise': 0.0,
            'expert_load_balance_weight': (
                0.01
                if self.experts_enabled and self.replaces_ffn
                else 0.0
            ),
            'expert_use_shared_expert': False,
            'expert_layer_frequency': 1 if self.experts_enabled else 0,
            'expert_replaces_ffn': (
                self.replaces_ffn if self.experts_enabled else False
            ),
            'expert_variant_name': self.name,
        }


def default_expert_variants() -> list[ExpertVariant]:
    return [
        ExpertVariant('dense-control', False),
        ExpertVariant('expert-null', True, 2, 1, False),
        ExpertVariant('experts-2-top1', True, 2, 1),
        ExpertVariant('experts-4-top1', True, 4, 1),
        ExpertVariant('experts-4-top2', True, 4, 2),
    ]


def _expert_sweep_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# SARN-Dense Phase 8 Expert Sweep',
        '',
        'Generated at: {}'.format(summary['timestamp']),
        '',
        '| Variant | Experts | Top-k | Replaces FFN | Eval Loss | Accuracy | '
        'Task Accuracy | Tokens/Sec | Total Params | Active Params | '
        'Expert Params | Active Expert Params | Active Experts | Entropy | '
        'Balance | Max Load | Min Load | Dropped | Status |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
        '---:|---:|---:|---:|---:|---|',
    ]
    for item in summary['results']:
        lines.append(
            '| {config_name} | {expert_num_experts} | {expert_top_k} | '
            '{expert_replaces_ffn} | {eval_loss:.6g} | '
            '{token_accuracy:.4f} | {task_token_accuracy:.4f} | '
            '{tokens_per_second:.3f} | {parameter_count} | '
            '{active_parameter_count} | {expert_parameter_count} | '
            '{expert_active_parameter_count} | {expert_active_experts} | '
            '{expert_router_entropy:.4f} | '
            '{expert_load_balance_score:.4f} | '
            '{expert_max_load_fraction:.4f} | '
            '{expert_min_load_fraction:.4f} | '
            '{expert_dropped_token_fraction:.4f} | {status} |'.format(**item)
        )
    lines.extend(
        [
            '',
            '## Limits',
            '',
            '- Results use generated toy tasks and tiny local CPU budgets.',
            '- The null control keeps the normal dense FFN and performs no expert routing.',
            '- Capacity metadata is recorded, but this prototype does not drop tokens.',
            '- Routing labels do not demonstrate expert specialization.',
            '- This is neither distributed/production MoE nor SARN-Hybrid.',
            '- No serving-time weight mutation or persistent state exists.',
        ]
    )
    return '\n'.join(lines) + '\n'


def run_expert_sweep(
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
    variants = default_expert_variants()
    results: list[dict[str, Any]] = []

    for variant in variants:
        variant_started = time.perf_counter()
        variant_dir = output_dir / variant.name
        model_config = variant.model_config()
        if sequence_length > model_config.max_seq_len:
            raise ValueError('sequence_length exceeds the expert sweep context')
        training_config = TrainingConfig(
            learning_rate=1.0e-2,
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_steps=train_steps,
            device=device,
        )
        seed_config = SeedConfig(seed=seed)
        command_args = {
            'parent_command': 'sweep-experts',
            **variant.to_dict(),
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
        task_result = evaluate_tasks(
            model=model,
            output_dir=variant_dir / 'tasks',
            seed_config=seed_config,
            decoding_config=decoding_config,
            device=device,
            batch_size=batch_size,
            sequence_length=sequence_length,
            task_names=TOY_TASK_NAMES,
            split='validation',
            command_args=command_args,
            artifacts=checkpoint_artifacts,
        )
        record_manifest(registry_path, task_result.manifest_path)

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
        task_metrics = task_result.metrics
        bench_metrics = bench_result.metrics
        results.append(
            {
                'config_name': variant.name,
                **variant.to_dict(),
                'expert_active_experts': int(
                    bench_metrics['expert_active_experts']
                ),
                'expert_router_entropy': float(
                    bench_metrics['expert_router_entropy']
                ),
                'expert_load_balance_score': float(
                    bench_metrics['expert_load_balance_score']
                ),
                'expert_max_load_fraction': float(
                    bench_metrics['expert_max_load_fraction']
                ),
                'expert_min_load_fraction': float(
                    bench_metrics['expert_min_load_fraction']
                ),
                'expert_dropped_token_fraction': float(
                    bench_metrics['expert_dropped_token_fraction']
                ),
                'expert_parameter_count': int(
                    bench_metrics['expert_parameter_count']
                ),
                'expert_active_parameter_count': int(
                    bench_metrics['expert_active_parameter_count']
                ),
                'expert_layer_count': int(bench_metrics['expert_layer_count']),
                'model_config': model_config.to_dict(),
                'train_loss_start': train_result.initial_loss,
                'train_loss_end': train_result.final_loss,
                'eval_loss': float(eval_metrics['validation_loss']),
                'perplexity': float(eval_metrics['perplexity']),
                'token_accuracy': float(eval_metrics['token_accuracy']),
                'task_eval_loss': float(
                    task_metrics['aggregate_validation_loss']
                ),
                'task_perplexity': float(task_metrics['aggregate_perplexity']),
                'task_token_accuracy': float(
                    task_metrics['aggregate_token_accuracy']
                ),
                'task_metrics': task_metrics['tasks'],
                'tokens_per_second': float(bench_metrics['tokens_per_second']),
                'parameter_count': int(bench_metrics['parameter_count']),
                'active_parameter_count': int(
                    bench_metrics['active_parameter_count']
                ),
                'parameter_memory_bytes': int(
                    bench_metrics['parameter_memory_bytes']
                ),
                'runtime_duration_ms': (
                    time.perf_counter() - variant_started
                )
                * 1000.0,
                'seed': seed,
                'config_hash': config_hash(model_config.to_dict()),
                'checkpoint_path': str(checkpoint),
                'manifest_path': str(eval_result.manifest_path),
                'manifest_paths': {
                    'train': str(train_result.manifest_path),
                    'eval': str(eval_result.manifest_path),
                    'tasks': str(task_result.manifest_path),
                    'bench': str(bench_result.manifest_path),
                },
                'device': device,
                'status': 'completed',
            }
        )

    metrics = {
        'config_count': len(results),
        'completed_count': sum(
            item['status'] == 'completed' for item in results
        ),
        'dense_control_count': sum(
            item['config_name'] == 'dense-control' for item in results
        ),
        'expert_null_count': sum(
            item['config_name'] == 'expert-null' for item in results
        ),
        'experts_enabled_count': sum(
            item['experts_enabled'] for item in results
        ),
        'routed_variant_count': sum(
            item['expert_replaces_ffn'] for item in results
        ),
    }
    artifacts = {
        'summary_json': str(output_dir / 'expert-sweep-summary.json'),
        'summary_markdown': str(output_dir / 'expert-sweep-summary.md'),
        'registry': str(registry_path),
    }
    summary = build_common_summary(
        command='sweep-experts',
        run_id=run_id,
        created_at=created_at,
        seed=seed,
        device=device,
        configuration={
            'variants': [variant.to_dict() for variant in variants],
            'tasks': list(TOY_TASK_NAMES),
            'train_steps': train_steps,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'max_new_tokens': max_new_tokens,
            'bench_repeats': bench_repeats,
        },
        metrics=metrics,
        artifacts=artifacts,
        schema_version=EXPERT_SWEEP_SCHEMA_VERSION,
    )
    summary.update(
        {
            'experts_enabled': None,
            'expert_num_experts': None,
            'expert_top_k': None,
            'expert_capacity_factor': None,
            'expert_hidden_dim': None,
            'expert_router_noise': None,
            'expert_load_balance_weight': None,
            'expert_use_shared_expert': None,
            'expert_layer_frequency': None,
            'expert_replaces_ffn': None,
            'expert_variant_name': 'expert-sweep',
            'results': results,
            'limitations': [
                'SARN-Dense remains the default and only complete model path.',
                'Toy results do not establish expert specialization.',
                'This prototype has no distributed routing or token dropping.',
                'It is not production MoE or SARN-Hybrid.',
            ],
        }
    )
    write_json(output_dir / 'expert-sweep-summary.json', summary)
    (output_dir / 'expert-sweep-summary.md').write_text(
        _expert_sweep_markdown(summary), encoding='utf-8', newline='\n'
    )
    record_entry(
        registry_path,
        {
            'run_id': run_id,
            'command_name': 'sweep-experts',
            'timestamp': created_at,
            'git_commit': summary['git_commit_hash'],
            'package_version': summary['package_version'],
            'config_hash': summary['config_hash'],
            'checkpoint_path': None,
            'manifest_path': str(output_dir / 'expert-sweep-summary.json'),
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


def _balanced_score(
    item: dict[str, Any], results: list[dict[str, Any]]
) -> float:
    max_accuracy = max(float(value['token_accuracy']) for value in results)
    max_task_accuracy = max(
        float(value['task_token_accuracy']) for value in results
    )
    max_inverse_loss = max(
        1.0 / max(float(value['eval_loss']), 1.0e-12) for value in results
    )
    max_speed = max(float(value['tokens_per_second']) for value in results)
    return (
        float(item['token_accuracy']) / max(max_accuracy, 1.0e-12)
        + float(item['task_token_accuracy'])
        / max(max_task_accuracy, 1.0e-12)
        + (1.0 / max(float(item['eval_loss']), 1.0e-12)) / max_inverse_loss
        + float(item['tokens_per_second']) / max(max_speed, 1.0e-12)
        + float(item['expert_load_balance_score'])
    ) / 5.0


def _expert_comparison_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# SARN-Dense Expert Comparison',
        '',
        'Generated at: {}'.format(summary['timestamp']),
        '',
        '| Criterion | Variant | Score |',
        '|---|---|---:|',
    ]
    for criterion, winner in summary['winners'].items():
        lines.append(
            '| {} | {} | {:.6g} |'.format(
                criterion, winner['config_name'], winner['score']
            )
        )
    lines.extend(
        [
            '',
            '## Control deltas',
            '',
            '| Expert Variant | Versus | Eval-loss Delta | Accuracy Delta | '
            'Task Accuracy Delta |',
            '|---|---|---:|---:|---:|',
        ]
    )
    for item in summary['control_comparisons']:
        lines.append(
            '| {expert_variant} | {control} | {eval_loss_delta:.6g} | '
            '{token_accuracy_delta:.6g} | '
            '{task_accuracy_delta:.6g} |'.format(**item)
        )
    lines.extend(
        [
            '',
            '- Rankings cover generated toy data at tiny CPU budgets only.',
            '- Routing balance does not demonstrate expert specialization.',
            '- Active parameters are a per-token top-k estimate; total parameters include all experts.',
            '- No distributed routing, custom kernel, or token dropping is implemented.',
            '- This is not full production MoE or SARN-Hybrid.',
            '- The balanced score is a heuristic, not an acceptance claim.',
        ]
    )
    return '\n'.join(lines) + '\n'


def compare_experts(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    sweep_path = input_dir / 'expert-sweep-summary.json'
    sweep = json.loads(sweep_path.read_text(encoding='utf-8'))
    results = sweep.get('results') or []
    if not results:
        raise ValueError('expert summary does not contain results')
    routed_results = [
        item
        for item in results
        if item.get('experts_enabled') and item.get('expert_replaces_ffn')
    ]
    if not routed_results:
        raise ValueError('expert summary has no routed expert results')

    best_loss = _best_by(results, 'eval_loss')
    best_perplexity = _best_by(results, 'perplexity')
    best_accuracy = _best_by(results, 'token_accuracy', reverse=True)
    best_task_accuracy = _best_by(
        results, 'task_token_accuracy', reverse=True
    )
    best_speed = _best_by(results, 'tokens_per_second', reverse=True)
    best_quality_active = max(
        results,
        key=lambda item: float(item['task_token_accuracy'])
        / float(item['active_parameter_count']),
    )
    best_quality_total = max(
        results,
        key=lambda item: float(item['task_token_accuracy'])
        / float(item['parameter_count']),
    )
    best_balance = _best_by(
        routed_results, 'expert_load_balance_score', reverse=True
    )
    balanced_expert = max(
        routed_results, key=lambda item: _balanced_score(item, results)
    )
    winners = {
        'best_eval_loss': {
            'config_name': best_loss['config_name'],
            'score': float(best_loss['eval_loss']),
        },
        'best_perplexity': {
            'config_name': best_perplexity['config_name'],
            'score': float(best_perplexity['perplexity']),
        },
        'best_token_accuracy': {
            'config_name': best_accuracy['config_name'],
            'score': float(best_accuracy['token_accuracy']),
        },
        'best_task_accuracy': {
            'config_name': best_task_accuracy['config_name'],
            'score': float(best_task_accuracy['task_token_accuracy']),
        },
        'best_tokens_per_second': {
            'config_name': best_speed['config_name'],
            'score': float(best_speed['tokens_per_second']),
        },
        'best_quality_per_active_parameter': {
            'config_name': best_quality_active['config_name'],
            'score': float(best_quality_active['task_token_accuracy'])
            / float(best_quality_active['active_parameter_count']),
        },
        'best_quality_per_total_parameter': {
            'config_name': best_quality_total['config_name'],
            'score': float(best_quality_total['task_token_accuracy'])
            / float(best_quality_total['parameter_count']),
        },
        'best_routing_balance': {
            'config_name': best_balance['config_name'],
            'score': float(best_balance['expert_load_balance_score']),
        },
        'best_balanced_expert': {
            'config_name': balanced_expert['config_name'],
            'score': _balanced_score(balanced_expert, results),
        },
    }
    controls = {
        str(item['config_name']): item
        for item in results
        if item['config_name'] in ('dense-control', 'expert-null')
    }
    control_comparisons: list[dict[str, Any]] = []
    for expert_item in routed_results:
        for control_name, control in controls.items():
            control_comparisons.append(
                {
                    'expert_variant': expert_item['config_name'],
                    'control': control_name,
                    'eval_loss_delta': float(expert_item['eval_loss'])
                    - float(control['eval_loss']),
                    'token_accuracy_delta': float(
                        expert_item['token_accuracy']
                    )
                    - float(control['token_accuracy']),
                    'task_accuracy_delta': float(
                        expert_item['task_token_accuracy']
                    )
                    - float(control['task_token_accuracy']),
                }
            )

    created_at = datetime.now(timezone.utc).isoformat()
    artifacts = {
        'comparison_json': str(output_dir / 'expert-comparison.json'),
        'comparison_markdown': str(output_dir / 'expert-comparison.md'),
    }
    summary = build_common_summary(
        command='compare-experts',
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
        schema_version=EXPERT_COMPARISON_SCHEMA_VERSION,
    )
    summary.update(
        {
            'experts_enabled': None,
            'expert_num_experts': None,
            'expert_top_k': None,
            'expert_capacity_factor': None,
            'expert_hidden_dim': None,
            'expert_router_noise': None,
            'expert_load_balance_weight': None,
            'expert_use_shared_expert': None,
            'expert_layer_frequency': None,
            'expert_replaces_ffn': None,
            'expert_variant_name': 'expert-comparison',
            'winners': winners,
            'control_comparisons': control_comparisons,
            'results': results,
            'notes': [
                'Toy metrics do not demonstrate expert specialization.',
                'Active parameter counts estimate per-token top-k execution.',
                'No distributed routing, token dropping, or custom kernel exists.',
                'This is not production MoE, SARN-Hybrid, or a Phase 9 module.',
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'expert-comparison.json', summary)
    (output_dir / 'expert-comparison.md').write_text(
        _expert_comparison_markdown(summary), encoding='utf-8', newline='\n'
    )
    return summary
