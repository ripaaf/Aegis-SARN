'''Phase 6 matched graph-message-passing experiments within SARN-Dense.'''

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
from aegis_sarn.sarn.data import GRAPH_TASK_NAMES
from aegis_sarn.sarn.training import run_smoke_training
from aegis_sarn.utils import config_hash, write_json

GRAPH_SWEEP_SCHEMA_VERSION = 'aegis.graph_sweep/v1'
GRAPH_COMPARISON_SCHEMA_VERSION = 'aegis.graph_comparison/v1'


@dataclass(frozen=True, slots=True)
class GraphVariant:
    name: str
    workspace_enabled: bool
    graph_enabled: bool
    edge_mode: str
    num_cycles: int
    num_slots: int = 4
    top_k: int | None = None

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=256,
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_hidden_dim=96,
            workspace_enabled=self.workspace_enabled,
            workspace_num_slots=self.num_slots if self.workspace_enabled else 0,
            graph_enabled=self.graph_enabled,
            graph_num_cycles=self.num_cycles,
            graph_edge_mode=self.edge_mode,  # type: ignore[arg-type]
            graph_top_k=self.top_k,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'graph_variant_name': self.name,
            'workspace_variant_name': self.name,
            'workspace_enabled': self.workspace_enabled,
            'workspace_num_slots': (
                self.num_slots if self.workspace_enabled else 0
            ),
            'workspace_gated_writeback': self.workspace_enabled,
            'graph_enabled': self.graph_enabled,
            'graph_num_cycles': self.num_cycles if self.graph_enabled else 0,
            'graph_edge_mode': self.edge_mode if self.graph_enabled else 'none',
            'graph_top_k': self.top_k if self.graph_enabled else None,
            'graph_gated_update': self.graph_enabled,
        }


def default_graph_variants() -> list[GraphVariant]:
    return [
        GraphVariant('dense-control', False, False, 'none', 0),
        GraphVariant('workspace-control', True, False, 'none', 0),
        GraphVariant('graph-null', True, True, 'none', 1),
        GraphVariant('graph-identity', True, True, 'frozen_identity', 1),
        GraphVariant('graph-dense-cycle1', True, True, 'learned_dense', 1),
        GraphVariant('graph-dense-cycle2', True, True, 'learned_dense', 2),
    ]


def _graph_sweep_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# SARN-Dense Phase 6 Graph Sweep',
        '',
        'Generated at: {}'.format(summary['timestamp']),
        '',
        'Matched tiny runs compare dense, workspace-only, null-edge, frozen-edge, '
        'and learned-edge controls.',
        '',
        '| Variant | Workspace | Graph | Edge Mode | Cycles | Eval Loss | '
        'Token Accuracy | Structural Accuracy | Tokens/Sec | Params | '
        'Active Params | Graph Params | Gate Mean | Message Norm | Status |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for item in summary['results']:
        lines.append(
            '| {config_name} | {workspace_enabled} | {graph_enabled} | '
            '{graph_edge_mode} | {graph_num_cycles} | {eval_loss:.6g} | '
            '{token_accuracy:.4f} | {task_token_accuracy:.4f} | '
            '{tokens_per_second:.3f} | {parameter_count} | '
            '{active_parameter_count} | {graph_parameter_count} | '
            '{graph_gate_mean:.6g} | {graph_message_norm:.6g} | {status} |'.format(
                **item
            )
        )
    lines.extend(
        [
            '',
            '## Limits',
            '',
            '- Results use generated toy and structural data at tiny budgets.',
            '- Local CPU throughput is hardware-specific and noisy.',
            '- Latent slots are tensor states, not interpretable concepts.',
            '- Graph cycles are not resettable or persistent memory.',
            '- These measurements do not demonstrate formal or human-like reasoning.',
            '- SARN-Hybrid and Phase 7+ mechanisms are not implemented here.',
        ]
    )
    return '\n'.join(lines) + '\n'


def run_graph_sweep(
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
    variants = default_graph_variants()
    results: list[dict[str, Any]] = []

    for variant in variants:
        variant_started = time.perf_counter()
        variant_dir = output_dir / variant.name
        model_config = variant.model_config()
        if sequence_length > model_config.max_seq_len:
            raise ValueError('sequence_length exceeds the fixed graph sweep context')
        training_config = TrainingConfig(
            learning_rate=1.0e-2,
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_steps=train_steps,
            device=device,
        )
        seed_config = SeedConfig(seed=seed)
        command_args = {
            'parent_command': 'sweep-graph',
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
            task_names=GRAPH_TASK_NAMES,
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
                'workspace_parameter_count': int(
                    bench_metrics['workspace_parameter_count']
                ),
                'workspace_gate_mean': float(bench_metrics['workspace_gate_mean']),
                'workspace_norm': float(bench_metrics['workspace_norm']),
                'graph_parameter_count': int(
                    bench_metrics['graph_parameter_count']
                ),
                'graph_gate_mean': float(bench_metrics['graph_gate_mean']),
                'graph_message_norm': float(
                    bench_metrics['graph_message_norm']
                ),
                'graph_slot_norm': float(bench_metrics['graph_slot_norm']),
                'model_config': model_config.to_dict(),
                'train_loss_start': train_result.initial_loss,
                'train_loss_end': train_result.final_loss,
                'eval_loss': float(eval_metrics['validation_loss']),
                'perplexity': float(eval_metrics['perplexity']),
                'token_accuracy': float(eval_metrics['token_accuracy']),
                'task_eval_loss': float(
                    task_metrics['aggregate_validation_loss']
                ),
                'task_perplexity': float(
                    task_metrics['aggregate_perplexity']
                ),
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
        'completed_count': sum(item['status'] == 'completed' for item in results),
        'dense_control_count': sum(
            item['config_name'] == 'dense-control' for item in results
        ),
        'workspace_control_count': sum(
            item['config_name'] == 'workspace-control' for item in results
        ),
        'graph_enabled_count': sum(item['graph_enabled'] for item in results),
        'graph_null_count': sum(
            item['config_name'] == 'graph-null' for item in results
        ),
        'graph_identity_count': sum(
            item['config_name'] == 'graph-identity' for item in results
        ),
    }
    artifacts = {
        'summary_json': str(output_dir / 'graph-sweep-summary.json'),
        'summary_markdown': str(output_dir / 'graph-sweep-summary.md'),
        'registry': str(registry_path),
    }
    summary = build_common_summary(
        command='sweep-graph',
        run_id=run_id,
        created_at=created_at,
        seed=seed,
        device=device,
        configuration={
            'variants': [variant.to_dict() for variant in variants],
            'structural_tasks': list(GRAPH_TASK_NAMES),
            'train_steps': train_steps,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'max_new_tokens': max_new_tokens,
            'bench_repeats': bench_repeats,
        },
        metrics=metrics,
        artifacts=artifacts,
        schema_version=GRAPH_SWEEP_SCHEMA_VERSION,
    )
    summary.update(
        {
            'workspace_enabled': None,
            'workspace_num_slots': None,
            'workspace_gated_writeback': None,
            'workspace_variant_name': 'graph-sweep',
            'graph_enabled': None,
            'graph_num_cycles': None,
            'graph_edge_mode': None,
            'graph_top_k': None,
            'graph_gated_update': None,
            'graph_variant_name': 'graph-sweep',
            'results': results,
        }
    )
    write_json(output_dir / 'graph-sweep-summary.json', summary)
    (output_dir / 'graph-sweep-summary.md').write_text(
        _graph_sweep_markdown(summary), encoding='utf-8', newline='\n'
    )
    record_entry(
        registry_path,
        {
            'run_id': run_id,
            'command_name': 'sweep-graph',
            'timestamp': created_at,
            'git_commit': summary['git_commit_hash'],
            'package_version': summary['package_version'],
            'config_hash': summary['config_hash'],
            'checkpoint_path': None,
            'manifest_path': str(output_dir / 'graph-sweep-summary.json'),
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
        + float(item['task_token_accuracy']) / max(max_task_accuracy, 1.0e-12)
        + (1.0 / max(float(item['eval_loss']), 1.0e-12)) / max_inverse_loss
        + float(item['tokens_per_second']) / max(max_speed, 1.0e-12)
    ) / 4.0


def _graph_comparison_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# SARN-Dense Graph Comparison',
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
            '| Graph Variant | Versus | Eval-loss Delta | Accuracy Delta | '
            'Structural-accuracy Delta |',
            '|---|---|---:|---:|---:|',
        ]
    )
    for item in summary['control_comparisons']:
        lines.append(
            '| {graph_variant} | {control} | {eval_loss_delta:.6g} | '
            '{token_accuracy_delta:.6g} | '
            '{task_token_accuracy_delta:.6g} |'.format(**item)
        )
    lines.extend(
        [
            '',
            '- Lower eval-loss delta is better; higher accuracy deltas are better.',
            '- Rankings cover generated toy/structural data only.',
            '- No result is evidence of formal logic or human-like reasoning.',
            '- Graph state is transient and is not resettable or persistent memory.',
            '- The balanced score is a documented heuristic, not an acceptance claim.',
        ]
    )
    return '\n'.join(lines) + '\n'


def compare_graph(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    sweep_path = input_dir / 'graph-sweep-summary.json'
    sweep = json.loads(sweep_path.read_text(encoding='utf-8'))
    results = sweep.get('results') or []
    if not results:
        raise ValueError('graph summary does not contain results')
    graph_results = [item for item in results if item.get('graph_enabled')]
    if not graph_results:
        raise ValueError('graph summary does not contain graph-enabled results')

    best_loss = _best_by(results, 'eval_loss')
    best_perplexity = _best_by(results, 'perplexity')
    best_accuracy = _best_by(results, 'token_accuracy', reverse=True)
    best_speed = _best_by(results, 'tokens_per_second', reverse=True)
    best_quality_per_parameter = max(
        results,
        key=lambda item: float(item['token_accuracy'])
        / float(item['parameter_count']),
    )
    best_quality_per_second = max(
        results,
        key=lambda item: float(item['token_accuracy'])
        * float(item['tokens_per_second']),
    )
    balanced_graph = max(
        graph_results, key=lambda item: _balanced_score(item, results)
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
        'best_tokens_per_second': {
            'config_name': best_speed['config_name'],
            'score': float(best_speed['tokens_per_second']),
        },
        'best_quality_per_parameter': {
            'config_name': best_quality_per_parameter['config_name'],
            'score': float(best_quality_per_parameter['token_accuracy'])
            / float(best_quality_per_parameter['parameter_count']),
        },
        'best_quality_per_second': {
            'config_name': best_quality_per_second['config_name'],
            'score': float(best_quality_per_second['token_accuracy'])
            * float(best_quality_per_second['tokens_per_second']),
        },
        'best_balanced_graph': {
            'config_name': balanced_graph['config_name'],
            'score': _balanced_score(balanced_graph, results),
        },
    }
    controls = {
        str(item['config_name']): item
        for item in results
        if item['config_name'] in ('dense-control', 'workspace-control')
    }
    control_comparisons: list[dict[str, Any]] = []
    for graph_item in graph_results:
        for control_name, control in controls.items():
            control_comparisons.append(
                {
                    'graph_variant': graph_item['config_name'],
                    'control': control_name,
                    'eval_loss_delta': float(graph_item['eval_loss'])
                    - float(control['eval_loss']),
                    'token_accuracy_delta': float(
                        graph_item['token_accuracy']
                    )
                    - float(control['token_accuracy']),
                    'task_token_accuracy_delta': float(
                        graph_item['task_token_accuracy']
                    )
                    - float(control['task_token_accuracy']),
                }
            )

    created_at = datetime.now(timezone.utc).isoformat()
    artifacts = {
        'comparison_json': str(output_dir / 'graph-comparison.json'),
        'comparison_markdown': str(output_dir / 'graph-comparison.md'),
    }
    summary = build_common_summary(
        command='compare-graph',
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
        schema_version=GRAPH_COMPARISON_SCHEMA_VERSION,
    )
    summary.update(
        {
            'workspace_enabled': None,
            'workspace_num_slots': None,
            'workspace_gated_writeback': None,
            'workspace_variant_name': 'graph-comparison',
            'graph_enabled': None,
            'graph_num_cycles': None,
            'graph_edge_mode': None,
            'graph_top_k': None,
            'graph_gated_update': None,
            'graph_variant_name': 'graph-comparison',
            'winners': winners,
            'control_comparisons': control_comparisons,
            'results': results,
            'notes': [
                'All variants share width, depth, attention, FFN, context, data, and seed.',
                'Graph edges and cycle count are controlled behind explicit flags.',
                'Toy metrics do not demonstrate formal or human-like reasoning.',
                'No persistent/resettable memory or Phase 7+ module is implemented.',
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'graph-comparison.json', summary)
    (output_dir / 'graph-comparison.md').write_text(
        _graph_comparison_markdown(summary), encoding='utf-8', newline='\n'
    )
    return summary
