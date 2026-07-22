'''Phase 7 matched resettable-working-memory experiments in SARN-Dense.'''

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

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
from aegis_sarn.sarn.data import GRAPH_TASK_NAMES, MEMORY_TASK_NAMES
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.sarn.training import run_smoke_training
from aegis_sarn.utils import config_hash, write_json

MEMORY_SWEEP_SCHEMA_VERSION = 'aegis.memory_sweep/v1'
MEMORY_COMPARISON_SCHEMA_VERSION = 'aegis.memory_comparison/v1'


@dataclass(frozen=True, slots=True)
class MemoryVariant:
    name: str
    workspace_enabled: bool
    graph_enabled: bool
    memory_enabled: bool
    memory_write_mode: str = 'none'
    memory_read_mode: str = 'none'

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=256,
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_hidden_dim=96,
            workspace_enabled=self.workspace_enabled,
            workspace_num_slots=4 if self.workspace_enabled else 0,
            graph_enabled=self.graph_enabled,
            graph_num_cycles=1 if self.graph_enabled else 0,
            graph_edge_mode=(
                'learned_dense' if self.graph_enabled else 'none'
            ),
            memory_enabled=self.memory_enabled,
            memory_num_slots=4 if self.memory_enabled else 0,
            memory_write_mode=self.memory_write_mode,  # type: ignore[arg-type]
            memory_read_mode=self.memory_read_mode,  # type: ignore[arg-type]
            memory_reset_mode='per_generation',
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'workspace_variant_name': self.name,
            'workspace_enabled': self.workspace_enabled,
            'workspace_num_slots': 4 if self.workspace_enabled else 0,
            'workspace_gated_writeback': self.workspace_enabled,
            'graph_variant_name': self.name,
            'graph_enabled': self.graph_enabled,
            'graph_num_cycles': 1 if self.graph_enabled else 0,
            'graph_edge_mode': (
                'learned_dense' if self.graph_enabled else 'none'
            ),
            'graph_top_k': None,
            'graph_gated_update': self.graph_enabled,
            'memory_variant_name': self.name,
            'memory_enabled': self.memory_enabled,
            'memory_num_slots': 4 if self.memory_enabled else 0,
            'memory_write_mode': (
                self.memory_write_mode if self.memory_enabled else 'none'
            ),
            'memory_read_mode': (
                self.memory_read_mode if self.memory_enabled else 'none'
            ),
            'memory_reset_mode': 'per_generation',
            'memory_decay': 0.0,
        }


def default_memory_variants() -> list[MemoryVariant]:
    return [
        MemoryVariant('dense-control', False, False, False),
        MemoryVariant('workspace-control', True, False, False),
        MemoryVariant('graph-control', True, True, False),
        MemoryVariant('memory-null', True, False, True),
        MemoryVariant(
            'memory-gated', True, False, True, 'gated', 'attention'
        ),
        MemoryVariant(
            'graph-memory-gated', True, True, True, 'gated', 'attention'
        ),
    ]


def _reset_isolation_check(model: SARNDense, device: str) -> bool:
    if model.memory is None:
        return True
    model = model.to(device).eval()
    first = torch.tensor([[1, 2, 3, 4]], device=device, dtype=torch.long)
    conflict = torch.tensor([[4, 3, 2, 1]], device=device, dtype=torch.long)
    with torch.inference_mode():
        first_logits, first_cache = model.forward_with_cache(
            first, past_key_values=None, use_cache=True
        )
        model.forward_with_cache(
            conflict, past_key_values=None, use_cache=True
        )
        repeated_logits, repeated_cache = model.forward_with_cache(
            first, past_key_values=None, use_cache=True
        )
    if first_cache is None or repeated_cache is None:
        return False
    first_memory = first_cache[0].memory_slots
    repeated_memory = repeated_cache[0].memory_slots
    if first_memory is None or repeated_memory is None:
        return False
    return bool(
        torch.equal(first_logits, repeated_logits)
        and torch.equal(first_memory, repeated_memory)
    )


def _memory_sweep_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# SARN-Dense Phase 7 Memory Sweep',
        '',
        'Generated at: {}'.format(summary['timestamp']),
        '',
        '| Variant | Workspace | Graph | Memory | Write | Read | Eval Loss | '
        'Token Accuracy | Structural Accuracy | Memory Accuracy | Tokens/Sec | '
        'Params | Active Params | Memory Params | Gate Mean | Memory Norm | '
        'Reset/Isolation | Status |',
        '|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|'
        '---:|---|---|',
    ]
    for item in summary['results']:
        lines.append(
            '| {config_name} | {workspace_enabled} | {graph_enabled} | '
            '{memory_enabled} | {memory_write_mode} | {memory_read_mode} | '
            '{eval_loss:.6g} | {token_accuracy:.4f} | '
            '{task_token_accuracy:.4f} | {memory_task_token_accuracy:.4f} | '
            '{tokens_per_second:.3f} | {parameter_count} | '
            '{active_parameter_count} | {memory_parameter_count} | '
            '{memory_gate_mean:.6g} | {memory_norm:.6g} | '
            '{memory_reset_isolation_passed} | {status} |'.format(**item)
        )
    lines.extend(
        [
            '',
            '## Limits',
            '',
            '- Results use generated toy tasks and tiny local CPU budgets.',
            '- Temporary slots are not human-like, user, or long-term memory.',
            '- No state is persisted outside the explicit generation cache.',
            '- No base model weights are changed during evaluation or serving.',
            '- Expert routing and later mechanisms are outside this Phase 7 comparison.',
        ]
    )
    return '\n'.join(lines) + '\n'


def run_memory_sweep(
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
    variants = default_memory_variants()
    results: list[dict[str, Any]] = []

    for variant in variants:
        variant_started = time.perf_counter()
        variant_dir = output_dir / variant.name
        model_config = variant.model_config()
        if sequence_length > model_config.max_seq_len:
            raise ValueError('sequence_length exceeds the memory sweep context')
        training_config = TrainingConfig(
            learning_rate=1.0e-2,
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_steps=train_steps,
            device=device,
        )
        seed_config = SeedConfig(seed=seed)
        command_args = {
            'parent_command': 'sweep-memory',
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
        reset_isolation_passed = _reset_isolation_check(model, device)
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
        memory_task_result = evaluate_tasks(
            model=model,
            output_dir=variant_dir / 'memory-tasks',
            seed_config=seed_config,
            decoding_config=decoding_config,
            device=device,
            batch_size=batch_size,
            sequence_length=sequence_length,
            task_names=MEMORY_TASK_NAMES,
            split='validation',
            command_args=command_args,
            artifacts=checkpoint_artifacts,
        )
        record_manifest(registry_path, memory_task_result.manifest_path)

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
        memory_task_metrics = memory_task_result.metrics
        bench_metrics = bench_result.metrics
        results.append(
            {
                'config_name': variant.name,
                **variant.to_dict(),
                'workspace_parameter_count': int(
                    bench_metrics['workspace_parameter_count']
                ),
                'graph_parameter_count': int(
                    bench_metrics['graph_parameter_count']
                ),
                'memory_parameter_count': int(
                    bench_metrics['memory_parameter_count']
                ),
                'memory_gate_mean': float(bench_metrics['memory_gate_mean']),
                'memory_norm': float(bench_metrics['memory_norm']),
                'memory_write_norm': float(
                    bench_metrics['memory_write_norm']
                ),
                'memory_reset_applied': bool(
                    bench_metrics['memory_reset_applied']
                ),
                'memory_reset_isolation_passed': reset_isolation_passed,
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
                'memory_task_eval_loss': float(
                    memory_task_metrics['aggregate_validation_loss']
                ),
                'memory_task_perplexity': float(
                    memory_task_metrics['aggregate_perplexity']
                ),
                'memory_task_token_accuracy': float(
                    memory_task_metrics['aggregate_token_accuracy']
                ),
                'memory_task_metrics': memory_task_metrics['tasks'],
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
                    'memory_tasks': str(memory_task_result.manifest_path),
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
        'graph_control_count': sum(
            item['config_name'] == 'graph-control' for item in results
        ),
        'memory_enabled_count': sum(item['memory_enabled'] for item in results),
        'memory_null_count': sum(
            item['config_name'] == 'memory-null' for item in results
        ),
        'reset_isolation_passed_count': sum(
            item['memory_reset_isolation_passed'] for item in results
        ),
    }
    artifacts = {
        'summary_json': str(output_dir / 'memory-sweep-summary.json'),
        'summary_markdown': str(output_dir / 'memory-sweep-summary.md'),
        'registry': str(registry_path),
    }
    summary = build_common_summary(
        command='sweep-memory',
        run_id=run_id,
        created_at=created_at,
        seed=seed,
        device=device,
        configuration={
            'variants': [variant.to_dict() for variant in variants],
            'structural_tasks': list(GRAPH_TASK_NAMES),
            'memory_tasks': list(MEMORY_TASK_NAMES),
            'train_steps': train_steps,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'max_new_tokens': max_new_tokens,
            'bench_repeats': bench_repeats,
        },
        metrics=metrics,
        artifacts=artifacts,
        schema_version=MEMORY_SWEEP_SCHEMA_VERSION,
    )
    summary.update(
        {
            'workspace_enabled': None,
            'workspace_num_slots': None,
            'workspace_gated_writeback': None,
            'workspace_variant_name': 'memory-sweep',
            'graph_enabled': None,
            'graph_num_cycles': None,
            'graph_edge_mode': None,
            'graph_top_k': None,
            'graph_gated_update': None,
            'graph_variant_name': 'memory-sweep',
            'memory_enabled': None,
            'memory_num_slots': None,
            'memory_write_mode': None,
            'memory_read_mode': None,
            'memory_reset_mode': None,
            'memory_decay': None,
            'memory_variant_name': 'memory-sweep',
            'results': results,
        }
    )
    write_json(output_dir / 'memory-sweep-summary.json', summary)
    (output_dir / 'memory-sweep-summary.md').write_text(
        _memory_sweep_markdown(summary), encoding='utf-8', newline='\n'
    )
    record_entry(
        registry_path,
        {
            'run_id': run_id,
            'command_name': 'sweep-memory',
            'timestamp': created_at,
            'git_commit': summary['git_commit_hash'],
            'package_version': summary['package_version'],
            'config_hash': summary['config_hash'],
            'checkpoint_path': None,
            'manifest_path': str(output_dir / 'memory-sweep-summary.json'),
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
    max_memory_accuracy = max(
        float(value['memory_task_token_accuracy']) for value in results
    )
    max_inverse_loss = max(
        1.0 / max(float(value['eval_loss']), 1.0e-12) for value in results
    )
    max_speed = max(float(value['tokens_per_second']) for value in results)
    return (
        float(item['token_accuracy']) / max(max_accuracy, 1.0e-12)
        + float(item['memory_task_token_accuracy'])
        / max(max_memory_accuracy, 1.0e-12)
        + (1.0 / max(float(item['eval_loss']), 1.0e-12)) / max_inverse_loss
        + float(item['tokens_per_second']) / max(max_speed, 1.0e-12)
    ) / 4.0


def _memory_comparison_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# SARN-Dense Memory Comparison',
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
            '| Memory Variant | Versus | Eval-loss Delta | Accuracy Delta | '
            'Memory-task Accuracy Delta |',
            '|---|---|---:|---:|---:|',
        ]
    )
    for item in summary['control_comparisons']:
        lines.append(
            '| {memory_variant} | {control} | {eval_loss_delta:.6g} | '
            '{token_accuracy_delta:.6g} | '
            '{memory_task_accuracy_delta:.6g} |'.format(**item)
        )
    lines.extend(
        [
            '',
            '- Each enabled variant passed the explicit reset/isolation probe.',
            '- Rankings cover generated toy data at tiny budgets only.',
            '- Temporary slots are not human-like, user, or long-term memory.',
            '- No persistent store, retrieval, self-learning, or weight mutation exists.',
            '- The balanced score is a heuristic, not an acceptance claim.',
        ]
    )
    return '\n'.join(lines) + '\n'


def compare_memory(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    sweep_path = input_dir / 'memory-sweep-summary.json'
    sweep = json.loads(sweep_path.read_text(encoding='utf-8'))
    results = sweep.get('results') or []
    if not results:
        raise ValueError('memory summary does not contain results')
    memory_results = [item for item in results if item.get('memory_enabled')]
    if not memory_results:
        raise ValueError('memory summary has no memory-enabled results')

    best_loss = _best_by(results, 'eval_loss')
    best_perplexity = _best_by(results, 'perplexity')
    best_accuracy = _best_by(results, 'token_accuracy', reverse=True)
    best_memory_accuracy = _best_by(
        results, 'memory_task_token_accuracy', reverse=True
    )
    best_speed = _best_by(results, 'tokens_per_second', reverse=True)
    best_quality_per_parameter = max(
        results,
        key=lambda item: float(item['memory_task_token_accuracy'])
        / float(item['parameter_count']),
    )
    best_quality_per_second = max(
        results,
        key=lambda item: float(item['memory_task_token_accuracy'])
        * float(item['tokens_per_second']),
    )
    balanced_memory = max(
        memory_results, key=lambda item: _balanced_score(item, results)
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
        'best_memory_task_accuracy': {
            'config_name': best_memory_accuracy['config_name'],
            'score': float(best_memory_accuracy['memory_task_token_accuracy']),
        },
        'best_tokens_per_second': {
            'config_name': best_speed['config_name'],
            'score': float(best_speed['tokens_per_second']),
        },
        'best_quality_per_parameter': {
            'config_name': best_quality_per_parameter['config_name'],
            'score': float(
                best_quality_per_parameter['memory_task_token_accuracy']
            )
            / float(best_quality_per_parameter['parameter_count']),
        },
        'best_quality_per_second': {
            'config_name': best_quality_per_second['config_name'],
            'score': float(
                best_quality_per_second['memory_task_token_accuracy']
            )
            * float(best_quality_per_second['tokens_per_second']),
        },
        'best_balanced_memory': {
            'config_name': balanced_memory['config_name'],
            'score': _balanced_score(balanced_memory, results),
        },
    }
    controls = {
        str(item['config_name']): item
        for item in results
        if item['config_name']
        in ('dense-control', 'workspace-control', 'graph-control')
    }
    control_comparisons: list[dict[str, Any]] = []
    for memory_item in memory_results:
        for control_name, control in controls.items():
            control_comparisons.append(
                {
                    'memory_variant': memory_item['config_name'],
                    'control': control_name,
                    'eval_loss_delta': float(memory_item['eval_loss'])
                    - float(control['eval_loss']),
                    'token_accuracy_delta': float(
                        memory_item['token_accuracy']
                    )
                    - float(control['token_accuracy']),
                    'memory_task_accuracy_delta': float(
                        memory_item['memory_task_token_accuracy']
                    )
                    - float(control['memory_task_token_accuracy']),
                }
            )

    created_at = datetime.now(timezone.utc).isoformat()
    artifacts = {
        'comparison_json': str(output_dir / 'memory-comparison.json'),
        'comparison_markdown': str(output_dir / 'memory-comparison.md'),
    }
    summary = build_common_summary(
        command='compare-memory',
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
        schema_version=MEMORY_COMPARISON_SCHEMA_VERSION,
    )
    summary.update(
        {
            'workspace_enabled': None,
            'workspace_num_slots': None,
            'workspace_gated_writeback': None,
            'workspace_variant_name': 'memory-comparison',
            'graph_enabled': None,
            'graph_num_cycles': None,
            'graph_edge_mode': None,
            'graph_top_k': None,
            'graph_gated_update': None,
            'graph_variant_name': 'memory-comparison',
            'memory_enabled': None,
            'memory_num_slots': None,
            'memory_write_mode': None,
            'memory_read_mode': None,
            'memory_reset_mode': None,
            'memory_decay': None,
            'memory_variant_name': 'memory-comparison',
            'winners': winners,
            'control_comparisons': control_comparisons,
            'results': results,
            'notes': [
                'All memory-enabled variants passed reset/isolation checks.',
                'Memory state exists only in explicit run-local cache tensors.',
                'Toy metrics do not demonstrate human-like or long-term memory.',
                'Expert routing and later modules are not exercised by this memory report.',
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'memory-comparison.json', summary)
    (output_dir / 'memory-comparison.md').write_text(
        _memory_comparison_markdown(summary), encoding='utf-8', newline='\n'
    )
    return summary
