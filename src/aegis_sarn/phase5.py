'''Phase 5 matched latent-workspace experiments within SARN-Dense.'''

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
from aegis_sarn.eval import benchmark_generation, evaluate_toy
from aegis_sarn.phase3 import build_common_summary
from aegis_sarn.registry import record_entry, record_manifest
from aegis_sarn.sarn.checkpoint import load_checkpoint
from aegis_sarn.sarn.training import run_smoke_training
from aegis_sarn.utils import config_hash, write_json

WORKSPACE_SWEEP_SCHEMA_VERSION = 'aegis.workspace_sweep/v1'
WORKSPACE_COMPARISON_SCHEMA_VERSION = 'aegis.workspace_comparison/v1'


@dataclass(frozen=True, slots=True)
class WorkspaceVariant:
    name: str
    enabled: bool
    num_slots: int
    gated_writeback: bool

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=256,
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_hidden_dim=96,
            workspace_enabled=self.enabled,
            workspace_num_slots=self.num_slots,
            workspace_gated_writeback=self.gated_writeback,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'workspace_variant_name': self.name,
            'workspace_enabled': self.enabled,
            'workspace_num_slots': self.num_slots,
            'workspace_gated_writeback': self.enabled
            and self.gated_writeback,
        }


def default_workspace_variants() -> list[WorkspaceVariant]:
    return [
        WorkspaceVariant('dense-control', False, 0, False),
        WorkspaceVariant('workspace-null', True, 2, False),
        WorkspaceVariant('workspace-slots2', True, 2, True),
        WorkspaceVariant('workspace-slots4', True, 4, True),
    ]


def _workspace_sweep_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# SARN-Dense Phase 5 Workspace Sweep',
        '',
        f"Generated at: {summary['timestamp']}",
        '',
        'Matched tiny runs compare the dense control with bounded experimental '
        'latent-workspace variants.',
        '',
        '| Variant | Enabled | Slots | Writeback | Eval Loss | Perplexity | '
        'Token Accuracy | Tokens/Sec | Params | Workspace Params | Gate Mean | '
        'Workspace Norm | Status |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for item in summary['results']:
        lines.append(
            '| {config_name} | {workspace_enabled} | {workspace_num_slots} | '
            '{workspace_gated_writeback} | {eval_loss:.6g} | {perplexity:.6g} | '
            '{token_accuracy:.4f} | {tokens_per_second:.3f} | {parameter_count} | '
            '{workspace_parameter_count} | {workspace_gate_mean:.6g} | '
            '{workspace_norm:.6g} | {status} |'.format(**item)
        )
    lines.extend(
        [
            '',
            '## Limits',
            '',
            '- Results use generated toy data and local CPU measurements.',
            '- Latent slots are learned tensor states, not human-like concepts.',
            '- This prototype is neither graph message passing nor persistent memory.',
            '- This Phase 5 artifact does not exercise the separate Phase 6 graph path.',
            '- Later graph and memory mechanisms are outside this Phase 5 comparison.',
        ]
    )
    return '\n'.join(lines) + '\n'


def run_workspace_sweep(
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
    variants = default_workspace_variants()
    results: list[dict[str, Any]] = []

    for variant in variants:
        variant_started = time.perf_counter()
        variant_dir = output_dir / variant.name
        model_config = variant.model_config()
        if sequence_length > model_config.max_seq_len:
            raise ValueError('sequence_length exceeds the fixed workspace sweep context')
        training_config = TrainingConfig(
            learning_rate=1.0e-2,
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_steps=train_steps,
            device=device,
        )
        seed_config = SeedConfig(seed=seed)
        command_args = {
            'parent_command': 'sweep-workspace',
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
                'workspace_variant_name': variant.name,
                'workspace_enabled': bool(bench_metrics['workspace_enabled']),
                'workspace_num_slots': int(bench_metrics['workspace_num_slots']),
                'workspace_gated_writeback': bool(
                    bench_metrics['workspace_gated_writeback']
                ),
                'workspace_parameter_count': int(
                    bench_metrics['workspace_parameter_count']
                ),
                'workspace_gate_mean': float(bench_metrics['workspace_gate_mean']),
                'workspace_norm': float(bench_metrics['workspace_norm']),
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
        'dense_control_count': sum(
            not item['workspace_enabled'] for item in results
        ),
        'workspace_enabled_count': sum(
            item['workspace_enabled'] for item in results
        ),
        'workspace_null_count': sum(
            item['workspace_enabled']
            and not item['workspace_gated_writeback']
            for item in results
        ),
    }
    artifacts = {
        'summary_json': str(output_dir / 'workspace-sweep-summary.json'),
        'summary_markdown': str(output_dir / 'workspace-sweep-summary.md'),
        'registry': str(registry_path),
    }
    summary = build_common_summary(
        command='sweep-workspace',
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
        schema_version=WORKSPACE_SWEEP_SCHEMA_VERSION,
    )
    summary.update(
        {
            'workspace_enabled': None,
            'workspace_num_slots': None,
            'workspace_gated_writeback': None,
            'workspace_variant_name': 'workspace-sweep',
            'results': results,
        }
    )
    write_json(output_dir / 'workspace-sweep-summary.json', summary)
    (output_dir / 'workspace-sweep-summary.md').write_text(
        _workspace_sweep_markdown(summary), encoding='utf-8', newline='\n'
    )
    record_entry(
        registry_path,
        {
            'run_id': run_id,
            'command_name': 'sweep-workspace',
            'timestamp': created_at,
            'git_commit': summary['git_commit_hash'],
            'package_version': summary['package_version'],
            'config_hash': summary['config_hash'],
            'checkpoint_path': None,
            'manifest_path': str(output_dir / 'workspace-sweep-summary.json'),
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


def _workspace_comparison_markdown(summary: dict[str, Any]) -> str:
    lines = [
        '# SARN-Dense Workspace Comparison',
        '',
        f"Generated at: {summary['timestamp']}",
        '',
        'This report compares a dense control with bounded experimental latent slots.',
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
            '- Quality rankings describe generated toy tasks only.',
            '- Local CPU speed is noisy and hardware-specific.',
            '- The balanced score is a simple documented heuristic.',
            '- These results are not evidence of human-like reasoning or memory.',
            '- No graph message passing is exercised by this Phase 5 report.',
            '- No persistent or resettable memory is implemented.',
        ]
    )
    return '\n'.join(lines) + '\n'


def compare_workspace(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    sweep_path = input_dir / 'workspace-sweep-summary.json'
    sweep = json.loads(sweep_path.read_text(encoding='utf-8'))
    results = sweep.get('results') or []
    if not results:
        raise ValueError('workspace summary does not contain results')

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
    max_accuracy = max(float(item['token_accuracy']) for item in results)
    max_inverse_loss = max(
        1.0 / max(float(item['eval_loss']), 1.0e-12) for item in results
    )
    max_speed = max(float(item['tokens_per_second']) for item in results)
    max_parameter_efficiency = max(
        1.0 / float(item['parameter_count']) for item in results
    )

    def balanced_score(item: dict[str, Any]) -> float:
        return (
            float(item['token_accuracy']) / max(max_accuracy, 1.0e-12)
            + (1.0 / max(float(item['eval_loss']), 1.0e-12)) / max_inverse_loss
            + float(item['tokens_per_second']) / max(max_speed, 1.0e-12)
            + (1.0 / float(item['parameter_count']))
            / max_parameter_efficiency
        ) / 4.0

    balanced = max(results, key=balanced_score)
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
        'best_balanced_workspace': {
            'config_name': balanced['config_name'],
            'score': balanced_score(balanced),
        },
    }
    created_at = datetime.now(timezone.utc).isoformat()
    artifacts = {
        'comparison_json': str(output_dir / 'workspace-comparison.json'),
        'comparison_markdown': str(output_dir / 'workspace-comparison.md'),
    }
    summary = build_common_summary(
        command='compare-workspace',
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
        schema_version=WORKSPACE_COMPARISON_SCHEMA_VERSION,
    )
    summary.update(
        {
            'workspace_enabled': None,
            'workspace_num_slots': None,
            'workspace_gated_writeback': None,
            'workspace_variant_name': 'workspace-comparison',
            'winners': winners,
            'results': results,
            'notes': [
                'All variants share width, depth, attention, FFN, context, data, and seed.',
                'Workspace slot count and writeback are the controlled variables.',
                'Toy metrics do not demonstrate reasoning, concepts, or memory.',
                'The separate Phase 6 graph path is not exercised by this report.',
                'SARN-Hybrid and later modules are not evaluated by this Phase 5 report.',
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'workspace-comparison.json', summary)
    (output_dir / 'workspace-comparison.md').write_text(
        _workspace_comparison_markdown(summary), encoding='utf-8', newline='\n'
    )
    return summary
