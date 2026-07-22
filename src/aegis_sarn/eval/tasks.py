'''Task-level toy evaluation for the SARN-Dense baseline.'''

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from aegis_sarn.aegis.trace import TraceRecorder
from aegis_sarn.config import DecodingConfig, RunManifest, SeedConfig
from aegis_sarn.eval.harness import HarnessResult
from aegis_sarn.eval.loss import language_model_loss
from aegis_sarn.sarn.data import TOY_TASK_NAMES, make_toy_task_batch
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.utils import (
    config_hash,
    device_info,
    git_commit,
    normalize_json,
    package_version,
    set_global_seed,
    write_json,
)


def _evaluate_batch(
    model: SARNDense,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    with torch.inference_mode():
        logits = model(input_ids)
        loss = float(language_model_loss(logits, labels).item())
        accuracy = float(logits.argmax(dim=-1).eq(labels).float().mean().item())
    return loss, accuracy


def evaluate_tasks(
    model: SARNDense,
    output_dir: Path,
    seed_config: SeedConfig,
    decoding_config: DecodingConfig,
    device: str = 'cpu',
    batch_size: int = 4,
    sequence_length: int = 16,
    task_names: list[str] | tuple[str, ...] = TOY_TASK_NAMES,
    split: str = 'validation',
    command_args: dict[str, object] | None = None,
    artifacts: dict[str, str] | None = None,
) -> HarnessResult:
    if sequence_length > model.config.max_seq_len:
        raise ValueError('task evaluation sequence_length exceeds model max_seq_len')
    if not task_names:
        raise ValueError('at least one task is required')

    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    trace = TraceRecorder(run_id)
    trace.emit('eval_tasks.started', 'eval.tasks', {'device': device, 'tasks': list(task_names)})
    started = time.perf_counter()
    set_global_seed(seed_config)
    selected_device = torch.device(device)
    model = model.to(selected_device).eval()

    per_task: list[dict[str, object]] = []
    for task_name in task_names:
        batch = make_toy_task_batch(
            task_name,
            batch_size=batch_size,
            sequence_length=sequence_length,
            vocab_size=model.config.vocab_size,
            seed=seed_config.seed,
            split=split,
        )
        input_ids = batch.input_ids.to(selected_device)
        labels = batch.labels.to(selected_device)
        loss, accuracy = _evaluate_batch(model, input_ids, labels)
        perplexity = math.exp(min(loss, 80.0))
        per_task.append(
            {
                'task': batch.task,
                'validation_loss': loss,
                'token_accuracy': accuracy,
                'perplexity': perplexity,
                'examples': batch_size,
                'sequence_length': sequence_length,
                'split': split,
            }
        )
        trace.emit(
            'eval_tasks.task_completed',
            'eval.tasks',
            {'task': batch.task, 'validation_loss': loss, 'token_accuracy': accuracy},
        )

    aggregate_loss = sum(float(item['validation_loss']) for item in per_task) / len(per_task)
    aggregate_accuracy = sum(float(item['token_accuracy']) for item in per_task) / len(per_task)
    aggregate_perplexity = sum(float(item['perplexity']) for item in per_task) / len(per_task)
    duration_ms = (time.perf_counter() - started) * 1000.0
    metrics: dict[str, object] = {
        'tasks': per_task,
        'task_count': len(per_task),
        'aggregate_validation_loss': aggregate_loss,
        'aggregate_token_accuracy': aggregate_accuracy,
        'aggregate_perplexity': aggregate_perplexity,
        'validation_loss': aggregate_loss,
        'token_accuracy': aggregate_accuracy,
        'perplexity': aggregate_perplexity,
        'runtime_duration_ms': duration_ms,
        'dataset_name': 'toy/task_suite',
        'split': split,
        'examples_per_task': batch_size,
        'sequence_length': sequence_length,
    }
    metrics.update(model.workspace_metrics())
    metrics.update(model.graph_metrics())
    metrics.update(model.memory_metrics())
    metrics.update(model.expert_metrics())
    trace.emit('eval_tasks.completed', 'eval.tasks', {'status': 'completed'})

    configuration = {
        'model': model.config.to_dict(),
        'decoding': decoding_config.to_dict(),
        'seed': seed_config.to_dict(),
        'device': device,
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'tasks': list(task_names),
        'split': split,
    }
    manifest = RunManifest(
        run_id=run_id,
        run_name='eval-tasks',
        created_at=created_at,
        status='completed',
        model_config=model.config.to_dict(),
        training_config={},
        seed_config=seed_config.to_dict(),
        runtime_config={'device': device},
        decoding_config=decoding_config.to_dict(),
        package_version=package_version(),
        git_commit=git_commit(),
        device_info=device_info(device),
        command='eval-tasks',
        command_args=normalize_json(command_args or {}),
        artifacts={} if artifacts is None else artifacts,
        metrics=metrics,
        trace_events=[event.to_dict() for event in trace.events],
        config_hash=config_hash(configuration),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f'eval-tasks-{run_id}.json'
    write_json(manifest_path, manifest.to_dict())
    return HarnessResult(metrics=metrics, manifest_path=manifest_path, run_id=run_id)
