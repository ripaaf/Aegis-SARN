'''Minimal deterministic training and Phase 1 smoke workflow.'''

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from torch.optim import AdamW, Optimizer

from aegis_sarn.aegis.trace import TraceRecorder
from aegis_sarn.config import (
    ArtifactConfig,
    ModelConfig,
    RunManifest,
    SeedConfig,
    TrainingConfig,
)
from aegis_sarn.eval import evaluate_loss, language_model_loss
from aegis_sarn.sarn.checkpoint import load_checkpoint, save_checkpoint
from aegis_sarn.sarn.data import ToyBatch, repeated_pattern_batch
from aegis_sarn.sarn.generation import generate_greedy
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.utils import (
    config_hash,
    device_info,
    git_commit,
    normalize_json,
    package_version,
    set_global_seed,
    sha256_file,
    write_json,
)


@dataclass(slots=True)
class SmokeTrainingResult:
    initial_loss: float
    final_loss: float
    evaluation_loss: float
    losses: list[float]
    checkpoint_path: Path
    manifest_path: Path
    generated_ids: list[int]
    completed_step: int

    def to_dict(self) -> dict[str, object]:
        return {
            'initial_loss': self.initial_loss,
            'final_loss': self.final_loss,
            'evaluation_loss': self.evaluation_loss,
            'losses': self.losses,
            'checkpoint_path': str(self.checkpoint_path),
            'manifest_path': str(self.manifest_path),
            'generated_ids': self.generated_ids,
            'completed_step': self.completed_step,
        }


def train_steps(
    model: SARNDense,
    batch: ToyBatch,
    config: TrainingConfig,
    steps: int,
    optimizer: Optimizer | None = None,
) -> tuple[list[float], Optimizer]:
    if steps <= 0:
        raise ValueError('steps must be positive')
    device = torch.device(config.device)
    model.to(device)
    model.train()
    optimizer = optimizer or AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    input_ids = batch.input_ids.to(device)
    labels = batch.labels.to(device)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = language_model_loss(model(input_ids), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite training loss')
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return losses, optimizer


def run_smoke_training(
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    seed_config: SeedConfig | None = None,
    artifact_config: ArtifactConfig | None = None,
    command_args: dict[str, object] | None = None,
) -> SmokeTrainingResult:
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()
    seed_config = seed_config or SeedConfig()
    artifact_config = artifact_config or ArtifactConfig()
    if training_config.sequence_length > model_config.max_seq_len:
        raise ValueError('training sequence_length exceeds model max_seq_len')
    if model_config.vocab_size < 5:
        raise ValueError('smoke training requires vocab_size >= 5')

    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    trace = TraceRecorder(run_id)
    trace.emit(
        'train.started',
        'sarn.trainer',
        {'steps': training_config.max_steps, 'device': training_config.device},
    )
    set_global_seed(seed_config)
    batch = repeated_pattern_batch(
        batch_size=training_config.batch_size,
        sequence_length=training_config.sequence_length,
    )
    model = SARNDense(model_config)
    losses, optimizer = train_steps(
        model, batch, training_config, training_config.max_steps
    )
    trace.emit(
        'train.completed',
        'sarn.trainer',
        {'initial_loss': losses[0], 'final_loss': losses[-1]},
    )
    save_checkpoint(
        artifact_config.checkpoint_path,
        model,
        optimizer,
        training_config.max_steps,
        training_config,
    )
    trace.emit(
        'checkpoint.saved',
        'sarn.checkpoint',
        {'path': str(artifact_config.checkpoint_path), 'step': training_config.max_steps},
    )

    resumed_model = SARNDense(model_config).to(training_config.device)
    resumed_optimizer = AdamW(
        resumed_model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loaded = load_checkpoint(
        artifact_config.checkpoint_path,
        resumed_model,
        resumed_optimizer,
        map_location=training_config.device,
    )
    trace.emit('checkpoint.loaded', 'sarn.checkpoint', {'step': loaded.step})
    resumed_losses, resumed_optimizer = train_steps(
        resumed_model, batch, training_config, 1, resumed_optimizer
    )
    completed_step = loaded.step + 1
    trace.emit(
        'train.resumed',
        'sarn.trainer',
        {'completed_step': completed_step, 'loss': resumed_losses[-1]},
    )
    save_checkpoint(
        artifact_config.checkpoint_path,
        resumed_model,
        resumed_optimizer,
        completed_step,
        training_config,
    )

    all_losses = [*losses, *resumed_losses]
    evaluation_loss = evaluate_loss(
        resumed_model, batch.input_ids, batch.labels
    )
    trace.emit(
        'evaluation.completed',
        'sarn.evaluation',
        {'loss': evaluation_loss},
    )
    prompt = batch.input_ids[:1, :4].to(training_config.device)
    generated = generate_greedy(resumed_model, prompt, max_new_tokens=8)
    trace.emit(
        'generation.completed',
        'sarn.generation',
        {'prompt_tokens': prompt.shape[1], 'generated_tokens': 8},
    )

    trace.emit('run.completed', 'sarn.training_workflow', {'status': 'completed'})
    configuration_payload = {
        'model': model_config.to_dict(),
        'training': training_config.to_dict(),
        'seed': seed_config.to_dict(),
        'artifact': artifact_config.to_dict(),
    }
    manifest = RunManifest(
        run_id=run_id,
        run_name='phase1-smoke',
        created_at=created_at,
        status='completed',
        model_config=model_config.to_dict(),
        training_config=training_config.to_dict(),
        seed_config=seed_config.to_dict(),
        runtime_config={'device': training_config.device},
        decoding_config={'strategy': 'greedy', 'max_new_tokens': 8},
        package_version=package_version(),
        git_commit=git_commit(),
        device_info=device_info(training_config.device),
        command='train-smoke',
        command_args=normalize_json(command_args or {}),
        artifacts={
            'checkpoint': str(artifact_config.checkpoint_path),
            'checkpoint_digest': sha256_file(artifact_config.checkpoint_path),
        },
        metrics={
            'initial_loss': all_losses[0],
            'final_loss': all_losses[-1],
            'evaluation_loss': evaluation_loss,
            'completed_step': completed_step,
            'dataset_name': 'toy/repeated_pattern',
            'task': batch.task,
            'split': 'train-smoke',
            'examples': training_config.batch_size,
            'sequence_length': training_config.sequence_length,
            **resumed_model.workspace_metrics(),
        },
        trace_events=[event.to_dict() for event in trace.events],
        config_hash=config_hash(configuration_payload),
    )
    write_json(artifact_config.manifest_path, manifest.to_dict())
    return SmokeTrainingResult(
        initial_loss=all_losses[0],
        final_loss=all_losses[-1],
        evaluation_loss=evaluation_loss,
        losses=all_losses,
        checkpoint_path=artifact_config.checkpoint_path,
        manifest_path=artifact_config.manifest_path,
        generated_ids=generated[0].tolist(),
        completed_step=completed_step,
    )

