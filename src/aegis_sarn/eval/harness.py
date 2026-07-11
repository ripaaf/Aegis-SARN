'''Deterministic toy evaluation with a reproducibility manifest.'''

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from aegis_sarn.aegis.trace import TraceRecorder
from aegis_sarn.config import DecodingConfig, RunManifest, SeedConfig
from aegis_sarn.eval.loss import language_model_loss
from aegis_sarn.sarn.data import repeated_pattern_batch
from aegis_sarn.sarn.generation import generate
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.sarn.tokenizer import ByteTokenizer
from aegis_sarn.utils import (
    config_hash,
    device_info,
    git_commit,
    normalize_json,
    package_version,
    set_global_seed,
    write_json,
)


@dataclass(slots=True)
class HarnessResult:
    metrics: dict[str, object]
    manifest_path: Path
    run_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            'run_id': self.run_id,
            'manifest_path': str(self.manifest_path),
            'metrics': self.metrics,
        }


def evaluate_toy(
    model: SARNDense,
    output_dir: Path,
    seed_config: SeedConfig,
    decoding_config: DecodingConfig,
    device: str = 'cpu',
    batch_size: int = 8,
    sequence_length: int = 24,
    command_args: dict[str, object] | None = None,
    artifacts: dict[str, str] | None = None,
) -> HarnessResult:
    if sequence_length > model.config.max_seq_len:
        raise ValueError('evaluation sequence_length exceeds model max_seq_len')
    if model.config.vocab_size < 5:
        raise ValueError('toy evaluation requires vocab_size >= 5')

    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    trace = TraceRecorder(run_id)
    trace.emit('eval.started', 'eval.toy', {'device': device})
    started = time.perf_counter()
    try:
        set_global_seed(seed_config)
        selected_device = torch.device(device)
        model = model.to(selected_device)
        model.eval()
        batch = repeated_pattern_batch(batch_size, sequence_length)
        input_ids = batch.input_ids.to(selected_device)
        labels = batch.labels.to(selected_device)
        with torch.inference_mode():
            logits = model(input_ids)
            loss = float(language_model_loss(logits, labels).item())
            accuracy = float(logits.argmax(dim=-1).eq(labels).float().mean().item())
        perplexity = math.exp(min(loss, 80.0))
        trace.emit(
            'eval.metrics_computed',
            'eval.toy',
            {'validation_loss': loss, 'token_accuracy': accuracy},
        )

        prompt = input_ids[:1, : min(4, sequence_length)]
        generated = generate(model, prompt, decoding_config)
        new_ids = generated[0, prompt.shape[1] :].tolist()
        if model.config.vocab_size == ByteTokenizer.vocab_size:
            sample = ByteTokenizer().decode(new_ids)
        else:
            sample = ' '.join(str(token_id) for token_id in new_ids)
        duration_ms = (time.perf_counter() - started) * 1000.0
        metrics: dict[str, object] = {
            'validation_loss': loss,
            'perplexity': perplexity,
            'token_accuracy': accuracy,
            'generation_sample': sample,
            'generation_token_ids': new_ids,
            'runtime_duration_ms': duration_ms,
            'dataset_name': 'toy/repeated_pattern',
            'task': batch.task,
            'split': 'validation',
            'examples': batch_size,
            'sequence_length': sequence_length,
        }
        trace.emit(
            'generation.completed',
            'eval.toy',
            {'generated_tokens': len(new_ids)},
        )
        trace.emit('eval.completed', 'eval.toy', {'status': 'completed'})
    except Exception as error:
        trace.emit(
            'eval.error',
            'eval.toy',
            {'error_type': type(error).__name__, 'message': str(error)},
        )
        raise

    configuration = {
        'model': model.config.to_dict(),
        'decoding': decoding_config.to_dict(),
        'seed': seed_config.to_dict(),
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'device': device,
    }
    manifest = RunManifest(
        run_id=run_id,
        run_name='eval-toy',
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
        command='eval-toy',
        command_args=normalize_json(command_args or {}),
        artifacts={} if artifacts is None else artifacts,
        metrics=metrics,
        trace_events=[event.to_dict() for event in trace.events],
        config_hash=config_hash(configuration),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f'eval-{run_id}.json'
    write_json(manifest_path, manifest.to_dict())
    return HarnessResult(metrics=metrics, manifest_path=manifest_path, run_id=run_id)

