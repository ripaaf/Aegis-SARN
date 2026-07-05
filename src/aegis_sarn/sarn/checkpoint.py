'''Atomic local checkpoints for trusted Phase 1 artifacts.'''

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer

from aegis_sarn.config import ModelConfig, TrainingConfig
from aegis_sarn.sarn.model import SARNDense


@dataclass(slots=True)
class LoadedCheckpoint:
    model: SARNDense
    step: int
    training_config: dict[str, Any] | None


def save_checkpoint(
    path: Path,
    model: SARNDense,
    optimizer: Optimizer | None,
    step: int,
    training_config: TrainingConfig | None = None,
) -> Path:
    if step < 0:
        raise ValueError('checkpoint step cannot be negative')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format_version': 'aegis.sarn_checkpoint/v1',
        'model_config': model.config.to_dict(),
        'model_state': model.state_dict(),
        'optimizer_state': None if optimizer is None else optimizer.state_dict(),
        'step': step,
        'training_config': None if training_config is None else training_config.to_dict(),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp'
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_checkpoint(
    path: Path,
    model: SARNDense | None = None,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = 'cpu',
) -> LoadedCheckpoint:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get('format_version') != 'aegis.sarn_checkpoint/v1':
        raise ValueError('unsupported checkpoint format')
    stored_config = ModelConfig.from_dict(payload['model_config'])
    if model is None:
        model = SARNDense(stored_config)
    elif model.config != stored_config:
        raise ValueError('checkpoint model config does not match the destination model')
    model.load_state_dict(payload['model_state'], strict=True)
    if optimizer is not None:
        if payload['optimizer_state'] is None:
            raise ValueError('checkpoint does not contain optimizer state')
        optimizer.load_state_dict(payload['optimizer_state'])
    return LoadedCheckpoint(
        model=model,
        step=int(payload['step']),
        training_config=payload['training_config'],
    )

