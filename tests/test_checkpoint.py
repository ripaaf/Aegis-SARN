from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW

from aegis_sarn.config import ModelConfig, TrainingConfig
from aegis_sarn.eval import language_model_loss
from aegis_sarn.sarn.checkpoint import load_checkpoint, save_checkpoint
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.utils import set_global_seed


def test_checkpoint_save_load_and_optimizer_resume_parity(
    tmp_path: Path, tiny_model_config: ModelConfig
) -> None:
    set_global_seed(23)
    model = SARNDense(tiny_model_config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randint(0, tiny_model_config.vocab_size, (2, 8))
    labels = torch.randint(0, tiny_model_config.vocab_size, (2, 8))
    loss = language_model_loss(model(inputs), labels)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        expected = model(inputs)

    path = tmp_path / 'checkpoint.pt'
    training_config = TrainingConfig(max_steps=1, sequence_length=8)
    save_checkpoint(path, model, optimizer, step=1, training_config=training_config)

    restored = SARNDense(tiny_model_config)
    restored_optimizer = AdamW(restored.parameters(), lr=1e-3)
    loaded = load_checkpoint(path, restored, restored_optimizer)
    restored.eval()
    with torch.inference_mode():
        actual = restored(inputs)

    assert loaded.step == 1
    assert loaded.training_config == training_config.to_dict()
    assert restored_optimizer.state_dict()['state']
    torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)

