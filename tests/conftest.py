from __future__ import annotations

import pytest

from aegis_sarn.config import ModelConfig


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ffn_hidden_dim=32,
        dropout=0.0,
    )

