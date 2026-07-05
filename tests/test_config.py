from pathlib import Path

import pytest

from aegis_sarn.config import (
    ArtifactConfig,
    ConfigError,
    DecodingConfig,
    ModelConfig,
    RuntimeConfig,
)


def test_model_config_round_trip_and_derived_head_dim() -> None:
    config = ModelConfig(d_model=32, n_heads=4, ffn_hidden_dim=64)
    assert ModelConfig.from_dict(config.to_dict()) == config
    assert config.head_dim == 8


def test_phase_one_rejects_gqa_configuration() -> None:
    with pytest.raises(ConfigError, match='MHA only'):
        ModelConfig(d_model=32, n_heads=4, n_kv_heads=2, ffn_hidden_dim=64)


def test_runtime_and_artifact_paths() -> None:
    runtime = RuntimeConfig(max_prompt_tokens=12, max_new_tokens=4)
    artifact = ArtifactConfig(output_dir=Path('example'))
    assert runtime.max_prompt_tokens == 12
    assert artifact.checkpoint_path == Path('example/sarn-dense-smoke.pt')


def test_decoding_config_rejects_unknown_strategy() -> None:
    with pytest.raises(ConfigError, match='strategy'):
        DecodingConfig(strategy='beam')  # type: ignore[arg-type]
