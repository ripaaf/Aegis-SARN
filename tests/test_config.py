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
    assert config.attention_type == 'mha'
    assert config.resolved_n_kv_heads == config.n_heads
    assert config.kv_group_size == 1


def test_phase_four_gqa_configuration_and_derived_groups() -> None:
    config = ModelConfig(
        d_model=32,
        n_heads=4,
        attention_type='gqa',
        n_kv_heads=2,
        ffn_hidden_dim=64,
    )
    assert config.resolved_n_kv_heads == 2
    assert config.kv_group_size == 2


@pytest.mark.parametrize(
    ('values', 'message'),
    [
        ({'attention_type': 'other'}, 'attention_type'),
        ({'attention_type': 'gqa', 'n_kv_heads': 0}, 'positive'),
        ({'attention_type': 'gqa', 'n_kv_heads': 8}, 'cannot exceed'),
        ({'attention_type': 'gqa', 'n_kv_heads': 3}, 'must divide'),
        ({'attention_type': 'mha', 'n_kv_heads': 2}, 'mha requires'),
    ],
)
def test_invalid_attention_configuration(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        ModelConfig(
            d_model=32,
            n_heads=4,
            ffn_hidden_dim=64,
            **values,  # type: ignore[arg-type]
        )


def test_runtime_and_artifact_paths() -> None:
    runtime = RuntimeConfig(max_prompt_tokens=12, max_new_tokens=4)
    artifact = ArtifactConfig(output_dir=Path('example'))
    assert runtime.max_prompt_tokens == 12
    assert artifact.checkpoint_path == Path('example/sarn-dense-smoke.pt')


def test_decoding_config_rejects_unknown_strategy() -> None:
    with pytest.raises(ConfigError, match='strategy'):
        DecodingConfig(strategy='beam')  # type: ignore[arg-type]
