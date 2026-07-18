from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
import torch

from aegis_sarn.cli import main
from aegis_sarn.config import DecodingConfig, ModelConfig, SeedConfig
from aegis_sarn.eval import benchmark_generation, language_model_loss
from aegis_sarn.phase3 import check_gates
from aegis_sarn.sarn.checkpoint import load_checkpoint, save_checkpoint
from aegis_sarn.sarn.generation import generate_greedy
from aegis_sarn.sarn.layers import RotaryEmbedding
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.utils import set_global_seed


def _attention_config(
    attention_type: Literal['mha', 'gqa'], n_kv_heads: int
) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=20,
        d_model=32,
        n_layers=2,
        n_heads=4,
        attention_type=attention_type,
        n_kv_heads=n_kv_heads,
        ffn_hidden_dim=64,
    )


def test_gqa_forward_shape_backward_and_rope() -> None:
    config = _attention_config('gqa', 2)
    model = SARNDense(config)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(inputs)
    loss = language_model_loss(logits, labels)
    loss.backward()

    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)
    query = torch.randn(2, config.n_heads, 5, config.head_dim)
    key = torch.randn(2, config.resolved_n_kv_heads, 5, config.head_dim)
    rotated_query, rotated_key = rope(query, key)
    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape


def test_gqa_with_equal_kv_heads_is_mha_equivalent() -> None:
    mha_config = _attention_config('mha', 4)
    gqa_config = _attention_config('gqa', 4)
    set_global_seed(17)
    mha = SARNDense(mha_config).eval()
    set_global_seed(17)
    gqa = SARNDense(gqa_config).eval()
    gqa.load_state_dict(mha.state_dict())
    inputs = torch.randint(0, mha_config.vocab_size, (2, 7))

    with torch.inference_mode():
        torch.testing.assert_close(mha(inputs), gqa(inputs), rtol=0.0, atol=0.0)


@pytest.mark.parametrize('n_kv_heads', [1, 2])
def test_gqa_cache_parity_and_stored_head_count(n_kv_heads: int) -> None:
    config = _attention_config('gqa', n_kv_heads)
    model = SARNDense(config).eval()
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    _, caches = model.forward_with_cache(prompt, use_cache=True)

    assert caches is not None
    assert all(cache.key.shape[1] == n_kv_heads for cache in caches)
    uncached = generate_greedy(model, prompt, 8, use_kv_cache=False)
    cached = generate_greedy(model, prompt, 8, use_kv_cache=True)
    torch.testing.assert_close(uncached, cached, rtol=0.0, atol=0.0)


def test_gqa_checkpoint_round_trip(tmp_path: Path) -> None:
    config = _attention_config('gqa', 2)
    model = SARNDense(config).eval()
    inputs = torch.randint(0, config.vocab_size, (2, 6))
    with torch.inference_mode():
        expected = model(inputs)
    checkpoint = tmp_path / 'gqa.pt'
    save_checkpoint(checkpoint, model, optimizer=None, step=3)
    loaded = load_checkpoint(checkpoint)

    assert loaded.model.config == config
    assert loaded.model.config.attention_type == 'gqa'
    assert loaded.model.config.resolved_n_kv_heads == 2
    loaded.model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(
            expected, loaded.model(inputs), rtol=0.0, atol=0.0
        )


def test_gqa_benchmark_reports_smaller_cache(tmp_path: Path) -> None:
    decoding = DecodingConfig(max_new_tokens=1, use_kv_cache=True)
    results = {}
    for name, model in (
        ('mha', SARNDense(_attention_config('mha', 4))),
        ('gqa', SARNDense(_attention_config('gqa', 2))),
    ):
        result = benchmark_generation(
            model=model,
            output_dir=tmp_path / name,
            seed_config=SeedConfig(seed=5),
            decoding_config=decoding,
            prompt_length=4,
            repeats=1,
        )
        results[name] = result.metrics

    assert results['gqa']['n_kv_heads'] == 2
    assert (
        results['gqa']['approximate_kv_cache_bytes']
        < results['mha']['approximate_kv_cache_bytes']
    )


@pytest.fixture(scope='module')
def attention_sweep_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp('phase4') / 'attention'
    exit_code = main(
        [
            'sweep-attention',
            '--output-dir',
            str(output_dir),
            '--device',
            'cpu',
            '--seed',
            '123',
            '--train-steps',
            '1',
            '--batch-size',
            '1',
            '--sequence-length',
            '8',
            '--max-new-tokens',
            '1',
            '--bench-repeats',
            '1',
            '--json',
        ]
    )
    assert exit_code == 0
    return output_dir


def test_attention_sweep_outputs_and_manifest_fields(
    attention_sweep_dir: Path,
) -> None:
    summary_path = attention_sweep_dir / 'attention-sweep-summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))

    assert summary['command'] == 'sweep-attention'
    assert summary['metrics']['mha_count'] == 1
    assert summary['metrics']['gqa_count'] == 2
    assert len(summary['results']) == 3
    assert (attention_sweep_dir / 'attention-sweep-summary.md').exists()
    for result in summary['results']:
        assert result['n_heads'] == 4
        assert result['n_kv_heads'] in (1, 2, 4)
        assert result['kv_group_size'] == 4 // result['n_kv_heads']
        for manifest_path in result['manifest_paths'].values():
            manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            assert manifest['attention_type'] == result['attention_type']
            assert manifest['n_heads'] == result['n_heads']
            assert manifest['n_kv_heads'] == result['n_kv_heads']
            assert manifest['kv_group_size'] == result['kv_group_size']


def test_compare_attention_and_phase4_gates(attention_sweep_dir: Path) -> None:
    report_dir = attention_sweep_dir / 'reports'
    exit_code = main(
        [
            'compare-attention',
            '--input',
            str(attention_sweep_dir),
            '--output-dir',
            str(report_dir),
            '--json',
        ]
    )
    gates = check_gates(attention_sweep_dir / 'attention-sweep-summary.json')
    failed = [check['name'] for check in gates['checks'] if not check['passed']]

    assert exit_code == 0
    assert (report_dir / 'attention-comparison.json').exists()
    assert (report_dir / 'attention-comparison.md').exists()
    comparison = json.loads(
        (report_dir / 'attention-comparison.json').read_text(encoding='utf-8')
    )
    assert 'best_quality_per_kv_cache_byte' in comparison['winners']
    assert gates['passed'], failed
    assert any(
        check['name'] == 'attention:gqa_reduces_kv_cache'
        for check in gates['checks']
    )


def test_phase4_cli_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(['sweep-attention', '--help'])
    output = capsys.readouterr().out

    assert raised.value.code == 0
    assert '--output-dir' in output
    assert '--device' in output
