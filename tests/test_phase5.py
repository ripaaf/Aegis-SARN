from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from aegis_sarn.cli import main
from aegis_sarn.config import ConfigError, ModelConfig
from aegis_sarn.eval import language_model_loss
from aegis_sarn.phase3 import check_gates
from aegis_sarn.sarn.checkpoint import load_checkpoint, save_checkpoint
from aegis_sarn.sarn.generation import generate_greedy
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.sarn.workspace import LatentWorkspace
from aegis_sarn.utils import set_global_seed


def _workspace_config(
    slots: int = 2, gated_writeback: bool = True
) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=20,
        d_model=32,
        n_layers=1,
        n_heads=4,
        ffn_hidden_dim=64,
        workspace_enabled=True,
        workspace_num_slots=slots,
        workspace_gated_writeback=gated_writeback,
    )


def test_workspace_defaults_and_validation() -> None:
    default = ModelConfig()
    assert default.workspace_enabled is False
    assert default.workspace_num_slots == 0
    assert default.workspace_gated_writeback is True

    ignored_slots = ModelConfig(workspace_enabled=False, workspace_num_slots=3)
    assert SARNDense(ignored_slots).workspace is None

    with pytest.raises(ConfigError, match='must be positive'):
        ModelConfig(workspace_enabled=True, workspace_num_slots=0)
    with pytest.raises(ConfigError, match='cannot be negative'):
        ModelConfig(workspace_num_slots=-1)
    with pytest.raises(ConfigError, match='workspace_dropout'):
        ModelConfig(workspace_dropout=1.0)
    with pytest.raises(ConfigError, match='workspace_read_mode'):
        ModelConfig(workspace_read_mode='other')  # type: ignore[arg-type]


def test_workspace_module_shape_state_diagnostics_and_backward() -> None:
    config = _workspace_config(slots=2)
    workspace = LatentWorkspace(config)
    inputs = torch.randn(2, 7, config.d_model, requires_grad=True)
    output, slots, diagnostics = workspace(inputs)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == inputs.shape
    assert slots.shape == (2, 2, config.d_model)
    assert diagnostics.num_slots == 2
    assert diagnostics.gate_mean.item() > 0
    assert diagnostics.workspace_norm.item() > 0
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in workspace.parameters()
    )


def test_workspace_enabled_model_forward_and_backward() -> None:
    config = _workspace_config(slots=4)
    model = SARNDense(config)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(inputs)
    loss = language_model_loss(logits, labels)
    loss.backward()

    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(loss)
    metrics = model.workspace_metrics()
    assert metrics['workspace_enabled'] is True
    assert metrics['workspace_num_slots'] == 4
    assert int(metrics['workspace_parameter_count']) > 0


def test_no_writeback_control_matches_dense_with_same_seed() -> None:
    common = {
        'vocab_size': 32,
        'max_seq_len': 20,
        'd_model': 32,
        'n_layers': 1,
        'n_heads': 4,
        'ffn_hidden_dim': 64,
    }
    set_global_seed(31)
    dense = SARNDense(ModelConfig(**common)).eval()
    set_global_seed(31)
    workspace_null = SARNDense(
        ModelConfig(
            **common,
            workspace_enabled=True,
            workspace_num_slots=2,
            workspace_gated_writeback=False,
        )
    ).eval()
    inputs = torch.randint(0, 32, (2, 8))

    with torch.inference_mode():
        torch.testing.assert_close(
            dense(inputs), workspace_null(inputs), rtol=0.0, atol=0.0
        )


def test_workspace_cached_generation_parity_and_transient_slots() -> None:
    config = _workspace_config(slots=2)
    model = SARNDense(config).eval()
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    _, caches = model.forward_with_cache(prompt, use_cache=True)

    assert caches is not None
    assert caches[0].workspace_slots is not None
    assert caches[0].workspace_slots.shape == (1, 2, config.d_model)
    uncached = generate_greedy(model, prompt, 8, use_kv_cache=False)
    cached = generate_greedy(model, prompt, 8, use_kv_cache=True)
    torch.testing.assert_close(uncached, cached, rtol=0.0, atol=0.0)


def test_workspace_checkpoint_round_trip(tmp_path: Path) -> None:
    config = _workspace_config(slots=4)
    model = SARNDense(config).eval()
    inputs = torch.randint(0, config.vocab_size, (2, 6))
    with torch.inference_mode():
        expected = model(inputs)
    checkpoint = tmp_path / 'workspace.pt'
    save_checkpoint(checkpoint, model, optimizer=None, step=4)
    loaded = load_checkpoint(checkpoint)

    assert loaded.model.config == config
    assert loaded.model.workspace is not None
    loaded.model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(
            expected, loaded.model(inputs), rtol=0.0, atol=0.0
        )


@pytest.fixture(scope='module')
def workspace_sweep_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp('phase5') / 'workspace'
    exit_code = main(
        [
            'sweep-workspace',
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


def test_workspace_sweep_outputs_and_manifest_fields(
    workspace_sweep_dir: Path,
) -> None:
    summary_path = workspace_sweep_dir / 'workspace-sweep-summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    required_workspace = {
        'workspace_enabled',
        'workspace_num_slots',
        'workspace_gated_writeback',
        'workspace_variant_name',
    }

    assert summary['command'] == 'sweep-workspace'
    assert summary['metrics']['dense_control_count'] == 1
    assert summary['metrics']['workspace_enabled_count'] == 3
    assert len(summary['results']) == 4
    assert (workspace_sweep_dir / 'workspace-sweep-summary.md').exists()
    for result in summary['results']:
        assert required_workspace.issubset(result)
        assert result['parameter_count'] > 0
        for manifest_path in result['manifest_paths'].values():
            manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            assert required_workspace.issubset(manifest)
            assert (
                manifest['workspace_variant_name']
                == result['workspace_variant_name']
            )


def test_compare_workspace_and_phase5_gates(
    workspace_sweep_dir: Path,
) -> None:
    report_dir = workspace_sweep_dir / 'reports'
    exit_code = main(
        [
            'compare-workspace',
            '--input',
            str(workspace_sweep_dir),
            '--output-dir',
            str(report_dir),
            '--json',
        ]
    )
    gates = check_gates(workspace_sweep_dir / 'workspace-sweep-summary.json')
    failed = [check['name'] for check in gates['checks'] if not check['passed']]

    assert exit_code == 0
    assert (report_dir / 'workspace-comparison.json').exists()
    assert (report_dir / 'workspace-comparison.md').exists()
    comparison = json.loads(
        (report_dir / 'workspace-comparison.json').read_text(encoding='utf-8')
    )
    assert 'best_perplexity' in comparison['winners']
    assert 'best_balanced_workspace' in comparison['winners']
    assert gates['passed'], failed
    assert any(
        check['name'] == 'workspace:dense_control_present'
        for check in gates['checks']
    )


def test_phase5_cli_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(['sweep-workspace', '--help'])
    output = capsys.readouterr().out

    assert raised.value.code == 0
    assert '--output-dir' in output
    assert '--device' in output
