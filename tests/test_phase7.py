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
from aegis_sarn.sarn.data import MEMORY_TASK_NAMES, make_toy_task_batch
from aegis_sarn.sarn.generation import generate_greedy
from aegis_sarn.sarn.memory import ResettableWorkingMemory
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.utils import set_global_seed


def _memory_config(graph_enabled: bool = False) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=20,
        d_model=32,
        n_layers=1,
        n_heads=4,
        ffn_hidden_dim=64,
        workspace_enabled=True,
        workspace_num_slots=4,
        graph_enabled=graph_enabled,
        graph_num_cycles=1 if graph_enabled else 0,
        graph_edge_mode='learned_dense' if graph_enabled else 'none',
        memory_enabled=True,
        memory_num_slots=4,
        memory_write_mode='gated',
        memory_read_mode='attention',
    )


def test_memory_defaults_and_config_validation() -> None:
    default = ModelConfig()
    assert default.memory_enabled is False
    assert default.memory_num_slots == 0
    assert default.memory_write_mode == 'none'
    assert default.memory_read_mode == 'none'
    assert default.memory_reset_mode == 'per_generation'
    assert SARNDense(default).memory is None

    with pytest.raises(ConfigError, match='requires workspace'):
        ModelConfig(memory_enabled=True, memory_num_slots=2)
    with pytest.raises(ConfigError, match='must be positive'):
        ModelConfig(
            workspace_enabled=True,
            workspace_num_slots=2,
            memory_enabled=True,
        )
    with pytest.raises(ConfigError, match='cannot be negative'):
        ModelConfig(memory_num_slots=-1)
    with pytest.raises(ConfigError, match='memory_decay'):
        ModelConfig(memory_decay=1.1)
    with pytest.raises(ConfigError, match='memory_write_mode'):
        ModelConfig(memory_write_mode='other')  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match='memory_read_mode'):
        ModelConfig(memory_read_mode='other')  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match='memory_reset_mode'):
        ModelConfig(memory_reset_mode='other')  # type: ignore[arg-type]


def test_memory_shape_diagnostics_backward_and_explicit_cache() -> None:
    config = _memory_config()
    memory = ResettableWorkingMemory(config)
    tokens = torch.randn(2, 7, config.d_model, requires_grad=True)
    workspace = torch.randn(2, 7, 4, config.d_model, requires_grad=True)
    output, returned_workspace, state, diagnostics = memory(tokens, workspace)
    output.square().mean().backward()

    assert output.shape == tokens.shape
    assert returned_workspace is workspace
    assert state.shape == (2, 4, config.d_model)
    assert diagnostics.enabled is True
    assert diagnostics.num_slots == 4
    assert diagnostics.reset_applied is True
    assert diagnostics.gate_mean.item() > 0
    assert diagnostics.memory_norm.item() > 0
    assert diagnostics.write_norm.item() > 0
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in memory.parameters()
    )

    continued = memory(tokens.detach(), workspace.detach(), state)
    assert continued[2].shape == state.shape
    assert continued[3].reset_applied is False


def test_memory_resets_and_isolates_independent_calls() -> None:
    memory = ResettableWorkingMemory(_memory_config()).eval()
    tokens = torch.randn(2, 6, 32)
    workspace = torch.randn(2, 6, 4, 32)

    with torch.inference_mode():
        first_output, _, first_state, first_diagnostics = memory(
            tokens, workspace
        )
        memory(torch.randn_like(tokens), torch.randn_like(workspace))
        repeated_output, _, repeated_state, repeated_diagnostics = memory(
            tokens, workspace
        )
        row_zero = memory(tokens[:1], workspace[:1])
        row_one = memory(tokens[1:], workspace[1:])

    torch.testing.assert_close(first_output, repeated_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_state, repeated_state, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first_output[:1], row_zero[0], rtol=0.0, atol=1e-7
    )
    torch.testing.assert_close(
        first_output[1:], row_one[0], rtol=0.0, atol=1e-7
    )
    torch.testing.assert_close(
        first_state[:1], row_zero[2], rtol=0.0, atol=1e-7
    )
    torch.testing.assert_close(
        first_state[1:], row_one[2], rtol=0.0, atol=1e-7
    )
    assert first_diagnostics.reset_applied
    assert repeated_diagnostics.reset_applied


def test_memory_null_is_exact_workspace_noop() -> None:
    common = {
        'vocab_size': 32,
        'max_seq_len': 20,
        'd_model': 32,
        'n_layers': 1,
        'n_heads': 4,
        'ffn_hidden_dim': 64,
        'workspace_enabled': True,
        'workspace_num_slots': 4,
    }
    set_global_seed(71)
    workspace_control = SARNDense(ModelConfig(**common)).eval()
    set_global_seed(71)
    memory_null = SARNDense(
        ModelConfig(
            **common,
            memory_enabled=True,
            memory_num_slots=4,
            memory_write_mode='none',
            memory_read_mode='none',
        )
    ).eval()
    inputs = torch.randint(0, 32, (2, 8))

    with torch.inference_mode():
        torch.testing.assert_close(
            workspace_control(inputs), memory_null(inputs), rtol=0.0, atol=0.0
        )
    assert memory_null.active_parameter_count() < memory_null.count_parameters()


@pytest.mark.parametrize('graph_enabled', [False, True])
def test_memory_model_forward_backward_and_controls(graph_enabled: bool) -> None:
    config = _memory_config(graph_enabled=graph_enabled)
    model = SARNDense(config)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(inputs)
    loss = language_model_loss(logits, labels)
    loss.backward()

    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(loss)
    metrics = model.memory_metrics()
    assert metrics['memory_enabled'] is True
    assert metrics['memory_num_slots'] == 4
    assert int(metrics['memory_parameter_count']) > 0

    dense = SARNDense(
        ModelConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_hidden_dim=64,
        )
    )
    workspace = SARNDense(
        ModelConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_hidden_dim=64,
            workspace_enabled=True,
            workspace_num_slots=4,
        )
    )
    assert dense(inputs).shape == logits.shape
    assert workspace(inputs).shape == logits.shape


def test_memory_checkpoint_generation_cache_and_weight_immutability(
    tmp_path: Path,
) -> None:
    config = _memory_config(graph_enabled=True)
    model = SARNDense(config).eval()
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }

    with torch.inference_mode():
        expected = model(prompt)
        _, cache = model.forward_with_cache(prompt, use_cache=True)
    assert cache is not None and cache[0].memory_slots is not None
    assert cache[0].memory_slots.shape == (1, 4, config.d_model)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0.0, atol=0.0)

    checkpoint = tmp_path / 'memory.pt'
    save_checkpoint(checkpoint, model, optimizer=None, step=7)
    loaded = load_checkpoint(checkpoint)
    assert loaded.model.config == config
    assert loaded.model.memory is not None
    loaded.model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(
            expected, loaded.model(prompt), rtol=0.0, atol=0.0
        )

    uncached = generate_greedy(loaded.model, prompt, 8, use_kv_cache=False)
    cached = generate_greedy(loaded.model, prompt, 8, use_kv_cache=True)
    torch.testing.assert_close(uncached, cached, rtol=0.0, atol=0.0)


def test_memory_tasks_are_deterministic_and_split_aware() -> None:
    for task_name in MEMORY_TASK_NAMES:
        first = make_toy_task_batch(
            task_name, 2, 12, vocab_size=32, seed=23, split='validation'
        )
        second = make_toy_task_batch(
            task_name, 2, 12, vocab_size=32, seed=23, split='validation'
        )
        train = make_toy_task_batch(
            task_name, 2, 12, vocab_size=32, seed=23, split='train'
        )
        assert first.task == task_name
        torch.testing.assert_close(first.input_ids, second.input_ids)
        torch.testing.assert_close(first.labels, second.labels)
        assert not torch.equal(first.input_ids, train.input_ids)


@pytest.fixture(scope='module')
def memory_sweep_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp('phase7') / 'memory'
    exit_code = main(
        [
            'sweep-memory',
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


def test_memory_sweep_outputs_and_manifest_fields(memory_sweep_dir: Path) -> None:
    summary_path = memory_sweep_dir / 'memory-sweep-summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    required_memory = {
        'memory_enabled',
        'memory_num_slots',
        'memory_write_mode',
        'memory_read_mode',
        'memory_reset_mode',
        'memory_decay',
        'memory_variant_name',
    }

    assert summary['command'] == 'sweep-memory'
    assert summary['metrics']['config_count'] == 6
    assert summary['metrics']['memory_enabled_count'] == 3
    assert summary['metrics']['reset_isolation_passed_count'] == 6
    assert (memory_sweep_dir / 'memory-sweep-summary.md').exists()
    assert (memory_sweep_dir / 'runs' / 'registry.json').exists()
    for result in summary['results']:
        assert required_memory.issubset(result)
        assert len(result['memory_task_metrics']) == len(MEMORY_TASK_NAMES)
        assert result['parameter_count'] > 0
        assert result['active_parameter_count'] > 0
        assert result['memory_reset_isolation_passed'] is True
        for manifest_path in result['manifest_paths'].values():
            manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            assert required_memory.issubset(manifest)
            assert (
                manifest['memory_variant_name']
                == result['memory_variant_name']
            )


def test_compare_memory_and_phase7_gates(memory_sweep_dir: Path) -> None:
    report_dir = memory_sweep_dir / 'reports'
    exit_code = main(
        [
            'compare-memory',
            '--input',
            str(memory_sweep_dir),
            '--output-dir',
            str(report_dir),
            '--json',
        ]
    )
    gates = check_gates(memory_sweep_dir / 'memory-sweep-summary.json')
    failed = [check['name'] for check in gates['checks'] if not check['passed']]

    assert exit_code == 0
    assert (report_dir / 'memory-comparison.json').exists()
    assert (report_dir / 'memory-comparison.md').exists()
    comparison = json.loads(
        (report_dir / 'memory-comparison.json').read_text(encoding='utf-8')
    )
    assert 'best_memory_task_accuracy' in comparison['winners']
    assert 'best_balanced_memory' in comparison['winners']
    assert comparison['control_comparisons']
    assert gates['passed'], failed
    assert any(
        check['name'] == 'memory:reset_isolation_passed'
        or check['name'].endswith(':reset_isolation_passed')
        for check in gates['checks']
    )


def test_phase7_cli_docs_and_scope(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(['sweep-memory', '--help'])
    output = capsys.readouterr().out
    assert raised.value.code == 0
    assert '--output-dir' in output
    assert '--device' in output

    root = Path(__file__).resolve().parents[1]
    readme = (root / 'README.md').read_text(encoding='utf-8')
    assert '.\\.venv\\Scripts\\aegis-sarn.exe sweep-memory' in readme
    assert 'aegis-sarn compare-memory' in readme
    assert 'memory is disabled by default' in readme.lower()

    forbidden_modules = (
        'moe.py',
        'ssm.py',
        'retrieval.py',
        'tools.py',
        'multimodal.py',
        'persistent_memory.py',
    )
    sarn_dir = root / 'src' / 'aegis_sarn' / 'sarn'
    assert not any((sarn_dir / name).exists() for name in forbidden_modules)
