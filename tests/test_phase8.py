from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from aegis_sarn.cli import main
from aegis_sarn.config import ConfigError, ModelConfig
from aegis_sarn.eval import language_model_loss
from aegis_sarn.phase3 import check_gates
from aegis_sarn.sarn.checkpoint import load_checkpoint, save_checkpoint
from aegis_sarn.sarn.data import TOY_TASK_NAMES
from aegis_sarn.sarn.experts import SparseExpertFFN
from aegis_sarn.sarn.generation import generate_greedy
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.utils import set_global_seed


def _expert_config(num_experts: int = 4, top_k: int = 1) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=20,
        d_model=32,
        n_layers=1,
        n_heads=4,
        ffn_hidden_dim=64,
        experts_enabled=True,
        expert_num_experts=num_experts,
        expert_top_k=top_k,
    )


def test_expert_defaults_and_config_validation() -> None:
    default = ModelConfig()
    assert default.experts_enabled is False
    assert default.expert_num_experts == 0
    assert default.expert_top_k == 1
    assert default.expert_replaces_ffn is True
    assert all(block.expert_ffn is None for block in SARNDense(default).blocks)

    with pytest.raises(ConfigError, match='greater than one'):
        ModelConfig(experts_enabled=True, expert_num_experts=1)
    with pytest.raises(ConfigError, match='cannot exceed'):
        ModelConfig(
            experts_enabled=True,
            expert_num_experts=2,
            expert_top_k=3,
        )
    with pytest.raises(ConfigError, match='expert_top_k'):
        ModelConfig(expert_top_k=0)
    with pytest.raises(ConfigError, match='capacity_factor'):
        ModelConfig(expert_capacity_factor=0.0)
    with pytest.raises(ConfigError, match='hidden_dim'):
        ModelConfig(expert_hidden_dim=0)
    with pytest.raises(ConfigError, match='router_noise'):
        ModelConfig(expert_router_noise=-0.1)
    with pytest.raises(ConfigError, match='load_balance_weight'):
        ModelConfig(expert_load_balance_weight=-0.1)
    with pytest.raises(ConfigError, match='layer_frequency'):
        ModelConfig(expert_layer_frequency=0)


def test_sparse_expert_shapes_top_k_diagnostics_and_backward() -> None:
    config = ModelConfig(
        **{
            **_expert_config(top_k=2).to_dict(),
            'expert_load_balance_weight': 0.01,
        }
    )
    module = SparseExpertFFN(config)
    inputs = torch.randn(2, 6, config.d_model, requires_grad=True)
    output, diagnostics = module(inputs)
    assert module.last_auxiliary_loss is not None
    (output.square().mean() + module.last_auxiliary_loss).backward()

    assert output.shape == inputs.shape
    assert module.last_top_k_indices is not None
    assert module.last_top_k_indices.shape == (12, 2)
    assert int(module.last_top_k_indices.min()) >= 0
    assert int(module.last_top_k_indices.max()) < config.expert_num_experts
    assert torch.all(
        module.last_top_k_indices[:, 0]
        != module.last_top_k_indices[:, 1]
    )
    values = diagnostics.to_dict()
    for name in (
        'expert_router_entropy',
        'expert_load_balance_score',
        'expert_max_load_fraction',
        'expert_min_load_fraction',
        'expert_dropped_token_fraction',
    ):
        assert math.isfinite(float(values[name]))
    assert diagnostics.active_experts <= config.expert_num_experts
    assert values['expert_dropped_token_fraction'] == 0.0
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert module.router.weight.grad is not None
    assert torch.isfinite(module.router.weight.grad).all()
    assert module.active_parameter_count() <= module.count_parameters()


def test_expert_null_exactly_matches_dense_control() -> None:
    common = {
        'vocab_size': 32,
        'max_seq_len': 20,
        'd_model': 32,
        'n_layers': 2,
        'n_heads': 4,
        'ffn_hidden_dim': 64,
    }
    set_global_seed(81)
    dense = SARNDense(ModelConfig(**common)).eval()
    set_global_seed(81)
    null = SARNDense(
        ModelConfig(
            **common,
            experts_enabled=True,
            expert_num_experts=2,
            expert_top_k=1,
            expert_replaces_ffn=False,
        )
    ).eval()
    inputs = torch.randint(0, 32, (2, 8))

    with torch.inference_mode():
        torch.testing.assert_close(dense(inputs), null(inputs), rtol=0.0, atol=0.0)
    assert dense.count_parameters() == null.count_parameters()
    assert null.expert_metrics()['expert_parameter_count'] == 0


def test_expert_model_forward_backward_frequency_and_combination() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=20,
        d_model=32,
        n_layers=3,
        n_heads=4,
        ffn_hidden_dim=64,
        experts_enabled=True,
        expert_num_experts=4,
        expert_top_k=1,
        expert_layer_frequency=2,
    )
    model = SARNDense(config)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(inputs)
    loss = language_model_loss(logits, labels) + model.expert_auxiliary_loss()
    loss.backward()

    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(loss)
    assert sum(block.expert_ffn is not None for block in model.blocks) == 2
    metrics = model.expert_metrics()
    assert metrics['experts_enabled'] is True
    assert metrics['expert_layer_count'] == 2
    assert int(metrics['expert_parameter_count']) > 0
    assert 0 < model.active_parameter_count() <= model.count_parameters()

    combined = SARNDense(
        ModelConfig(
            vocab_size=32,
            max_seq_len=20,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_hidden_dim=64,
            attention_type='gqa',
            n_kv_heads=2,
            workspace_enabled=True,
            workspace_num_slots=4,
            graph_enabled=True,
            graph_num_cycles=1,
            graph_edge_mode='learned_dense',
            memory_enabled=True,
            memory_num_slots=4,
            memory_write_mode='gated',
            memory_read_mode='attention',
            experts_enabled=True,
            expert_num_experts=2,
            expert_top_k=1,
        )
    )
    assert combined(inputs).shape == logits.shape


def test_expert_checkpoint_generation_and_weight_immutability(
    tmp_path: Path,
) -> None:
    config = _expert_config(num_experts=4, top_k=2)
    model = SARNDense(config).eval()
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    with torch.inference_mode():
        expected = model(prompt)
        model.forward_with_cache(prompt, use_cache=True)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0.0, atol=0.0)

    checkpoint = tmp_path / 'experts.pt'
    save_checkpoint(checkpoint, model, optimizer=None, step=8)
    loaded = load_checkpoint(checkpoint)
    assert loaded.model.config == config
    assert loaded.model.blocks[0].expert_ffn is not None
    loaded.model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(
            expected, loaded.model(prompt), rtol=0.0, atol=0.0
        )

    uncached = generate_greedy(loaded.model, prompt, 8, use_kv_cache=False)
    cached = generate_greedy(loaded.model, prompt, 8, use_kv_cache=True)
    torch.testing.assert_close(uncached, cached, rtol=0.0, atol=0.0)


@pytest.fixture(scope='module')
def expert_sweep_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp('phase8') / 'experts'
    exit_code = main(
        [
            'sweep-experts',
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


def test_expert_sweep_outputs_and_manifest_fields(expert_sweep_dir: Path) -> None:
    summary_path = expert_sweep_dir / 'expert-sweep-summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    required_expert = {
        'experts_enabled',
        'expert_num_experts',
        'expert_top_k',
        'expert_capacity_factor',
        'expert_hidden_dim',
        'expert_router_noise',
        'expert_load_balance_weight',
        'expert_use_shared_expert',
        'expert_layer_frequency',
        'expert_replaces_ffn',
        'expert_variant_name',
    }

    assert summary['command'] == 'sweep-experts'
    assert summary['metrics']['config_count'] == 5
    assert summary['metrics']['experts_enabled_count'] == 4
    assert summary['metrics']['routed_variant_count'] == 3
    assert (expert_sweep_dir / 'expert-sweep-summary.md').exists()
    assert (expert_sweep_dir / 'runs' / 'registry.json').exists()
    for result in summary['results']:
        assert required_expert.issubset(result)
        assert len(result['task_metrics']) == len(TOY_TASK_NAMES)
        assert 0 < result['active_parameter_count'] <= result['parameter_count']
        assert (
            result['expert_active_parameter_count']
            <= result['expert_parameter_count']
        )
        for manifest_path in result['manifest_paths'].values():
            manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            assert required_expert.issubset(manifest)
            assert (
                manifest['expert_variant_name']
                == result['expert_variant_name']
            )


def test_compare_experts_and_phase8_gates(expert_sweep_dir: Path) -> None:
    report_dir = expert_sweep_dir / 'reports'
    exit_code = main(
        [
            'compare-experts',
            '--input',
            str(expert_sweep_dir),
            '--output-dir',
            str(report_dir),
            '--json',
        ]
    )
    gates = check_gates(expert_sweep_dir / 'expert-sweep-summary.json')
    failed = [check['name'] for check in gates['checks'] if not check['passed']]

    assert exit_code == 0
    assert (report_dir / 'expert-comparison.json').exists()
    assert (report_dir / 'expert-comparison.md').exists()
    comparison = json.loads(
        (report_dir / 'expert-comparison.json').read_text(encoding='utf-8')
    )
    assert 'best_routing_balance' in comparison['winners']
    assert 'best_balanced_expert' in comparison['winners']
    assert comparison['control_comparisons']
    assert gates['passed'], failed
    assert any(
        check['name'] == 'experts:routed_variant_present'
        for check in gates['checks']
    )


def test_phase8_cli_docs_and_scope(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(['sweep-experts', '--help'])
    output = capsys.readouterr().out
    assert raised.value.code == 0
    assert '--output-dir' in output
    assert '--device' in output

    root = Path(__file__).resolve().parents[1]
    readme = (root / 'README.md').read_text(encoding='utf-8')
    assert '.\\.venv\\Scripts\\aegis-sarn.exe sweep-experts' in readme
    assert 'aegis-sarn compare-experts' in readme
    assert 'experts are disabled by default' in readme.lower()

    forbidden_modules = (
        'ssm.py',
        'mamba.py',
        'retrieval.py',
        'tools.py',
        'multimodal.py',
        'persistent_memory.py',
    )
    sarn_dir = root / 'src' / 'aegis_sarn' / 'sarn'
    assert not any((sarn_dir / name).exists() for name in forbidden_modules)
