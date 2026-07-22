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
from aegis_sarn.sarn.data import GRAPH_TASK_NAMES, make_toy_task_batch
from aegis_sarn.sarn.generation import generate_greedy
from aegis_sarn.sarn.graph import GraphMessagePassing
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.utils import set_global_seed


def _graph_config(
    edge_mode: str = 'learned_dense',
    cycles: int = 1,
    top_k: int | None = None,
) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=20,
        d_model=32,
        n_layers=1,
        n_heads=4,
        ffn_hidden_dim=64,
        workspace_enabled=True,
        workspace_num_slots=4,
        graph_enabled=True,
        graph_num_cycles=cycles,
        graph_edge_mode=edge_mode,  # type: ignore[arg-type]
        graph_top_k=top_k,
    )


def test_graph_defaults_and_config_validation() -> None:
    default = ModelConfig()
    assert default.graph_enabled is False
    assert default.graph_num_cycles == 0
    assert default.graph_edge_mode == 'none'
    assert default.graph_top_k is None
    assert SARNDense(default).graph is None

    with pytest.raises(ConfigError, match='requires workspace'):
        ModelConfig(graph_enabled=True, graph_num_cycles=1)
    with pytest.raises(ConfigError, match='must be positive'):
        ModelConfig(
            workspace_enabled=True,
            workspace_num_slots=2,
            graph_enabled=True,
            graph_num_cycles=0,
        )
    with pytest.raises(ConfigError, match='graph_edge_mode'):
        ModelConfig(graph_edge_mode='other')  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match='cannot exceed'):
        ModelConfig(
            workspace_enabled=True,
            workspace_num_slots=2,
            graph_enabled=True,
            graph_num_cycles=1,
            graph_edge_mode='learned_sparse',
            graph_top_k=3,
        )
    with pytest.raises(ConfigError, match='requires graph_top_k'):
        ModelConfig(
            workspace_enabled=True,
            workspace_num_slots=2,
            graph_enabled=True,
            graph_num_cycles=1,
            graph_edge_mode='learned_sparse',
        )


def test_graph_shape_diagnostics_backward_and_sparse_top_k() -> None:
    config = _graph_config(cycles=2)
    graph = GraphMessagePassing(config)
    slots = torch.randn(2, 5, 4, 32, requires_grad=True)
    output, diagnostics = graph(slots)
    output.square().mean().backward()

    assert output.shape == slots.shape
    assert diagnostics.num_cycles == 2
    assert diagnostics.edge_mode == 'learned_dense'
    assert diagnostics.gate_mean.item() > 0
    assert diagnostics.message_norm.item() > 0
    assert slots.grad is not None and torch.isfinite(slots.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in graph.parameters()
    )

    sparse = GraphMessagePassing(
        _graph_config(edge_mode='learned_sparse', top_k=2)
    )
    sparse_output, sparse_diagnostics = sparse(slots.detach())
    assert sparse_output.shape == slots.shape
    assert sparse_diagnostics.top_k == 2


def test_null_graph_is_exact_noop_and_matches_workspace_control() -> None:
    null_config = _graph_config(edge_mode='none')
    graph = GraphMessagePassing(null_config)
    slots = torch.randn(2, 4, 32)
    output, diagnostics = graph(slots)
    torch.testing.assert_close(output, slots, rtol=0.0, atol=0.0)
    assert diagnostics.gate_mean.item() == 0.0
    assert diagnostics.message_norm.item() == 0.0

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
    set_global_seed(61)
    workspace_control = SARNDense(ModelConfig(**common)).eval()
    set_global_seed(61)
    graph_null = SARNDense(
        ModelConfig(
            **common,
            graph_enabled=True,
            graph_num_cycles=1,
            graph_edge_mode='none',
        )
    ).eval()
    inputs = torch.randint(0, 32, (2, 8))
    with torch.inference_mode():
        torch.testing.assert_close(
            workspace_control(inputs), graph_null(inputs), rtol=0.0, atol=0.0
        )
    assert graph_null.active_parameter_count() == (
        graph_null.count_parameters()
        - int(graph_null.graph_metrics()['graph_parameter_count'])
    )


def test_graph_model_forward_backward_and_control_paths() -> None:
    config = _graph_config(edge_mode='frozen_identity')
    model = SARNDense(config)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(inputs)
    loss = language_model_loss(logits, labels)
    loss.backward()

    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(loss)
    metrics = model.graph_metrics()
    assert metrics['graph_enabled'] is True
    assert metrics['graph_edge_mode'] == 'frozen_identity'
    assert int(metrics['graph_parameter_count']) > 0

    dense = SARNDense(
        ModelConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_hidden_dim=64,
        )
    )
    workspace_only = SARNDense(
        ModelConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_hidden_dim=64,
            workspace_enabled=True,
            workspace_num_slots=2,
        )
    )
    assert dense(inputs).shape == logits.shape
    assert workspace_only(inputs).shape == logits.shape
    assert dense.graph is None
    assert workspace_only.graph is None


def test_graph_checkpoint_and_cached_generation_parity(tmp_path: Path) -> None:
    config = _graph_config(cycles=2)
    model = SARNDense(config).eval()
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.inference_mode():
        expected = model(prompt)
    checkpoint = tmp_path / 'graph.pt'
    save_checkpoint(checkpoint, model, optimizer=None, step=6)
    loaded = load_checkpoint(checkpoint)

    assert loaded.model.config == config
    assert loaded.model.graph is not None
    loaded.model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(
            expected, loaded.model(prompt), rtol=0.0, atol=0.0
        )
    uncached = generate_greedy(loaded.model, prompt, 8, use_kv_cache=False)
    cached = generate_greedy(loaded.model, prompt, 8, use_kv_cache=True)
    torch.testing.assert_close(uncached, cached, rtol=0.0, atol=0.0)


def test_graph_structural_tasks_are_deterministic() -> None:
    for task_name in GRAPH_TASK_NAMES:
        first = make_toy_task_batch(
            task_name, 2, 12, vocab_size=32, seed=19, split='validation'
        )
        second = make_toy_task_batch(
            task_name, 2, 12, vocab_size=32, seed=19, split='validation'
        )
        assert first.task == task_name
        torch.testing.assert_close(first.input_ids, second.input_ids)
        torch.testing.assert_close(first.labels, second.labels)

    short = make_toy_task_batch(
        'length_extrapolation', 1, 12, vocab_size=32, seed=19, split='train'
    )
    long = make_toy_task_batch(
        'length_extrapolation', 1, 12, vocab_size=32, seed=19, split='validation'
    )
    assert not torch.equal(short.input_ids, long.input_ids)


@pytest.fixture(scope='module')
def graph_sweep_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp('phase6') / 'graph'
    exit_code = main(
        [
            'sweep-graph',
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


def test_graph_sweep_outputs_and_manifest_fields(graph_sweep_dir: Path) -> None:
    summary_path = graph_sweep_dir / 'graph-sweep-summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    required_graph = {
        'graph_enabled',
        'graph_num_cycles',
        'graph_edge_mode',
        'graph_top_k',
        'graph_gated_update',
        'graph_variant_name',
    }

    assert summary['command'] == 'sweep-graph'
    assert summary['metrics']['config_count'] == 6
    assert summary['metrics']['graph_enabled_count'] == 4
    assert (graph_sweep_dir / 'graph-sweep-summary.md').exists()
    for result in summary['results']:
        assert required_graph.issubset(result)
        assert len(result['task_metrics']) == len(GRAPH_TASK_NAMES)
        assert result['parameter_count'] > 0
        for manifest_path in result['manifest_paths'].values():
            manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            assert required_graph.issubset(manifest)
            assert manifest['graph_variant_name'] == result['graph_variant_name']


def test_compare_graph_and_phase6_gates(graph_sweep_dir: Path) -> None:
    report_dir = graph_sweep_dir / 'reports'
    exit_code = main(
        [
            'compare-graph',
            '--input',
            str(graph_sweep_dir),
            '--output-dir',
            str(report_dir),
            '--json',
        ]
    )
    gates = check_gates(graph_sweep_dir / 'graph-sweep-summary.json')
    failed = [check['name'] for check in gates['checks'] if not check['passed']]

    assert exit_code == 0
    assert (report_dir / 'graph-comparison.json').exists()
    assert (report_dir / 'graph-comparison.md').exists()
    comparison = json.loads(
        (report_dir / 'graph-comparison.json').read_text(encoding='utf-8')
    )
    assert 'best_balanced_graph' in comparison['winners']
    assert comparison['control_comparisons']
    assert gates['passed'], failed
    assert any(
        check['name'] == 'graph:workspace_control_present'
        for check in gates['checks']
    )


def test_phase6_cli_help_docs_and_scope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(['sweep-graph', '--help'])
    output = capsys.readouterr().out
    assert raised.value.code == 0
    assert '--output-dir' in output
    assert '--device' in output

    root = Path(__file__).resolve().parents[1]
    readme = (root / 'README.md').read_text(encoding='utf-8')
    assert '.\\.venv\\Scripts\\aegis-sarn.exe sweep-graph' in readme
    assert 'aegis-sarn compare-graph' in readme
    assert 'graph is disabled by default' in readme.lower()

    forbidden_modules = (
        'moe.py',
        'ssm.py',
        'retrieval.py',
        'tools.py',
        'multimodal.py',
    )
    sarn_dir = root / 'src' / 'aegis_sarn' / 'sarn'
    assert not any((sarn_dir / name).exists() for name in forbidden_modules)
