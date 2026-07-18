from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from aegis_sarn.cli import main
from aegis_sarn.config import ModelConfig, RunManifest, SeedConfig
from aegis_sarn.utils import write_json


def _manifest(
    *,
    run_id: str,
    command: str,
    model_config: dict[str, object],
    metrics: dict[str, object],
    artifacts: dict[str, str] | None = None,
) -> dict[str, object]:
    return RunManifest(
        run_id=run_id,
        run_name=command,
        created_at=datetime.now(timezone.utc).isoformat(),
        status='completed',
        model_config=model_config,
        training_config={},
        seed_config=SeedConfig(seed=123).to_dict(),
        runtime_config={'device': 'cpu'},
        decoding_config={'strategy': 'greedy', 'max_new_tokens': 2},
        package_version='0.1.0-test',
        git_commit='test-commit',
        device_info={'device': 'cpu'},
        command=command,
        command_args={'command': command},
        artifacts={} if artifacts is None else artifacts,
        metrics=metrics,
        trace_events=[],
        config_hash=f'sha256:{run_id}',
    ).to_dict()


def test_run_registry_entry_creation_and_listing(
    capsys: object, tmp_path: Path
) -> None:
    output_dir = tmp_path / 'runs'
    registry = output_dir / 'registry.json'
    exit_code = main(
        [
            'eval-toy',
            '--output-dir',
            str(output_dir),
            '--registry',
            str(registry),
            '--sequence-length',
            '8',
            '--batch-size',
            '2',
            '--max-new-tokens',
            '1',
            '--json',
        ]
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    registry_payload = json.loads(registry.read_text(encoding='utf-8'))
    entry = registry_payload['runs'][payload['run_id']]

    assert exit_code == 0
    assert entry['command_name'] == 'eval-toy'
    assert entry['status'] == 'completed'
    assert entry['manifest_path'] == payload['manifest_path']
    assert entry['metrics_summary']['validation_loss'] > 0

    assert main(['list-runs', '--registry', str(registry), '--json']) == 0
    listed = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert listed['runs'][0]['run_id'] == payload['run_id']


def test_report_baseline_creates_markdown_and_json(
    capsys: object, tmp_path: Path
) -> None:
    run_dir = tmp_path / 'phase2'
    report_dir = tmp_path / 'reports'
    registry = run_dir / 'runs' / 'registry.json'
    model_config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ffn_hidden_dim=32,
    ).to_dict()
    checkpoint = run_dir / 'train' / 'sarn-dense-smoke.pt'
    write_json(
        run_dir / 'train' / 'run-manifest.json',
        _manifest(
            run_id='train-1',
            command='train-smoke',
            model_config=model_config,
            artifacts={'checkpoint': str(checkpoint)},
            metrics={
                'initial_loss': 3.0,
                'final_loss': 1.0,
                'evaluation_loss': 0.9,
                'completed_step': 7,
                'dataset_name': 'toy/repeated_pattern',
                'task': 'repeated_pattern',
                'split': 'train-smoke',
                'examples': 2,
                'sequence_length': 8,
            },
        ),
    )
    write_json(
        run_dir / 'eval' / 'eval.json',
        _manifest(
            run_id='eval-1',
            command='eval-toy',
            model_config=model_config,
            artifacts={'checkpoint': str(checkpoint)},
            metrics={
                'validation_loss': 0.8,
                'perplexity': 2.22,
                'token_accuracy': 0.75,
                'generation_sample': '1 2',
                'generation_token_ids': [1, 2],
                'dataset_name': 'toy/repeated_pattern',
                'task': 'repeated_pattern',
                'split': 'validation',
                'examples': 2,
                'sequence_length': 8,
            },
        ),
    )
    write_json(
        run_dir / 'bench' / 'bench.json',
        _manifest(
            run_id='bench-1',
            command='bench',
            model_config=model_config,
            artifacts={'checkpoint': str(checkpoint)},
            metrics={
                'tokens_per_second': 123.0,
                'runtime_duration_ms': 4.5,
                'prompt_length': 4,
                'generated_tokens': 2,
                'repeats': 1,
                'parameter_count': 1000,
                'active_parameter_count': 1000,
                'parameter_memory_bytes': 4000,
                'approximate_kv_cache_bytes': 256,
            },
        ),
    )

    exit_code = main(
        [
            'report-baseline',
            '--run-dir',
            str(run_dir),
            '--output-dir',
            str(report_dir),
            '--registry',
            str(registry),
            '--json',
        ]
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    markdown_path = Path(payload['markdown_path'])
    json_path = Path(payload['json_path'])
    summary = json.loads(json_path.read_text(encoding='utf-8'))
    registry_payload = json.loads(registry.read_text(encoding='utf-8'))

    assert exit_code == 0
    assert markdown_path.exists()
    assert json_path.exists()
    assert {
        'schema_version',
        'run_id',
        'command',
        'timestamp',
        'git_commit_hash',
        'package_version',
        'config',
        'config_hash',
        'seed',
        'device',
        'status',
        'metrics',
        'artifacts',
        'limitations',
    }.issubset(summary)
    assert summary['evaluation']['token_accuracy'] == 0.75
    assert summary['benchmark']['tokens_per_second'] == 123.0
    assert 'not a real language benchmark' in markdown_path.read_text(encoding='utf-8')
    assert any(
        entry['command_name'] == 'report-baseline'
        for entry in registry_payload['runs'].values()
    )


def test_eval_multiseed_returns_aggregate_metrics(
    capsys: object, tmp_path: Path
) -> None:
    output_dir = tmp_path / 'multiseed'
    exit_code = main(
        [
            'eval-multiseed',
            '--output-dir',
            str(output_dir),
            '--seeds',
            '1',
            '2',
            '--sequence-length',
            '8',
            '--batch-size',
            '2',
            '--max-new-tokens',
            '1',
            '--json',
        ]
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    metrics = payload['metrics']

    assert exit_code == 0
    assert metrics['num_seeds'] == 2
    assert len(metrics['individual_seed_results']) == 2
    assert metrics['mean_validation_loss'] > 0
    assert metrics['std_validation_loss'] >= 0
    assert 0.0 <= metrics['mean_token_accuracy'] <= 1.0
    assert Path(payload['manifest_path']).exists()


def test_reproduce_phase2_runs_end_to_end(
    capsys: object, tmp_path: Path
) -> None:
    exit_code = main(
        [
            'reproduce-phase2',
            '--output-dir',
            str(tmp_path),
            '--device',
            'cpu',
            '--seed',
            '123',
            '--train-steps',
            '6',
            '--batch-size',
            '2',
            '--sequence-length',
            '8',
            '--d-model',
            '16',
            '--layers',
            '1',
            '--heads',
            '2',
            '--learning-rate',
            '0.02',
            '--max-new-tokens',
            '2',
            '--bench-repeats',
            '1',
        ]
    )
    summary = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    registry_payload = json.loads(
        Path(summary['registry_path']).read_text(encoding='utf-8')
    )
    commands = {entry['command_name'] for entry in registry_payload['runs'].values()}

    assert exit_code == 0
    checkpoint_path = Path(summary['checkpoint_path'])
    assert checkpoint_path == tmp_path / 'train' / 'sarn-dense-smoke.pt'
    assert checkpoint_path.exists()
    assert Path(summary['report_markdown_path']).exists()
    assert Path(summary['report_json_path']).exists()
    assert {'train-smoke', 'eval-toy', 'bench', 'report-baseline'}.issubset(commands)


def test_dataset_and_model_cards_exist_and_mention_toy_limitations() -> None:
    datasets = Path('docs/datasets.md').read_text(encoding='utf-8').lower()
    model_card = Path('docs/model-card-sarn-dense.md').read_text(
        encoding='utf-8'
    ).lower()

    assert 'not a real language benchmark' in datasets
    assert 'future replacement' in datasets
    assert 'toy-byte' in model_card
    assert 'control baseline' in model_card
    assert 'sarn-hybrid' in model_card
    assert 'not a claim' in model_card


def test_windows_powershell_readme_examples_do_not_use_linux_continuation() -> None:
    readme = Path('README.md').read_text(encoding='utf-8')
    start = readme.index('### Windows PowerShell')
    end = readme.index('Run the deterministic CPU smoke trainer', start)
    section = readme[start:end]

    assert not re.search(r'\\[ \t]*\r?\n', section)
    assert 'reproduce-phase2 --output-dir artifacts/phase2-check' in section
    assert 'list-runs --registry artifacts/phase2-check/runs/registry.json' in section
    assert 'report-baseline --help' in section
    assert '--checkpoint artifacts/phase2-check/train/sarn-dense-smoke.pt' in section
    assert '--num-seeds 3 --json' in section
