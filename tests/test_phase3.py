from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from aegis_sarn.cli import main
from aegis_sarn.sarn.data import TOY_TASK_NAMES, make_toy_task_batch


def _run_tiny_sweep(capsys: object, tmp_path: Path) -> Path:
    output_dir = tmp_path / 'phase3-sweep'
    exit_code = main(
        [
            'sweep-baseline',
            '--output-dir',
            str(output_dir),
            '--device',
            'cpu',
            '--seed',
            '123',
            '--train-steps',
            '2',
            '--batch-size',
            '1',
            '--max-new-tokens',
            '1',
            '--bench-repeats',
            '1',
            '--json',
        ]
    )
    capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 0
    return output_dir


def test_sweep_baseline_creates_summary_json_and_markdown(
    capsys: object, tmp_path: Path
) -> None:
    output_dir = _run_tiny_sweep(capsys, tmp_path)
    summary_path = output_dir / 'sweep-summary.json'
    markdown_path = output_dir / 'sweep-summary.md'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))

    assert summary_path.exists()
    assert markdown_path.exists()
    assert summary['command'] == 'sweep-baseline'
    assert len(summary['results']) == 3
    assert summary['metrics']['config_count'] == 3


def test_sweep_baseline_records_parameter_count_and_metrics(
    capsys: object, tmp_path: Path
) -> None:
    output_dir = _run_tiny_sweep(capsys, tmp_path)
    summary = json.loads((output_dir / 'sweep-summary.json').read_text(encoding='utf-8'))
    result = summary['results'][0]

    assert result['parameter_count'] > 0
    assert result['active_parameter_count'] == result['parameter_count']
    assert result['parameter_memory_bytes'] > 0
    assert result['tokens_per_second'] > 0
    assert result['eval_loss'] > 0
    assert 0.0 <= result['token_accuracy'] <= 1.0


def test_compare_baselines_reads_sweep_and_creates_reports(
    capsys: object, tmp_path: Path
) -> None:
    sweep_dir = _run_tiny_sweep(capsys, tmp_path)
    report_dir = tmp_path / 'reports'

    exit_code = main(
        [
            'compare-baselines',
            '--input',
            str(sweep_dir),
            '--output-dir',
            str(report_dir),
            '--json',
        ]
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert exit_code == 0
    assert Path(payload['comparison_json_path']).exists()
    assert Path(payload['comparison_markdown_path']).exists()
    assert 'best_balanced_config' in payload['winners']


def test_check_gates_passes_with_valid_summary(
    capsys: object, tmp_path: Path
) -> None:
    sweep_dir = _run_tiny_sweep(capsys, tmp_path)
    exit_code = main(
        [
            'check-gates',
            '--summary',
            str(sweep_dir / 'sweep-summary.json'),
            '--max-eval-loss',
            '100',
            '--min-token-accuracy',
            '0',
            '--min-tokens-per-second',
            '0',
            '--json',
        ]
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert exit_code == 0
    assert payload['passed'] is True


def test_check_gates_fails_with_bad_threshold(
    capsys: object, tmp_path: Path
) -> None:
    sweep_dir = _run_tiny_sweep(capsys, tmp_path)
    exit_code = main(
        [
            'check-gates',
            '--summary',
            str(sweep_dir / 'sweep-summary.json'),
            '--max-eval-loss',
            '0',
            '--json',
        ]
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert exit_code == 1
    assert payload['passed'] is False


def test_manifests_include_required_phase3_fields(
    capsys: object, tmp_path: Path
) -> None:
    sweep_dir = _run_tiny_sweep(capsys, tmp_path)
    summary = json.loads((sweep_dir / 'sweep-summary.json').read_text(encoding='utf-8'))
    manifest_path = Path(summary['results'][0]['manifest_paths']['train'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    required = {
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
    }

    assert required.issubset(manifest)


def test_artifact_policy_doc_exists() -> None:
    text = Path('docs/artifacts.md').read_text(encoding='utf-8').lower()

    assert 'checkpoints' in text
    assert 'artifacts/' in text
    assert 'direct deterministic writes' in text


def test_stronger_toy_tasks_are_deterministic() -> None:
    for task_name in TOY_TASK_NAMES:
        left = make_toy_task_batch(
            task_name,
            batch_size=2,
            sequence_length=8,
            vocab_size=32,
            seed=17,
            split='validation',
        )
        right = make_toy_task_batch(
            task_name,
            batch_size=2,
            sequence_length=8,
            vocab_size=32,
            seed=17,
            split='validation',
        )
        assert left.task == right.task
        assert torch.equal(left.input_ids, right.input_ids)
        assert torch.equal(left.labels, right.labels)


def test_eval_tasks_reports_per_task_metrics(capsys: object, tmp_path: Path) -> None:
    exit_code = main(
        [
            'eval-tasks',
            '--output-dir',
            str(tmp_path),
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
    assert metrics['task_count'] == len(TOY_TASK_NAMES)
    assert len(metrics['tasks']) == len(TOY_TASK_NAMES)
    assert metrics['aggregate_validation_loss'] > 0
    assert 0.0 <= metrics['aggregate_token_accuracy'] <= 1.0


def test_readme_powershell_examples_do_not_use_linux_backslash() -> None:
    readme = Path('README.md').read_text(encoding='utf-8')
    start = readme.index('### Windows PowerShell')
    end = readme.index('Normal command examples', start)
    section = readme[start:end]

    assert '.\\.venv\\Scripts\\aegis-sarn.exe' in section
    assert '`' in section
    assert not re.search(r'\\[ \t]*\r?\n', section)


def test_no_sarn_hybrid_source_modules_are_required() -> None:
    source_files = [path for path in Path('src').rglob('*.py')]

    assert source_files
    assert all('hybrid' not in str(path).lower() for path in source_files)
