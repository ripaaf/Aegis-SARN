from __future__ import annotations

import json
from pathlib import Path

from aegis_sarn.cli import main


def test_cli_fake_backend_emits_json_trace(
    capsys: object, tmp_path: Path
) -> None:
    exit_code = main(
        [
            'run',
            '--backend',
            'fake',
            '--prompt',
            'hello',
            '--max-new-tokens',
            '3',
            '--output-dir',
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload['status'] == 'completed'
    assert payload['trace']
    assert payload['backend'] == 'fake'
    manifest_path = Path(payload['manifest_path'])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert_manifest_reproducibility_fields(manifest)
    assert manifest['trace_events'][0]['event_type'] == 'run.created'
    assert manifest['trace_events'][-1]['event_type'] == 'run.completed'


def test_eval_cli_writes_json_metrics(capsys: object, tmp_path: Path) -> None:
    exit_code = main(
        [
            'eval-toy',
            '--output-dir',
            str(tmp_path),
            '--sequence-length',
            '8',
            '--batch-size',
            '2',
            '--max-new-tokens',
            '2',
            '--json',
        ]
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    manifest_path = Path(payload['manifest_path'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert exit_code == 0
    assert manifest_path.parent == tmp_path
    assert manifest['metrics']['validation_loss'] > 0
    assert manifest['metrics']['perplexity'] > 0
    assert 0.0 <= manifest['metrics']['token_accuracy'] <= 1.0
    assert manifest['trace_events'][0]['event_type'] == 'eval.started'
    assert manifest['trace_events'][-1]['event_type'] == 'eval.completed'
    assert_manifest_reproducibility_fields(manifest)


def test_bench_cli_writes_json_metrics(capsys: object, tmp_path: Path) -> None:
    exit_code = main(
        [
            'bench',
            '--output-dir',
            str(tmp_path),
            '--prompt-length',
            '4',
            '--max-new-tokens',
            '2',
            '--repeats',
            '1',
            '--use-kv-cache',
            '--json',
        ]
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    manifest_path = Path(payload['manifest_path'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    metrics = manifest['metrics']
    assert exit_code == 0
    assert metrics['tokens_per_second'] > 0
    assert metrics['active_parameter_count'] == metrics['parameter_count']
    assert metrics['generated_tokens'] == 2
    assert manifest['trace_events'][0]['event_type'] == 'bench.started'
    assert manifest['trace_events'][-1]['event_type'] == 'bench.completed'
    assert_manifest_reproducibility_fields(manifest)


def assert_manifest_reproducibility_fields(manifest: dict[str, object]) -> None:
    required = {
        'command_args',
        'config_hash',
        'created_at',
        'device_info',
        'git_commit',
        'metrics',
        'package_version',
        'seed_config',
        'trace_events',
    }
    assert required.issubset(manifest)
