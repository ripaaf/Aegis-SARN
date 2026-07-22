'''Command-line entry points for the SARN-Dense baseline and runtime paths.'''

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from aegis_sarn.aegis import FakeBackend, RunRequest, SARNBackend, SessionController
from aegis_sarn.config import (
    ArtifactConfig,
    DecodingConfig,
    ModelConfig,
    RunManifest,
    SeedConfig,
    TrainingConfig,
)
from aegis_sarn.eval import benchmark_generation, evaluate_tasks, evaluate_toy
from aegis_sarn.phase3 import check_gates, compare_baselines, run_baseline_sweep
from aegis_sarn.phase4 import compare_attention, run_attention_sweep
from aegis_sarn.phase5 import compare_workspace, run_workspace_sweep
from aegis_sarn.registry import record_manifest, registry_entries
from aegis_sarn.reporting import write_baseline_report
from aegis_sarn.sarn.checkpoint import load_checkpoint
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.sarn.training import run_smoke_training
from aegis_sarn.utils import (
    config_hash,
    device_info,
    git_commit,
    normalize_json,
    package_version,
    set_global_seed,
    sha256_file,
    write_json,
)


def _add_decoding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--strategy', choices=('greedy', 'sample'), default='greedy')
    parser.add_argument('--max-new-tokens', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--top-p', type=float)
    parser.add_argument('--stop-token-id', type=int)
    parser.add_argument('--use-kv-cache', action='store_true')
    parser.add_argument('--seed', type=int, default=7)


def _add_registry_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--registry',
        type=Path,
        help='run registry path; defaults to <output-dir>/registry.json',
    )


def _decoding_config(
    arguments: argparse.Namespace, seed_override: int | None = None
) -> DecodingConfig:
    return DecodingConfig(
        strategy=arguments.strategy,
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
        top_p=arguments.top_p,
        stop_token_id=arguments.stop_token_id,
        use_kv_cache=arguments.use_kv_cache,
        seed=arguments.seed if seed_override is None else seed_override,
    )


def _safe_args(arguments: argparse.Namespace) -> dict[str, object]:
    values = normalize_json(vars(arguments))
    if isinstance(values, dict) and 'prompt' in values:
        prompt = str(values['prompt'])
        values['prompt'] = '<redacted>'
        values['prompt_length_chars'] = len(prompt)
    return values


def _registry_path(arguments: argparse.Namespace) -> Path:
    registry = getattr(arguments, 'registry', None)
    if registry is not None:
        return registry
    return getattr(arguments, 'output_dir', Path('runs')) / 'registry.json'


def _report_registry_path(arguments: argparse.Namespace) -> Path:
    registry = getattr(arguments, 'registry', None)
    if registry is not None:
        return registry
    return arguments.run_dir / 'runs' / 'registry.json'


def _record_manifest(arguments: argparse.Namespace, manifest_path: Path) -> None:
    record_manifest(_registry_path(arguments), manifest_path)


def _latest_train_checkpoint(registry: Path | None) -> Path | None:
    if registry is None or not registry.exists():
        return None
    candidates = [
        entry
        for entry in registry_entries(registry)
        if entry.get('command_name') == 'train-smoke' and entry.get('checkpoint_path')
    ]
    if not candidates:
        return None
    return Path(str(candidates[-1]['checkpoint_path']))


def _checkpoint_argument(arguments: argparse.Namespace) -> Path | None:
    checkpoint = getattr(arguments, 'checkpoint', None)
    if checkpoint is not None:
        return checkpoint
    return _latest_train_checkpoint(getattr(arguments, 'registry', None))


def _load_model(
    checkpoint: Path | None, device: str, seed: int
) -> tuple[SARNDense, dict[str, str]]:
    artifacts: dict[str, str] = {}
    if checkpoint is not None:
        model = load_checkpoint(checkpoint, map_location=device).model
        artifacts = {
            'checkpoint': str(checkpoint),
            'checkpoint_digest': sha256_file(checkpoint),
        }
    else:
        set_global_seed(SeedConfig(seed=seed))
        model = SARNDense(ModelConfig())
    return model, artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='aegis-sarn')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run', help='run a prompt through Aegis')
    run_parser.add_argument('--prompt', required=True)
    run_parser.add_argument('--checkpoint', type=Path)
    run_parser.add_argument('--backend', choices=('sarn', 'fake'), default='sarn')
    run_parser.add_argument('--max-prompt-tokens', type=int, default=48)
    run_parser.add_argument('--wall-time-ms', type=int, default=30_000)
    run_parser.add_argument('--device', default='cpu')
    run_parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    _add_decoding_arguments(run_parser)
    _add_registry_argument(run_parser)

    train_parser = subparsers.add_parser(
        'train-smoke', help='overfit a generated batch and write a checkpoint'
    )
    train_parser.add_argument('--output-dir', type=Path, default=Path('artifacts/phase1'))
    train_parser.add_argument('--steps', type=int, default=60)
    train_parser.add_argument('--batch-size', type=int, default=8)
    train_parser.add_argument('--sequence-length', type=int, default=24)
    train_parser.add_argument('--d-model', type=int, default=48)
    train_parser.add_argument('--layers', type=int, default=2)
    train_parser.add_argument('--heads', type=int, default=4)
    train_parser.add_argument(
        '--attention-type', choices=('mha', 'gqa'), default='mha'
    )
    train_parser.add_argument('--n-kv-heads', type=int)
    train_parser.add_argument('--workspace-enabled', action='store_true')
    train_parser.add_argument('--workspace-num-slots', type=int, default=0)
    train_parser.add_argument('--workspace-no-writeback', action='store_true')
    train_parser.add_argument('--workspace-dropout', type=float, default=0.0)
    train_parser.add_argument(
        '--workspace-read-mode',
        choices=('cross_attention', 'mean_pool'),
        default='cross_attention',
    )
    train_parser.add_argument('--learning-rate', type=float, default=5e-3)
    train_parser.add_argument('--seed', type=int, default=7)
    train_parser.add_argument('--device', default='cpu')
    _add_registry_argument(train_parser)

    eval_parser = subparsers.add_parser('eval-toy', help='evaluate toy metrics')
    eval_parser.add_argument('--checkpoint', type=Path)
    eval_parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    eval_parser.add_argument('--batch-size', type=int, default=8)
    eval_parser.add_argument('--sequence-length', type=int, default=24)
    eval_parser.add_argument('--device', default='cpu')
    eval_parser.add_argument('--json', action='store_true')
    _add_decoding_arguments(eval_parser)
    _add_registry_argument(eval_parser)

    multiseed_parser = subparsers.add_parser(
        'eval-multiseed', help='run toy evaluation across multiple seeds'
    )
    multiseed_parser.add_argument('--checkpoint', type=Path)
    multiseed_parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    multiseed_parser.add_argument('--batch-size', type=int, default=8)
    multiseed_parser.add_argument('--sequence-length', type=int, default=24)
    multiseed_parser.add_argument('--device', default='cpu')
    multiseed_parser.add_argument('--seeds', nargs='*', type=int)
    multiseed_parser.add_argument('--num-seeds', type=int, default=3)
    multiseed_parser.add_argument('--json', action='store_true')
    _add_decoding_arguments(multiseed_parser)
    _add_registry_argument(multiseed_parser)

    tasks_parser = subparsers.add_parser(
        'eval-tasks', help='evaluate deterministic toy task suite'
    )
    tasks_parser.add_argument('--checkpoint', type=Path)
    tasks_parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    tasks_parser.add_argument('--batch-size', type=int, default=4)
    tasks_parser.add_argument('--sequence-length', type=int, default=16)
    tasks_parser.add_argument('--tasks', nargs='*')
    tasks_parser.add_argument('--device', default='cpu')
    tasks_parser.add_argument('--json', action='store_true')
    _add_decoding_arguments(tasks_parser)
    _add_registry_argument(tasks_parser)

    bench_parser = subparsers.add_parser('bench', help='benchmark dense generation')
    bench_parser.add_argument('--checkpoint', type=Path)
    bench_parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    bench_parser.add_argument('--prompt-length', type=int, default=16)
    bench_parser.add_argument('--repeats', type=int, default=3)
    bench_parser.add_argument('--device', default='cpu')
    bench_parser.add_argument('--json', action='store_true')
    _add_decoding_arguments(bench_parser)
    _add_registry_argument(bench_parser)

    list_parser = subparsers.add_parser('list-runs', help='list registered runs')
    list_parser.add_argument('--registry', type=Path, default=Path('runs/registry.json'))
    list_parser.add_argument('--json', action='store_true')

    report_parser = subparsers.add_parser(
        'report-baseline', help='generate the SARN-Dense Phase 2 baseline report'
    )
    report_parser.add_argument('--run-dir', type=Path, default=Path('artifacts/phase2-repro'))
    report_parser.add_argument('--output-dir', type=Path, default=Path('artifacts/reports'))
    report_parser.add_argument('--registry', type=Path)
    report_parser.add_argument('--json', action='store_true')

    reproduce_parser = subparsers.add_parser(
        'reproduce-phase2', help='run the small CPU Phase 2 reproduction pipeline'
    )
    reproduce_parser.add_argument('--output-dir', type=Path, default=Path('artifacts/phase2-repro'))
    reproduce_parser.add_argument('--device', default='cpu')
    reproduce_parser.add_argument('--seed', type=int, default=123)
    reproduce_parser.add_argument('--train-steps', type=int, default=20)
    reproduce_parser.add_argument('--batch-size', type=int, default=4)
    reproduce_parser.add_argument('--sequence-length', type=int, default=16)
    reproduce_parser.add_argument('--d-model', type=int, default=32)
    reproduce_parser.add_argument('--layers', type=int, default=1)
    reproduce_parser.add_argument('--heads', type=int, default=4)
    reproduce_parser.add_argument('--learning-rate', type=float, default=8e-3)
    reproduce_parser.add_argument('--max-new-tokens', type=int, default=4)
    reproduce_parser.add_argument('--bench-repeats', type=int, default=1)

    sweep_parser = subparsers.add_parser(
        'sweep-baseline', help='train/evaluate/benchmark tiny SARN-Dense configs'
    )
    sweep_parser.add_argument('--output-dir', type=Path, default=Path('artifacts/phase3-sweep'))
    sweep_parser.add_argument('--device', default='cpu')
    sweep_parser.add_argument('--seed', type=int, default=123)
    sweep_parser.add_argument('--train-steps', type=int)
    sweep_parser.add_argument('--batch-size', type=int, default=2)
    sweep_parser.add_argument('--max-new-tokens', type=int, default=2)
    sweep_parser.add_argument('--bench-repeats', type=int, default=1)
    sweep_parser.add_argument('--json', action='store_true')

    compare_parser = subparsers.add_parser(
        'compare-baselines', help='compare Phase 3 sweep outputs'
    )
    compare_parser.add_argument('--input', type=Path, default=Path('artifacts/phase3-sweep'))
    compare_parser.add_argument('--output-dir', type=Path, default=Path('artifacts/reports'))
    compare_parser.add_argument('--json', action='store_true')

    attention_sweep_parser = subparsers.add_parser(
        'sweep-attention',
        help='compare matched MHA and experimental GQA SARN-Dense variants',
    )
    attention_sweep_parser.add_argument(
        '--output-dir', type=Path, default=Path('artifacts/phase4-attention')
    )
    attention_sweep_parser.add_argument('--device', default='cpu')
    attention_sweep_parser.add_argument('--seed', type=int, default=123)
    attention_sweep_parser.add_argument('--train-steps', type=int, default=8)
    attention_sweep_parser.add_argument('--batch-size', type=int, default=2)
    attention_sweep_parser.add_argument('--sequence-length', type=int, default=16)
    attention_sweep_parser.add_argument('--max-new-tokens', type=int, default=2)
    attention_sweep_parser.add_argument('--bench-repeats', type=int, default=1)
    attention_sweep_parser.add_argument('--json', action='store_true')

    attention_compare_parser = subparsers.add_parser(
        'compare-attention', help='compare Phase 4 MHA/GQA sweep outputs'
    )
    attention_compare_parser.add_argument(
        '--input', type=Path, default=Path('artifacts/phase4-attention')
    )
    attention_compare_parser.add_argument(
        '--output-dir', type=Path, default=Path('artifacts/reports')
    )
    attention_compare_parser.add_argument('--json', action='store_true')

    workspace_sweep_parser = subparsers.add_parser(
        'sweep-workspace',
        help='compare matched dense and experimental latent-workspace variants',
    )
    workspace_sweep_parser.add_argument(
        '--output-dir', type=Path, default=Path('artifacts/phase5-workspace')
    )
    workspace_sweep_parser.add_argument('--device', default='cpu')
    workspace_sweep_parser.add_argument('--seed', type=int, default=123)
    workspace_sweep_parser.add_argument('--train-steps', type=int, default=8)
    workspace_sweep_parser.add_argument('--batch-size', type=int, default=2)
    workspace_sweep_parser.add_argument('--sequence-length', type=int, default=16)
    workspace_sweep_parser.add_argument('--max-new-tokens', type=int, default=2)
    workspace_sweep_parser.add_argument('--bench-repeats', type=int, default=1)
    workspace_sweep_parser.add_argument('--json', action='store_true')

    workspace_compare_parser = subparsers.add_parser(
        'compare-workspace', help='compare Phase 5 workspace sweep outputs'
    )
    workspace_compare_parser.add_argument(
        '--input', type=Path, default=Path('artifacts/phase5-workspace')
    )
    workspace_compare_parser.add_argument(
        '--output-dir', type=Path, default=Path('artifacts/reports')
    )
    workspace_compare_parser.add_argument('--json', action='store_true')

    gates_parser = subparsers.add_parser(
        'check-gates', help='check basic experiment quality gates'
    )
    gates_parser.add_argument('--summary', type=Path, required=True)
    gates_parser.add_argument('--max-eval-loss', type=float)
    gates_parser.add_argument('--min-token-accuracy', type=float)
    gates_parser.add_argument('--min-tokens-per-second', type=float)
    gates_parser.add_argument('--json', action='store_true')
    return parser


def _run_command(arguments: argparse.Namespace) -> int:
    model: SARNDense | None = None
    artifacts: dict[str, str] = {}
    if arguments.backend == 'fake':
        backend = FakeBackend()
    else:
        model, artifacts = _load_model(
            _checkpoint_argument(arguments), arguments.device, arguments.seed
        )
        backend = SARNBackend(model=model, device=arguments.device)

    request = RunRequest(
        prompt=arguments.prompt,
        max_new_tokens=arguments.max_new_tokens,
        max_prompt_tokens=arguments.max_prompt_tokens,
        wall_time_ms=arguments.wall_time_ms,
        seed=arguments.seed,
        decoding_strategy=arguments.strategy,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
        top_p=arguments.top_p,
        stop_token_id=arguments.stop_token_id,
        use_kv_cache=arguments.use_kv_cache,
    )
    result = SessionController(backend).run(request)
    configuration = {
        'request': request.to_dict(),
        'model': {} if model is None else model.config.to_dict(),
    }
    manifest = RunManifest(
        run_id=result.run_id,
        run_name='run',
        created_at=(
            result.trace[0].timestamp
            if result.trace
            else datetime.now(timezone.utc).isoformat()
        ),
        status=result.status,
        model_config={} if model is None else model.config.to_dict(),
        training_config={},
        seed_config=SeedConfig(seed=arguments.seed).to_dict(),
        runtime_config={
            'device': arguments.device,
            'max_prompt_tokens': arguments.max_prompt_tokens,
            'wall_time_ms': arguments.wall_time_ms,
        },
        decoding_config=_decoding_config(arguments).to_dict(),
        package_version=package_version(),
        git_commit=git_commit(),
        device_info=device_info(arguments.device),
        command='run',
        command_args=_safe_args(arguments),
        artifacts=artifacts,
        metrics={'status': result.status, **result.usage.to_dict()},
        trace_events=[event.to_dict() for event in result.trace],
        config_hash=config_hash(configuration),
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = arguments.output_dir / f'run-{result.run_id}.json'
    manifest_payload = manifest.to_dict()
    write_json(manifest_path, manifest_payload)
    _record_manifest(arguments, manifest_path)
    result.manifest_path = str(manifest_path)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.status == 'completed' else 1


def _train_command(arguments: argparse.Namespace) -> int:
    model_config = ModelConfig(
        vocab_size=256,
        max_seq_len=max(64, arguments.sequence_length),
        d_model=arguments.d_model,
        n_layers=arguments.layers,
        n_heads=arguments.heads,
        attention_type=arguments.attention_type,
        n_kv_heads=arguments.n_kv_heads,
        ffn_hidden_dim=arguments.d_model * 3,
        workspace_enabled=arguments.workspace_enabled,
        workspace_num_slots=arguments.workspace_num_slots,
        workspace_gated_writeback=not arguments.workspace_no_writeback,
        workspace_dropout=arguments.workspace_dropout,
        workspace_read_mode=arguments.workspace_read_mode,
    )
    training_config = TrainingConfig(
        learning_rate=arguments.learning_rate,
        batch_size=arguments.batch_size,
        sequence_length=arguments.sequence_length,
        max_steps=arguments.steps,
        device=arguments.device,
    )
    result = run_smoke_training(
        model_config=model_config,
        training_config=training_config,
        seed_config=SeedConfig(seed=arguments.seed),
        artifact_config=ArtifactConfig(output_dir=arguments.output_dir),
        command_args=_safe_args(arguments),
    )
    _record_manifest(arguments, result.manifest_path)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.final_loss < result.initial_loss else 2


def _eval_command(arguments: argparse.Namespace) -> int:
    model, artifacts = _load_model(
        _checkpoint_argument(arguments), arguments.device, arguments.seed
    )
    result = evaluate_toy(
        model=model,
        output_dir=arguments.output_dir,
        seed_config=SeedConfig(seed=arguments.seed),
        decoding_config=_decoding_config(arguments),
        device=arguments.device,
        batch_size=arguments.batch_size,
        sequence_length=arguments.sequence_length,
        command_args=_safe_args(arguments),
        artifacts=artifacts,
    )
    _record_manifest(arguments, result.manifest_path)
    if arguments.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
    else:
        metrics = result.metrics
        print('Toy evaluation completed')
        print('  validation loss: {:.6f}'.format(float(metrics['validation_loss'])))
        print('  perplexity: {:.6f}'.format(float(metrics['perplexity'])))
        print('  token accuracy: {:.4f}'.format(float(metrics['token_accuracy'])))
        print('  duration ms: {:.3f}'.format(float(metrics['runtime_duration_ms'])))
        print('  sample: {!r}'.format(metrics['generation_sample']))
        print('  manifest: {}'.format(result.manifest_path))
    return 0


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    return sum(values) / len(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def _multiseed_values(arguments: argparse.Namespace) -> list[int]:
    if arguments.seeds:
        seeds = arguments.seeds
    else:
        if arguments.num_seeds <= 0:
            raise ValueError('num-seeds must be positive')
        seeds = [arguments.seed + offset for offset in range(arguments.num_seeds)]
    if any(seed < 0 for seed in seeds):
        raise ValueError('seeds must be non-negative')
    return seeds


def _eval_multiseed_command(arguments: argparse.Namespace) -> int:
    seeds = _multiseed_values(arguments)
    individual_results: list[dict[str, object]] = []
    validation_losses: list[float] = []
    token_accuracies: list[float] = []
    perplexities: list[float] = []
    aggregate_artifacts: dict[str, str] = {}
    model_config: dict[str, object] = {}

    for seed in seeds:
        model, artifacts = _load_model(_checkpoint_argument(arguments), arguments.device, seed)
        aggregate_artifacts = aggregate_artifacts or artifacts
        model_config = model.config.to_dict()
        command_args = _safe_args(arguments)
        command_args['parent_command'] = 'eval-multiseed'
        command_args['seed'] = seed
        result = evaluate_toy(
            model=model,
            output_dir=arguments.output_dir,
            seed_config=SeedConfig(seed=seed),
            decoding_config=_decoding_config(arguments, seed_override=seed),
            device=arguments.device,
            batch_size=arguments.batch_size,
            sequence_length=arguments.sequence_length,
            command_args=command_args,
            artifacts=artifacts,
        )
        _record_manifest(arguments, result.manifest_path)
        metrics = result.metrics
        validation_loss = float(metrics['validation_loss'])
        token_accuracy = float(metrics['token_accuracy'])
        perplexity = float(metrics['perplexity'])
        validation_losses.append(validation_loss)
        token_accuracies.append(token_accuracy)
        perplexities.append(perplexity)
        individual_results.append(
            {
                'seed': seed,
                'run_id': result.run_id,
                'manifest_path': str(result.manifest_path),
                'validation_loss': validation_loss,
                'token_accuracy': token_accuracy,
                'perplexity': perplexity,
            }
        )

    mean_loss, std_loss = _mean_and_std(validation_losses)
    mean_accuracy, std_accuracy = _mean_and_std(token_accuracies)
    mean_perplexity, std_perplexity = _mean_and_std(perplexities)
    metrics: dict[str, object] = {
        'mean_validation_loss': mean_loss,
        'std_validation_loss': std_loss,
        'mean_token_accuracy': mean_accuracy,
        'std_token_accuracy': std_accuracy,
        'mean_perplexity': mean_perplexity,
        'std_perplexity': std_perplexity,
        'individual_seed_results': individual_results,
        'num_seeds': len(seeds),
        'seeds': seeds,
    }
    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = RunManifest(
        run_id=run_id,
        run_name='eval-multiseed',
        created_at=created_at,
        status='completed',
        model_config=model_config,
        training_config={},
        seed_config={'seed': arguments.seed, 'seeds': seeds},
        runtime_config={'device': arguments.device},
        decoding_config=_decoding_config(arguments).to_dict(),
        package_version=package_version(),
        git_commit=git_commit(),
        device_info=device_info(arguments.device),
        command='eval-multiseed',
        command_args=_safe_args(arguments),
        artifacts=aggregate_artifacts,
        metrics=metrics,
        trace_events=[],
        config_hash=config_hash(
            {
                'model': model_config,
                'decoding': _decoding_config(arguments).to_dict(),
                'seeds': seeds,
                'device': arguments.device,
            }
        ),
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = arguments.output_dir / f'eval-multiseed-{run_id}.json'
    write_json(manifest_path, manifest.to_dict())
    _record_manifest(arguments, manifest_path)
    payload = {'run_id': run_id, 'manifest_path': str(manifest_path), 'metrics': metrics}
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print('Multi-seed toy evaluation completed')
        print('  seeds: {}'.format(', '.join(str(seed) for seed in seeds)))
        print('  mean validation loss: {:.6f}'.format(mean_loss))
        print('  std validation loss: {:.6f}'.format(std_loss))
        print('  mean token accuracy: {:.4f}'.format(mean_accuracy))
        print('  mean perplexity: {:.6f}'.format(mean_perplexity))
        print('  manifest: {}'.format(manifest_path))
    return 0


def _eval_tasks_command(arguments: argparse.Namespace) -> int:
    model, artifacts = _load_model(
        _checkpoint_argument(arguments), arguments.device, arguments.seed
    )
    task_kwargs: dict[str, object] = {}
    if arguments.tasks:
        task_kwargs['task_names'] = tuple(arguments.tasks)
    result = evaluate_tasks(
        model=model,
        output_dir=arguments.output_dir,
        seed_config=SeedConfig(seed=arguments.seed),
        decoding_config=_decoding_config(arguments),
        device=arguments.device,
        batch_size=arguments.batch_size,
        sequence_length=arguments.sequence_length,
        command_args=_safe_args(arguments),
        artifacts=artifacts,
        **task_kwargs,
    )
    _record_manifest(arguments, result.manifest_path)
    if arguments.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
    else:
        metrics = result.metrics
        print('Toy task evaluation completed')
        print('  tasks: {}'.format(metrics['task_count']))
        print('  aggregate validation loss: {:.6f}'.format(float(metrics['aggregate_validation_loss'])))
        print('  aggregate token accuracy: {:.4f}'.format(float(metrics['aggregate_token_accuracy'])))
        print('  aggregate perplexity: {:.6f}'.format(float(metrics['aggregate_perplexity'])))
        print('  manifest: {}'.format(result.manifest_path))
    return 0


def _bench_command(arguments: argparse.Namespace) -> int:
    model, artifacts = _load_model(
        _checkpoint_argument(arguments), arguments.device, arguments.seed
    )
    result = benchmark_generation(
        model=model,
        output_dir=arguments.output_dir,
        seed_config=SeedConfig(seed=arguments.seed),
        decoding_config=_decoding_config(arguments),
        device=arguments.device,
        prompt_length=arguments.prompt_length,
        repeats=arguments.repeats,
        command_args=_safe_args(arguments),
        artifacts=artifacts,
    )
    _record_manifest(arguments, result.manifest_path)
    if arguments.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        metrics = result.metrics
        print('Dense generation benchmark completed')
        print('  tokens/sec: {:.3f}'.format(float(metrics['tokens_per_second'])))
        print('  parameters: {}'.format(metrics['parameter_count']))
        print(
            '  prompt/generated: {}/{}'.format(
                metrics['prompt_length'], metrics['generated_tokens']
            )
        )
        print('  parameter bytes: {}'.format(metrics['parameter_memory_bytes']))
        print('  manifest: {}'.format(result.manifest_path))
    return 0


def _list_runs_command(arguments: argparse.Namespace) -> int:
    entries = registry_entries(arguments.registry)
    if arguments.json:
        print(
            json.dumps(
                {'registry_path': str(arguments.registry), 'runs': entries},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not entries:
        print('No runs registered at {}'.format(arguments.registry))
        return 0
    print('RUN_ID                               COMMAND          STATUS     SEED   DEVICE  TIMESTAMP')
    for entry in entries:
        print(
            '{:<36} {:<16} {:<10} {:<6} {:<7} {}'.format(
                entry['run_id'],
                entry.get('command_name') or '',
                entry.get('status') or '',
                '' if entry.get('seed') is None else entry.get('seed'),
                entry.get('device') or '',
                entry.get('timestamp') or '',
            )
        )
    return 0


def _report_command(arguments: argparse.Namespace) -> int:
    result = write_baseline_report(
        run_dir=arguments.run_dir,
        output_dir=arguments.output_dir,
        registry_path=_report_registry_path(arguments),
    )
    payload = result.to_dict()
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print('SARN-Dense baseline report generated')
        print('  markdown: {}'.format(result.markdown_path))
        print('  json: {}'.format(result.json_path))
    return 0


def _reproduce_command_args(arguments: argparse.Namespace, stage: str) -> dict[str, object]:
    values = _safe_args(arguments)
    values['stage'] = stage
    return values


def _reproduce_phase2_command(arguments: argparse.Namespace) -> int:
    output_dir = arguments.output_dir
    registry_path = output_dir / 'runs' / 'registry.json'
    train_dir = output_dir / 'train'
    eval_dir = output_dir / 'eval'
    bench_dir = output_dir / 'bench'
    report_dir = output_dir / 'reports'
    model_config = ModelConfig(
        vocab_size=256,
        max_seq_len=max(64, arguments.sequence_length + arguments.max_new_tokens),
        d_model=arguments.d_model,
        n_layers=arguments.layers,
        n_heads=arguments.heads,
        ffn_hidden_dim=arguments.d_model * 3,
    )
    training_config = TrainingConfig(
        learning_rate=arguments.learning_rate,
        batch_size=arguments.batch_size,
        sequence_length=arguments.sequence_length,
        max_steps=arguments.train_steps,
        device=arguments.device,
    )
    seed_config = SeedConfig(seed=arguments.seed)
    train_result = run_smoke_training(
        model_config=model_config,
        training_config=training_config,
        seed_config=seed_config,
        artifact_config=ArtifactConfig(output_dir=train_dir),
        command_args=_reproduce_command_args(arguments, 'train-smoke'),
    )
    record_manifest(registry_path, train_result.manifest_path)

    decoding_config = DecodingConfig(
        strategy='greedy',
        max_new_tokens=arguments.max_new_tokens,
        use_kv_cache=True,
        seed=arguments.seed,
    )
    model, artifacts = _load_model(train_result.checkpoint_path, arguments.device, arguments.seed)
    eval_result = evaluate_toy(
        model=model,
        output_dir=eval_dir,
        seed_config=seed_config,
        decoding_config=decoding_config,
        device=arguments.device,
        batch_size=arguments.batch_size,
        sequence_length=arguments.sequence_length,
        command_args=_reproduce_command_args(arguments, 'eval-toy'),
        artifacts=artifacts,
    )
    record_manifest(registry_path, eval_result.manifest_path)

    model, artifacts = _load_model(train_result.checkpoint_path, arguments.device, arguments.seed)
    bench_result = benchmark_generation(
        model=model,
        output_dir=bench_dir,
        seed_config=seed_config,
        decoding_config=decoding_config,
        device=arguments.device,
        prompt_length=min(arguments.sequence_length, model_config.max_seq_len),
        repeats=arguments.bench_repeats,
        command_args=_reproduce_command_args(arguments, 'bench'),
        artifacts=artifacts,
    )
    record_manifest(registry_path, bench_result.manifest_path)

    report_result = write_baseline_report(
        run_dir=output_dir,
        output_dir=report_dir,
        registry_path=registry_path,
    )
    summary = {
        'schema_version': 'aegis.phase2_reproduce/v1',
        'status': 'completed',
        'checkpoint_path': str(train_result.checkpoint_path),
        'train_manifest_path': str(train_result.manifest_path),
        'eval_manifest_path': str(eval_result.manifest_path),
        'bench_manifest_path': str(bench_result.manifest_path),
        'report_markdown_path': str(report_result.markdown_path),
        'report_json_path': str(report_result.json_path),
        'registry_path': str(registry_path),
    }
    write_json(output_dir / 'reproduce-summary.json', summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if train_result.final_loss < train_result.initial_loss else 2


def _sweep_baseline_command(arguments: argparse.Namespace) -> int:
    summary = run_baseline_sweep(
        output_dir=arguments.output_dir,
        device=arguments.device,
        seed=arguments.seed,
        train_steps=arguments.train_steps,
        batch_size=arguments.batch_size,
        max_new_tokens=arguments.max_new_tokens,
        bench_repeats=arguments.bench_repeats,
    )
    payload = {
        'run_id': summary['run_id'],
        'summary_json_path': summary['artifacts']['summary_json'],
        'summary_markdown_path': summary['artifacts']['summary_markdown'],
        'registry_path': summary['artifacts']['registry'],
        'metrics': summary['metrics'],
        'results': summary['results'],
    }
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print('SARN-Dense baseline sweep completed')
        print('  configs: {}'.format(summary['metrics']['config_count']))
        print('  summary: {}'.format(summary['artifacts']['summary_json']))
        print('  markdown: {}'.format(summary['artifacts']['summary_markdown']))
        print('  registry: {}'.format(summary['artifacts']['registry']))
    return 0


def _compare_baselines_command(arguments: argparse.Namespace) -> int:
    summary = compare_baselines(arguments.input, arguments.output_dir)
    payload = {
        'run_id': summary['run_id'],
        'comparison_json_path': summary['artifacts']['comparison_json'],
        'comparison_markdown_path': summary['artifacts']['comparison_markdown'],
        'winners': summary['winners'],
    }
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print('Baseline comparison generated')
        print('  markdown: {}'.format(summary['artifacts']['comparison_markdown']))
        print('  json: {}'.format(summary['artifacts']['comparison_json']))
        print('  balanced: {}'.format(summary['winners']['best_balanced_config']['config_name']))
    return 0


def _sweep_attention_command(arguments: argparse.Namespace) -> int:
    summary = run_attention_sweep(
        output_dir=arguments.output_dir,
        device=arguments.device,
        seed=arguments.seed,
        train_steps=arguments.train_steps,
        batch_size=arguments.batch_size,
        sequence_length=arguments.sequence_length,
        max_new_tokens=arguments.max_new_tokens,
        bench_repeats=arguments.bench_repeats,
    )
    payload = {
        'run_id': summary['run_id'],
        'summary_json_path': summary['artifacts']['summary_json'],
        'summary_markdown_path': summary['artifacts']['summary_markdown'],
        'registry_path': summary['artifacts']['registry'],
        'metrics': summary['metrics'],
        'results': summary['results'],
    }
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print('SARN-Dense attention sweep completed')
        print('  variants: {}'.format(summary['metrics']['config_count']))
        print('  summary: {}'.format(summary['artifacts']['summary_json']))
        print('  markdown: {}'.format(summary['artifacts']['summary_markdown']))
        print('  registry: {}'.format(summary['artifacts']['registry']))
    return 0


def _compare_attention_command(arguments: argparse.Namespace) -> int:
    summary = compare_attention(arguments.input, arguments.output_dir)
    payload = {
        'run_id': summary['run_id'],
        'comparison_json_path': summary['artifacts']['comparison_json'],
        'comparison_markdown_path': summary['artifacts']['comparison_markdown'],
        'winners': summary['winners'],
    }
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print('SARN-Dense attention comparison generated')
        print('  markdown: {}'.format(summary['artifacts']['comparison_markdown']))
        print('  json: {}'.format(summary['artifacts']['comparison_json']))
        print(
            '  balanced: {}'.format(
                summary['winners']['best_balanced_attention']['config_name']
            )
        )
    return 0


def _sweep_workspace_command(arguments: argparse.Namespace) -> int:
    summary = run_workspace_sweep(
        output_dir=arguments.output_dir,
        device=arguments.device,
        seed=arguments.seed,
        train_steps=arguments.train_steps,
        batch_size=arguments.batch_size,
        sequence_length=arguments.sequence_length,
        max_new_tokens=arguments.max_new_tokens,
        bench_repeats=arguments.bench_repeats,
    )
    payload = {
        'run_id': summary['run_id'],
        'summary_json_path': summary['artifacts']['summary_json'],
        'summary_markdown_path': summary['artifacts']['summary_markdown'],
        'registry_path': summary['artifacts']['registry'],
        'metrics': summary['metrics'],
        'results': summary['results'],
    }
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print('SARN-Dense workspace sweep completed')
        print('  variants: {}'.format(summary['metrics']['config_count']))
        print('  summary: {}'.format(summary['artifacts']['summary_json']))
        print('  markdown: {}'.format(summary['artifacts']['summary_markdown']))
        print('  registry: {}'.format(summary['artifacts']['registry']))
    return 0


def _compare_workspace_command(arguments: argparse.Namespace) -> int:
    summary = compare_workspace(arguments.input, arguments.output_dir)
    payload = {
        'run_id': summary['run_id'],
        'comparison_json_path': summary['artifacts']['comparison_json'],
        'comparison_markdown_path': summary['artifacts']['comparison_markdown'],
        'winners': summary['winners'],
    }
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print('SARN-Dense workspace comparison generated')
        print('  markdown: {}'.format(summary['artifacts']['comparison_markdown']))
        print('  json: {}'.format(summary['artifacts']['comparison_json']))
        print(
            '  balanced: {}'.format(
                summary['winners']['best_balanced_workspace']['config_name']
            )
        )
    return 0


def _check_gates_command(arguments: argparse.Namespace) -> int:
    result = check_gates(
        summary_path=arguments.summary,
        max_eval_loss=arguments.max_eval_loss,
        min_token_accuracy=arguments.min_token_accuracy,
        min_tokens_per_second=arguments.min_tokens_per_second,
    )
    if arguments.json:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        status = 'PASS' if result['passed'] else 'FAIL'
        print('Experiment gates: {}'.format(status))
        for check in result['checks']:
            mark = 'PASS' if check['passed'] else 'FAIL'
            print('  {} {} - {}'.format(mark, check['name'], check['detail']))
    return 0 if result['passed'] else 1


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == 'run':
        return _run_command(arguments)
    if arguments.command == 'train-smoke':
        return _train_command(arguments)
    if arguments.command == 'eval-toy':
        return _eval_command(arguments)
    if arguments.command == 'eval-multiseed':
        return _eval_multiseed_command(arguments)
    if arguments.command == 'eval-tasks':
        return _eval_tasks_command(arguments)
    if arguments.command == 'bench':
        return _bench_command(arguments)
    if arguments.command == 'list-runs':
        return _list_runs_command(arguments)
    if arguments.command == 'report-baseline':
        return _report_command(arguments)
    if arguments.command == 'reproduce-phase2':
        return _reproduce_phase2_command(arguments)
    if arguments.command == 'sweep-baseline':
        return _sweep_baseline_command(arguments)
    if arguments.command == 'compare-baselines':
        return _compare_baselines_command(arguments)
    if arguments.command == 'sweep-attention':
        return _sweep_attention_command(arguments)
    if arguments.command == 'compare-attention':
        return _compare_attention_command(arguments)
    if arguments.command == 'sweep-workspace':
        return _sweep_workspace_command(arguments)
    if arguments.command == 'compare-workspace':
        return _compare_workspace_command(arguments)
    if arguments.command == 'check-gates':
        return _check_gates_command(arguments)
    raise AssertionError(f'unhandled command: {arguments.command}')
