'''Command-line entry points for the Phase 1 smoke and runtime paths.'''

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from aegis_sarn.aegis import FakeBackend, RunRequest, SARNBackend, SessionController
from aegis_sarn.config import (
    ArtifactConfig,
    DecodingConfig,
    ModelConfig,
    RunManifest,
    SeedConfig,
    TrainingConfig,
)
from aegis_sarn.eval import benchmark_generation, evaluate_toy
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


def _decoding_config(arguments: argparse.Namespace) -> DecodingConfig:
    return DecodingConfig(
        strategy=arguments.strategy,
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
        top_p=arguments.top_p,
        stop_token_id=arguments.stop_token_id,
        use_kv_cache=arguments.use_kv_cache,
        seed=arguments.seed,
    )


def _safe_args(arguments: argparse.Namespace) -> dict[str, object]:
    values = normalize_json(vars(arguments))
    if isinstance(values, dict) and 'prompt' in values:
        prompt = str(values['prompt'])
        values['prompt'] = '<redacted>'
        values['prompt_length_chars'] = len(prompt)
    return values


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
    train_parser.add_argument('--learning-rate', type=float, default=5e-3)
    train_parser.add_argument('--seed', type=int, default=7)
    train_parser.add_argument('--device', default='cpu')

    eval_parser = subparsers.add_parser('eval-toy', help='evaluate toy metrics')
    eval_parser.add_argument('--checkpoint', type=Path)
    eval_parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    eval_parser.add_argument('--batch-size', type=int, default=8)
    eval_parser.add_argument('--sequence-length', type=int, default=24)
    eval_parser.add_argument('--device', default='cpu')
    eval_parser.add_argument('--json', action='store_true')
    _add_decoding_arguments(eval_parser)

    bench_parser = subparsers.add_parser('bench', help='benchmark dense generation')
    bench_parser.add_argument('--checkpoint', type=Path)
    bench_parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    bench_parser.add_argument('--prompt-length', type=int, default=16)
    bench_parser.add_argument('--repeats', type=int, default=3)
    bench_parser.add_argument('--device', default='cpu')
    bench_parser.add_argument('--json', action='store_true')
    _add_decoding_arguments(bench_parser)
    return parser


def _run_command(arguments: argparse.Namespace) -> int:
    model: SARNDense | None = None
    artifacts: dict[str, str] = {}
    if arguments.backend == 'fake':
        backend = FakeBackend()
    else:
        model, artifacts = _load_model(
            arguments.checkpoint, arguments.device, arguments.seed
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
    write_json(manifest_path, manifest.to_dict())
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
        ffn_hidden_dim=arguments.d_model * 3,
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
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.final_loss < result.initial_loss else 2


def _eval_command(arguments: argparse.Namespace) -> int:
    model, artifacts = _load_model(
        arguments.checkpoint, arguments.device, arguments.seed
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


def _bench_command(arguments: argparse.Namespace) -> int:
    model, artifacts = _load_model(
        arguments.checkpoint, arguments.device, arguments.seed
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


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == 'run':
        return _run_command(arguments)
    if arguments.command == 'train-smoke':
        return _train_command(arguments)
    if arguments.command == 'eval-toy':
        return _eval_command(arguments)
    if arguments.command == 'bench':
        return _bench_command(arguments)
    raise AssertionError(f'unhandled command: {arguments.command}')
