'''Command-line entry points for the Phase 1 smoke and runtime paths.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from aegis_sarn.aegis import FakeBackend, RunRequest, SARNBackend, SessionController
from aegis_sarn.config import ArtifactConfig, ModelConfig, SeedConfig, TrainingConfig
from aegis_sarn.sarn.checkpoint import load_checkpoint
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.sarn.training import run_smoke_training
from aegis_sarn.utils import set_global_seed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='aegis-sarn')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run', help='run a prompt through Aegis')
    run_parser.add_argument('--prompt', required=True)
    run_parser.add_argument('--checkpoint', type=Path)
    run_parser.add_argument('--backend', choices=('sarn', 'fake'), default='sarn')
    run_parser.add_argument('--max-new-tokens', type=int, default=8)
    run_parser.add_argument('--max-prompt-tokens', type=int, default=48)
    run_parser.add_argument('--wall-time-ms', type=int, default=30_000)
    run_parser.add_argument('--seed', type=int, default=7)
    run_parser.add_argument('--device', default='cpu')

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
    return parser


def _run_command(arguments: argparse.Namespace) -> int:
    if arguments.backend == 'fake':
        backend = FakeBackend()
    else:
        if arguments.checkpoint is not None:
            model = load_checkpoint(arguments.checkpoint, map_location=arguments.device).model
        else:
            set_global_seed(SeedConfig(seed=arguments.seed))
            model = SARNDense(ModelConfig())
        backend = SARNBackend(model=model, device=arguments.device)

    request = RunRequest(
        prompt=arguments.prompt,
        max_new_tokens=arguments.max_new_tokens,
        max_prompt_tokens=arguments.max_prompt_tokens,
        wall_time_ms=arguments.wall_time_ms,
        seed=arguments.seed,
    )
    result = SessionController(backend).run(request)
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
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.final_loss < result.initial_loss else 2


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == 'run':
        return _run_command(arguments)
    if arguments.command == 'train-smoke':
        return _train_command(arguments)
    raise AssertionError(f'unhandled command: {arguments.command}')

