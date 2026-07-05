from __future__ import annotations

import json
from pathlib import Path

from aegis_sarn.config import ArtifactConfig, ModelConfig, SeedConfig, TrainingConfig
from aegis_sarn.sarn.training import run_smoke_training


def test_one_batch_overfit_decreases_loss_and_resumes(tmp_path: Path) -> None:
    result = run_smoke_training(
        model_config=ModelConfig(
            vocab_size=32,
            max_seq_len=16,
            d_model=16,
            n_layers=1,
            n_heads=2,
            ffn_hidden_dim=32,
        ),
        training_config=TrainingConfig(
            learning_rate=2e-2,
            batch_size=4,
            sequence_length=12,
            max_steps=25,
        ),
        seed_config=SeedConfig(seed=31),
        artifact_config=ArtifactConfig(output_dir=tmp_path),
    )
    assert result.final_loss < result.initial_loss
    assert result.final_loss < 0.5 * result.initial_loss
    assert result.completed_step == 26
    assert result.checkpoint_path.exists()
    assert result.manifest_path.exists()
    assert len(result.generated_ids) == 12

    manifest = json.loads(result.manifest_path.read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed'
    assert manifest['metrics']['completed_step'] == 26
    assert manifest['artifacts']['checkpoint_digest'].startswith('sha256:')
    assert {
        'command_args',
        'config_hash',
        'created_at',
        'device_info',
        'git_commit',
        'metrics',
        'package_version',
        'seed_config',
        'trace_events',
    }.issubset(manifest)
    assert manifest['trace_events'][0]['event_type'] == 'train.started'
    assert manifest['trace_events'][-1]['event_type'] == 'run.completed'
