'''CPU-compatible dense baseline generation benchmark.'''

from __future__ import annotations

import time
from pathlib import Path

import torch

from aegis_sarn.aegis.trace import TraceRecorder
from aegis_sarn.config import DecodingConfig, RunManifest, SeedConfig
from aegis_sarn.eval.harness import HarnessResult
from aegis_sarn.sarn.generation import generate
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.utils import (
    config_hash,
    device_info,
    git_commit,
    normalize_json,
    package_version,
    set_global_seed,
    write_json,
)


def benchmark_generation(
    model: SARNDense,
    output_dir: Path,
    seed_config: SeedConfig,
    decoding_config: DecodingConfig,
    device: str = 'cpu',
    prompt_length: int = 16,
    repeats: int = 3,
    command_args: dict[str, object] | None = None,
    artifacts: dict[str, str] | None = None,
) -> HarnessResult:
    if prompt_length <= 0 or prompt_length > model.config.max_seq_len:
        raise ValueError('prompt_length must be in the model context range')
    if repeats <= 0:
        raise ValueError('repeats must be positive')

    from datetime import datetime, timezone
    from uuid import uuid4

    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    trace = TraceRecorder(run_id)
    trace.emit('bench.started', 'eval.benchmark', {'device': device})
    set_global_seed(seed_config)
    selected_device = torch.device(device)
    model = model.to(selected_device).eval()
    prompt = (
        torch.arange(prompt_length, device=selected_device, dtype=torch.long)
        % model.config.vocab_size
    ).unsqueeze(0)

    try:
        warmup_config = DecodingConfig(
            strategy='greedy', max_new_tokens=1, use_kv_cache=decoding_config.use_kv_cache
        )
        generate(model, prompt, warmup_config)
        trace.emit('bench.warmup_completed', 'eval.benchmark')
        started = time.perf_counter()
        generated_tokens = 0
        for _ in range(repeats):
            output = generate(model, prompt, decoding_config)
            generated_tokens += output.shape[1] - prompt.shape[1]
        duration_seconds = time.perf_counter() - started
        total_parameters = model.count_parameters()
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        cache_bytes = 0
        cache_bytes_per_token = 0
        if decoding_config.use_kv_cache:
            sample_parameter = next(model.parameters())
            cache_bytes_per_token = (
                model.config.n_layers
                * 2
                * model.config.resolved_n_kv_heads
                * model.config.head_dim
                * sample_parameter.element_size()
            )
            cache_bytes = cache_bytes_per_token * model.config.max_seq_len
        metrics: dict[str, object] = {
            'tokens_per_second': generated_tokens / max(duration_seconds, 1e-12),
            'runtime_duration_ms': duration_seconds * 1000.0,
            'prompt_length': prompt_length,
            'generated_tokens': generated_tokens,
            'generated_tokens_per_repeat': generated_tokens // repeats,
            'repeats': repeats,
            'parameter_count': total_parameters,
            'active_parameter_count': total_parameters,
            'parameter_memory_bytes': parameter_bytes,
            'approximate_kv_cache_bytes': cache_bytes,
            'kv_cache_bytes_per_token': cache_bytes_per_token,
            'attention_type': model.config.attention_type,
            'n_heads': model.config.n_heads,
            'n_kv_heads': model.config.resolved_n_kv_heads,
            'kv_group_size': model.config.kv_group_size,
            'device': str(selected_device),
        }
        metrics.update(model.workspace_metrics())
        trace.emit('bench.completed', 'eval.benchmark', metrics)
    except Exception as error:
        trace.emit(
            'bench.error',
            'eval.benchmark',
            {'error_type': type(error).__name__, 'message': str(error)},
        )
        raise

    configuration = {
        'model': model.config.to_dict(),
        'decoding': decoding_config.to_dict(),
        'seed': seed_config.to_dict(),
        'device': device,
        'prompt_length': prompt_length,
        'repeats': repeats,
    }
    manifest = RunManifest(
        run_id=run_id,
        run_name='bench',
        created_at=created_at,
        status='completed',
        model_config=model.config.to_dict(),
        training_config={},
        seed_config=seed_config.to_dict(),
        runtime_config={'device': device},
        decoding_config=decoding_config.to_dict(),
        package_version=package_version(),
        git_commit=git_commit(),
        device_info=device_info(device),
        command='bench',
        command_args=normalize_json(command_args or {}),
        artifacts={} if artifacts is None else artifacts,
        metrics=metrics,
        trace_events=[event.to_dict() for event in trace.events],
        config_hash=config_hash(configuration),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f'bench-{run_id}.json'
    write_json(manifest_path, manifest.to_dict())
    return HarnessResult(metrics=metrics, manifest_path=manifest_path, run_id=run_id)
