'''Validated, dependency-light configuration and manifest schemas.'''

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


class ConfigError(ValueError):
    '''Raised when a configuration violates a baseline invariant.'''


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 256
    max_seq_len: int = 64
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    attention_type: Literal['mha', 'gqa'] = 'mha'
    n_kv_heads: int | None = None
    ffn_hidden_dim: int = 192
    dropout: float = 0.0
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-6
    ffn_type: Literal['gated', 'standard'] = 'gated'
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        positive = {
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'ffn_hidden_dim': self.ffn_hidden_dim,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ConfigError(f'{name} must be positive')
        if self.d_model % self.n_heads != 0:
            raise ConfigError('d_model must be divisible by n_heads')
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ConfigError('attention head dimension must be even for RoPE')
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError('dropout must be in [0, 1)')
        if self.rope_base <= 1.0:
            raise ConfigError('rope_base must be greater than 1')
        if self.attention_type not in ('mha', 'gqa'):
            raise ConfigError('attention_type must be mha or gqa')
        resolved_kv_heads = self.resolved_n_kv_heads
        if resolved_kv_heads <= 0:
            raise ConfigError('n_kv_heads must be positive')
        if resolved_kv_heads > self.n_heads:
            raise ConfigError('n_kv_heads cannot exceed n_heads')
        if self.n_heads % resolved_kv_heads != 0:
            raise ConfigError('n_kv_heads must divide n_heads')
        if self.attention_type == 'mha' and resolved_kv_heads != self.n_heads:
            raise ConfigError('mha requires n_kv_heads to equal n_heads')

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def resolved_n_kv_heads(self) -> int:
        return self.n_heads if self.n_kv_heads is None else self.n_kv_heads

    @property
    def kv_group_size(self) -> int:
        return self.n_heads // self.resolved_n_kv_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        return cls(**values)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    batch_size: int = 8
    sequence_length: int = 32
    max_steps: int = 100
    grad_clip_norm: float | None = 1.0
    device: str = 'cpu'

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ConfigError('learning_rate must be positive')
        if self.weight_decay < 0:
            raise ConfigError('weight_decay cannot be negative')
        if self.batch_size <= 0 or self.sequence_length < 2 or self.max_steps <= 0:
            raise ConfigError(
                'batch_size/max_steps must be positive and sequence_length >= 2'
            )
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ConfigError('grad_clip_norm must be positive or None')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str = 'cpu'
    max_prompt_tokens: int = 64
    max_new_tokens: int = 16
    wall_time_ms: int = 30_000
    max_model_calls: int = 1

    def __post_init__(self) -> None:
        if self.max_prompt_tokens <= 0 or self.max_new_tokens <= 0:
            raise ConfigError('token budgets must be positive')
        if self.wall_time_ms <= 0 or self.max_model_calls <= 0:
            raise ConfigError('wall_time_ms and max_model_calls must be positive')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecodingConfig:
    strategy: Literal['greedy', 'sample'] = 'greedy'
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    stop_token_id: int | None = None
    use_kv_cache: bool = False
    seed: int = 7

    def __post_init__(self) -> None:
        if self.strategy not in ('greedy', 'sample'):
            raise ConfigError('strategy must be greedy or sample')
        if self.max_new_tokens < 0:
            raise ConfigError('max_new_tokens cannot be negative')
        if self.temperature <= 0:
            raise ConfigError('temperature must be positive')
        if self.top_k is not None and self.top_k <= 0:
            raise ConfigError('top_k must be positive or None')
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ConfigError('top_p must be in (0, 1] or None')
        if self.stop_token_id is not None and self.stop_token_id < 0:
            raise ConfigError('stop_token_id cannot be negative')
        if self.seed < 0:
            raise ConfigError('seed cannot be negative')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SeedConfig:
    seed: int = 7
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ConfigError('seed cannot be negative')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    output_dir: Path = Path('artifacts/phase1')
    checkpoint_name: str = 'sarn-dense-smoke.pt'
    manifest_name: str = 'run-manifest.json'

    def __post_init__(self) -> None:
        if not self.checkpoint_name.endswith('.pt'):
            raise ConfigError('checkpoint_name must end with .pt')
        if not self.manifest_name.endswith('.json'):
            raise ConfigError('manifest_name must end with .json')

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_name

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / self.manifest_name

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values['output_dir'] = str(self.output_dir)
        return values


@dataclass(slots=True)
class RunManifest:
    run_id: str
    run_name: str
    created_at: str
    status: str
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    seed_config: dict[str, Any]
    runtime_config: dict[str, Any] = field(default_factory=dict)
    decoding_config: dict[str, Any] = field(default_factory=dict)
    package_version: str = 'unknown'
    git_commit: str = 'unavailable'
    device_info: dict[str, Any] = field(default_factory=dict)
    command: str = ''
    command_args: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    config_hash: str = ''
    schema_version: str = 'aegis.run_manifest/v1'

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.model_config:
            n_heads = int(self.model_config.get('n_heads', 0))
            raw_kv_heads = self.model_config.get('n_kv_heads')
            n_kv_heads = n_heads if raw_kv_heads is None else int(raw_kv_heads)
            payload['attention_type'] = self.model_config.get('attention_type', 'mha')
            payload['n_heads'] = n_heads
            payload['n_kv_heads'] = n_kv_heads
            payload['kv_group_size'] = n_heads // n_kv_heads
        payload['timestamp'] = self.created_at
        payload['git_commit_hash'] = self.git_commit
        payload['config'] = {
            'model': self.model_config,
            'training': self.training_config,
            'runtime': self.runtime_config,
            'decoding': self.decoding_config,
            'command_args': self.command_args,
        }
        payload['seed'] = self.seed_config.get('seed')
        payload['device'] = (
            self.runtime_config.get('device')
            or self.device_info.get('device')
            or self.metrics.get('device')
        )
        payload['limitations'] = [
            'SARN-Dense is the only implemented model path for this artifact.',
            'Toy generated data is for reproducibility checks, not language capability claims.',
            'SARN-Hybrid and advanced modules are not implemented in this run.',
        ]
        return payload
