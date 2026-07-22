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
    workspace_enabled: bool = False
    workspace_num_slots: int = 0
    workspace_gated_writeback: bool = True
    workspace_dropout: float = 0.0
    workspace_read_mode: Literal['cross_attention', 'mean_pool'] = 'cross_attention'
    graph_enabled: bool = False
    graph_num_cycles: int = 0
    graph_edge_mode: Literal[
        'none',
        'learned_dense',
        'learned_sparse',
        'frozen_identity',
        'shuffled',
    ] = 'none'
    graph_top_k: int | None = None
    graph_gated_update: bool = True
    graph_residual_scale: float = 1.0
    graph_dropout: float = 0.0
    graph_after_workspace: bool = True
    memory_enabled: bool = False
    memory_num_slots: int = 0
    memory_write_mode: Literal['none', 'gated', 'hebbian_like'] = 'none'
    memory_read_mode: Literal['none', 'attention', 'slot_mix'] = 'none'
    memory_reset_mode: Literal[
        'per_forward', 'per_generation', 'manual'
    ] = 'per_generation'
    memory_decay: float = 0.0
    memory_gated_write: bool = True
    memory_requires_workspace: bool = True
    memory_after_graph: bool = True
    experts_enabled: bool = False
    expert_num_experts: int = 0
    expert_top_k: int = 1
    expert_capacity_factor: float = 1.0
    expert_hidden_dim: int | None = None
    expert_router_noise: float = 0.0
    expert_load_balance_weight: float = 0.0
    expert_use_shared_expert: bool = False
    expert_layer_frequency: int = 1
    expert_replaces_ffn: bool = True

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
        if self.workspace_num_slots < 0:
            raise ConfigError('workspace_num_slots cannot be negative')
        if self.workspace_enabled and self.workspace_num_slots <= 0:
            raise ConfigError(
                'workspace_num_slots must be positive when workspace is enabled'
            )
        if not 0.0 <= self.workspace_dropout < 1.0:
            raise ConfigError('workspace_dropout must be in [0, 1)')
        if self.workspace_read_mode not in ('cross_attention', 'mean_pool'):
            raise ConfigError(
                'workspace_read_mode must be cross_attention or mean_pool'
            )
        valid_graph_modes = (
            'none',
            'learned_dense',
            'learned_sparse',
            'frozen_identity',
            'shuffled',
        )
        if self.graph_edge_mode not in valid_graph_modes:
            raise ConfigError(
                'graph_edge_mode must be none, learned_dense, learned_sparse, '
                'frozen_identity, or shuffled'
            )
        if self.graph_num_cycles < 0:
            raise ConfigError('graph_num_cycles cannot be negative')
        if self.graph_top_k is not None:
            if self.graph_top_k <= 0:
                raise ConfigError('graph_top_k must be positive or None')
            if self.graph_top_k > self.workspace_num_slots:
                raise ConfigError('graph_top_k cannot exceed workspace_num_slots')
        if self.graph_edge_mode == 'learned_sparse' and self.graph_top_k is None:
            raise ConfigError('learned_sparse graph mode requires graph_top_k')
        if not 0.0 <= self.graph_dropout < 1.0:
            raise ConfigError('graph_dropout must be in [0, 1)')
        if self.graph_residual_scale < 0.0:
            raise ConfigError('graph_residual_scale cannot be negative')
        if self.graph_enabled:
            if not self.workspace_enabled:
                raise ConfigError('graph_enabled requires workspace_enabled=True')
            if self.graph_num_cycles <= 0:
                raise ConfigError(
                    'graph_num_cycles must be positive when graph is enabled'
                )
            if not self.graph_after_workspace:
                raise ConfigError(
                    'Phase 6 supports graph message passing only after slot accumulation'
                )
        if self.memory_num_slots < 0:
            raise ConfigError('memory_num_slots cannot be negative')
        if self.memory_write_mode not in ('none', 'gated', 'hebbian_like'):
            raise ConfigError(
                'memory_write_mode must be none, gated, or hebbian_like'
            )
        if self.memory_read_mode not in ('none', 'attention', 'slot_mix'):
            raise ConfigError(
                'memory_read_mode must be none, attention, or slot_mix'
            )
        if self.memory_reset_mode not in (
            'per_forward',
            'per_generation',
            'manual',
        ):
            raise ConfigError(
                'memory_reset_mode must be per_forward, per_generation, or manual'
            )
        if not 0.0 <= self.memory_decay <= 1.0:
            raise ConfigError('memory_decay must be in [0, 1]')
        if self.memory_enabled:
            if not self.workspace_enabled:
                raise ConfigError('memory_enabled requires workspace_enabled=True')
            if self.memory_num_slots <= 0:
                raise ConfigError(
                    'memory_num_slots must be positive when memory is enabled'
                )
            if not self.memory_requires_workspace:
                raise ConfigError(
                    'Phase 7 memory requires the bounded workspace path'
                )
            if not self.memory_after_graph:
                raise ConfigError(
                    'Phase 7 supports memory only after optional graph processing'
                )
        if self.expert_num_experts < 0:
            raise ConfigError('expert_num_experts cannot be negative')
        if self.expert_top_k <= 0:
            raise ConfigError('expert_top_k must be positive')
        if self.expert_capacity_factor <= 0.0:
            raise ConfigError('expert_capacity_factor must be positive')
        if self.expert_hidden_dim is not None and self.expert_hidden_dim <= 0:
            raise ConfigError('expert_hidden_dim must be positive or None')
        if self.expert_router_noise < 0.0:
            raise ConfigError('expert_router_noise cannot be negative')
        if self.expert_load_balance_weight < 0.0:
            raise ConfigError('expert_load_balance_weight cannot be negative')
        if self.expert_layer_frequency <= 0:
            raise ConfigError('expert_layer_frequency must be positive')
        if self.experts_enabled:
            if self.expert_num_experts <= 1:
                raise ConfigError(
                    'expert_num_experts must be greater than one when experts are enabled'
                )
            if self.expert_top_k > self.expert_num_experts:
                raise ConfigError(
                    'expert_top_k cannot exceed expert_num_experts'
                )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def resolved_n_kv_heads(self) -> int:
        return self.n_heads if self.n_kv_heads is None else self.n_kv_heads

    @property
    def kv_group_size(self) -> int:
        return self.n_heads // self.resolved_n_kv_heads

    @property
    def resolved_expert_hidden_dim(self) -> int:
        return (
            self.ffn_hidden_dim
            if self.expert_hidden_dim is None
            else self.expert_hidden_dim
        )

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
            workspace_enabled = bool(
                self.model_config.get('workspace_enabled', False)
            )
            configured_workspace_slots = int(
                self.model_config.get('workspace_num_slots', 0)
            )
            workspace_num_slots = (
                configured_workspace_slots if workspace_enabled else 0
            )
            workspace_gated_writeback = workspace_enabled and bool(
                self.model_config.get('workspace_gated_writeback', True)
            )
            variant_name = self.command_args.get('workspace_variant_name')
            if not variant_name:
                if not workspace_enabled:
                    variant_name = 'dense-control'
                elif not workspace_gated_writeback:
                    variant_name = 'workspace-null'
                else:
                    variant_name = f'workspace-slots{workspace_num_slots}'
            payload['workspace_enabled'] = workspace_enabled
            payload['workspace_num_slots'] = workspace_num_slots
            payload['workspace_gated_writeback'] = workspace_gated_writeback
            payload['workspace_variant_name'] = variant_name
            graph_enabled = bool(self.model_config.get('graph_enabled', False))
            graph_num_cycles = (
                int(self.model_config.get('graph_num_cycles', 0))
                if graph_enabled
                else 0
            )
            graph_edge_mode = (
                str(self.model_config.get('graph_edge_mode', 'none'))
                if graph_enabled
                else 'none'
            )
            graph_top_k = (
                self.model_config.get('graph_top_k') if graph_enabled else None
            )
            graph_gated_update = graph_enabled and bool(
                self.model_config.get('graph_gated_update', True)
            )
            graph_variant_name = self.command_args.get('graph_variant_name')
            if not graph_variant_name:
                if not workspace_enabled:
                    graph_variant_name = 'dense-control'
                elif not graph_enabled:
                    graph_variant_name = 'workspace-control'
                elif graph_edge_mode == 'none':
                    graph_variant_name = 'graph-null'
                elif graph_edge_mode == 'frozen_identity':
                    graph_variant_name = 'graph-identity'
                else:
                    graph_variant_name = (
                        f'graph-{graph_edge_mode}-cycle{graph_num_cycles}'
                    )
            payload['graph_enabled'] = graph_enabled
            payload['graph_num_cycles'] = graph_num_cycles
            payload['graph_edge_mode'] = graph_edge_mode
            payload['graph_top_k'] = graph_top_k
            payload['graph_gated_update'] = graph_gated_update
            payload['graph_variant_name'] = graph_variant_name
            memory_enabled = bool(
                self.model_config.get('memory_enabled', False)
            )
            memory_num_slots = (
                int(self.model_config.get('memory_num_slots', 0))
                if memory_enabled
                else 0
            )
            memory_write_mode = (
                str(self.model_config.get('memory_write_mode', 'none'))
                if memory_enabled
                else 'none'
            )
            memory_read_mode = (
                str(self.model_config.get('memory_read_mode', 'none'))
                if memory_enabled
                else 'none'
            )
            memory_reset_mode = (
                str(
                    self.model_config.get(
                        'memory_reset_mode', 'per_generation'
                    )
                )
                if memory_enabled
                else 'per_generation'
            )
            memory_variant_name = self.command_args.get('memory_variant_name')
            if not memory_variant_name:
                if not workspace_enabled:
                    memory_variant_name = 'dense-control'
                elif not memory_enabled:
                    memory_variant_name = (
                        'graph-control' if graph_enabled else 'workspace-control'
                    )
                elif memory_write_mode == 'none' and memory_read_mode == 'none':
                    memory_variant_name = 'memory-null'
                elif graph_enabled:
                    memory_variant_name = 'graph-memory-gated'
                else:
                    memory_variant_name = 'memory-gated'
            payload['memory_enabled'] = memory_enabled
            payload['memory_num_slots'] = memory_num_slots
            payload['memory_write_mode'] = memory_write_mode
            payload['memory_read_mode'] = memory_read_mode
            payload['memory_reset_mode'] = memory_reset_mode
            payload['memory_decay'] = (
                float(self.model_config.get('memory_decay', 0.0))
                if memory_enabled
                else 0.0
            )
            payload['memory_variant_name'] = memory_variant_name
            experts_enabled = bool(
                self.model_config.get('experts_enabled', False)
            )
            expert_num_experts = (
                int(self.model_config.get('expert_num_experts', 0))
                if experts_enabled
                else 0
            )
            expert_top_k = (
                int(self.model_config.get('expert_top_k', 1))
                if experts_enabled
                else 0
            )
            configured_expert_hidden = self.model_config.get(
                'expert_hidden_dim'
            )
            expert_hidden_dim = (
                int(
                    self.model_config.get('ffn_hidden_dim', 0)
                    if configured_expert_hidden is None
                    else configured_expert_hidden
                )
                if experts_enabled
                else 0
            )
            expert_replaces_ffn = experts_enabled and bool(
                self.model_config.get('expert_replaces_ffn', True)
            )
            expert_variant_name = self.command_args.get(
                'expert_variant_name'
            )
            if not expert_variant_name:
                if not experts_enabled:
                    expert_variant_name = 'dense-control'
                elif not expert_replaces_ffn:
                    expert_variant_name = 'expert-null'
                else:
                    expert_variant_name = (
                        f'experts-{expert_num_experts}-top{expert_top_k}'
                    )
            payload['experts_enabled'] = experts_enabled
            payload['expert_num_experts'] = expert_num_experts
            payload['expert_top_k'] = expert_top_k
            payload['expert_capacity_factor'] = (
                float(self.model_config.get('expert_capacity_factor', 1.0))
                if experts_enabled
                else 0.0
            )
            payload['expert_hidden_dim'] = expert_hidden_dim
            payload['expert_router_noise'] = (
                float(self.model_config.get('expert_router_noise', 0.0))
                if experts_enabled
                else 0.0
            )
            payload['expert_load_balance_weight'] = (
                float(
                    self.model_config.get('expert_load_balance_weight', 0.0)
                )
                if experts_enabled
                else 0.0
            )
            payload['expert_use_shared_expert'] = experts_enabled and bool(
                self.model_config.get('expert_use_shared_expert', False)
            )
            payload['expert_layer_frequency'] = (
                int(self.model_config.get('expert_layer_frequency', 1))
                if experts_enabled
                else 0
            )
            payload['expert_replaces_ffn'] = expert_replaces_ffn
            payload['expert_variant_name'] = expert_variant_name
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
            'Workspace, graph, and working-memory states are transient and not persistent memory.',
            'Working memory is bounded and resettable; it is not user or long-term memory.',
            'Graph message passing does not establish formal logic or human-like reasoning.',
            'Sparse routing is a tiny local prototype, not distributed MoE or proof of specialization.',
        ]
        return payload
