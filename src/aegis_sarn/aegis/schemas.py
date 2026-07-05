'''Versioned Phase 1 run schemas.'''

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import uuid4

from aegis_sarn.aegis.trace import TraceEvent


RunStatus = Literal[
    'completed',
    'partial',
    'budget_exhausted',
    'validation_error',
    'model_error',
    'internal_error',
]


@dataclass(frozen=True, slots=True)
class RunRequest:
    prompt: str
    max_new_tokens: int = 16
    max_prompt_tokens: int = 64
    wall_time_ms: int = 30_000
    seed: int = 7
    decoding_strategy: Literal['greedy', 'sample'] = 'greedy'
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    stop_token_id: int | None = None
    use_kv_cache: bool = False
    request_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    schema_version: str = 'aegis.run_request/v1'

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError('prompt cannot be empty')
        if self.max_new_tokens <= 0 or self.max_prompt_tokens <= 0:
            raise ValueError('token budgets must be positive')
        if self.wall_time_ms <= 0:
            raise ValueError('wall_time_ms must be positive')
        if self.seed < 0:
            raise ValueError('seed cannot be negative')
        if self.decoding_strategy not in ('greedy', 'sample'):
            raise ValueError('decoding_strategy must be greedy or sample')
        if self.temperature <= 0:
            raise ValueError('temperature must be positive')
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError('top_k must be positive or None')
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError('top_p must be in (0, 1] or None')
        if self.stop_token_id is not None and self.stop_token_id < 0:
            raise ValueError('stop_token_id cannot be negative')
        if self.schema_version != 'aegis.run_request/v1':
            raise ValueError('unsupported RunRequest schema version')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunUsage:
    prompt_tokens: int
    generated_tokens: int
    wall_time_ms: float
    model_calls: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunResult:
    request_id: str
    run_id: str
    status: RunStatus
    text: str
    backend: str
    usage: RunUsage
    trace: list[TraceEvent]
    limitations: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    schema_version: str = 'aegis.run_result/v1'

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema_version': self.schema_version,
            'request_id': self.request_id,
            'run_id': self.run_id,
            'status': self.status,
            'output': {'parts': [{'kind': 'text', 'content': self.text}]},
            'backend': self.backend,
            'assurance': {'state': 'unchecked', 'checks': []},
            'usage': self.usage.to_dict(),
            'trace': [event.to_dict() for event in self.trace],
            'limitations': self.limitations,
            'manifest_path': self.manifest_path,
        }
