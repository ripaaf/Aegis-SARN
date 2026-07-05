from __future__ import annotations

import pytest

from aegis_sarn.aegis import FakeBackend, RunRequest, SARNBackend, SessionController
from aegis_sarn.config import ModelConfig
from aegis_sarn.sarn.model import SARNDense


def test_fake_backend_produces_structured_ordered_trace() -> None:
    result = SessionController(FakeBackend(response='accepted')).run(
        RunRequest(prompt='hello', max_new_tokens=4)
    )
    payload = result.to_dict()
    assert result.status == 'completed'
    assert result.text == 'acce'
    assert [event.sequence for event in result.trace] == list(range(len(result.trace)))
    assert payload['trace'][0]['event_type'] == 'run.created'
    assert payload['trace'][-1]['event_type'] == 'run.completed'
    assert payload['usage']['model_calls'] == 1


def test_sarn_backend_runs_through_controller() -> None:
    model = SARNDense(
        ModelConfig(
            vocab_size=256,
            max_seq_len=16,
            d_model=16,
            n_layers=1,
            n_heads=2,
            ffn_hidden_dim=32,
        )
    )
    result = SessionController(SARNBackend(model)).run(
        RunRequest(
            prompt='abc', max_prompt_tokens=8, max_new_tokens=3, wall_time_ms=30_000
        )
    )
    assert result.status == 'completed'
    assert result.usage.prompt_tokens == 3
    assert result.usage.generated_tokens == 3
    assert result.backend == 'sarn-dense'

    incompatible = SARNDense(
        ModelConfig(
            vocab_size=32,
            max_seq_len=16,
            d_model=16,
            n_layers=1,
            n_heads=2,
            ffn_hidden_dim=32,
        )
    )
    with pytest.raises(ValueError, match='vocab_size'):
        SARNBackend(incompatible)


def test_wall_time_budget_is_reported() -> None:
    times = iter((0.0, 1.0))
    controller = SessionController(FakeBackend(), timer=lambda: next(times))
    result = controller.run(
        RunRequest(prompt='hello', max_new_tokens=2, wall_time_ms=10)
    )
    assert result.status == 'budget_exhausted'
    assert result.text == ''
    assert any(event.event_type == 'budget.exhausted' for event in result.trace)
