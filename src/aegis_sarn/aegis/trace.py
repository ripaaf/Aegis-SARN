'''Append-only structured trace events.'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class TraceEvent:
    run_id: str
    sequence: int
    event_type: str
    component: str
    timestamp: str
    payload: dict[str, Any]
    schema_version: str = 'aegis.trace_event/v1'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceRecorder:
    def __init__(self, run_id: str, clock: Callable[[], str] = utc_now) -> None:
        self.run_id = run_id
        self.clock = clock
        self.events: list[TraceEvent] = []

    def emit(
        self, event_type: str, component: str, payload: dict[str, Any] | None = None
    ) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
            sequence=len(self.events),
            event_type=event_type,
            component=component,
            timestamp=self.clock(),
            payload={} if payload is None else payload,
        )
        self.events.append(event)
        return event

