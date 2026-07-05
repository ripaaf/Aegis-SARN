'''Bounded synchronous session controller for Phase 1.'''

from __future__ import annotations

import time
from typing import Callable
from uuid import uuid4

from aegis_sarn.aegis.backends import GenerationBackend
from aegis_sarn.aegis.schemas import RunRequest, RunResult, RunUsage
from aegis_sarn.aegis.trace import TraceRecorder


class SessionController:
    def __init__(
        self,
        backend: GenerationBackend,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.backend = backend
        self.timer = timer

    def run(self, request: RunRequest) -> RunResult:
        run_id = str(uuid4())
        trace = TraceRecorder(run_id)
        started = self.timer()
        trace.emit('run.created', 'session_controller', {'request_id': request.request_id})
        trace.emit(
            'request.validated',
            'session_controller',
            {
                'max_prompt_tokens': request.max_prompt_tokens,
                'max_new_tokens': request.max_new_tokens,
                'wall_time_ms': request.wall_time_ms,
            },
        )
        trace.emit('backend.started', self.backend.name)
        try:
            output = self.backend.generate(request)
        except Exception as error:
            elapsed_ms = (self.timer() - started) * 1000.0
            trace.emit(
                'backend.failed',
                self.backend.name,
                {'error_type': type(error).__name__, 'message': str(error)},
            )
            trace.emit('run.failed', 'session_controller', {'status': 'model_error'})
            return RunResult(
                request_id=request.request_id,
                run_id=run_id,
                status='model_error',
                text='',
                backend=self.backend.name,
                usage=RunUsage(0, 0, elapsed_ms),
                trace=trace.events,
                limitations=['backend generation failed'],
            )

        elapsed_ms = (self.timer() - started) * 1000.0
        trace.emit(
            'backend.completed',
            self.backend.name,
            {
                'prompt_tokens': output.prompt_tokens,
                'generated_tokens': output.generated_tokens,
                'metadata': output.metadata,
            },
        )
        status = 'completed'
        text = output.text
        limitations: list[str] = []
        if elapsed_ms > request.wall_time_ms:
            status = 'budget_exhausted'
            text = ''
            limitations.append('wall-time budget was exceeded before completion')
            trace.emit(
                'budget.exhausted',
                'session_controller',
                {'budget': 'wall_time_ms', 'observed': elapsed_ms},
            )
        trace.emit('run.completed', 'session_controller', {'status': status})
        return RunResult(
            request_id=request.request_id,
            run_id=run_id,
            status=status,
            text=text,
            backend=self.backend.name,
            usage=RunUsage(
                prompt_tokens=output.prompt_tokens,
                generated_tokens=output.generated_tokens,
                wall_time_ms=elapsed_ms,
            ),
            trace=trace.events,
            limitations=limitations,
        )
