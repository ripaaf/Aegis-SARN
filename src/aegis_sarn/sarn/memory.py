'''Bounded cache-carried working memory for Phase 7 experiments.'''

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from aegis_sarn.config import ModelConfig


@dataclass(frozen=True, slots=True)
class MemoryDiagnostics:
    num_slots: int
    write_mode: str
    read_mode: str
    reset_mode: str
    gate_mean: Tensor
    memory_norm: Tensor
    write_norm: Tensor
    reset_applied: bool
    enabled: bool = True

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {
            'memory_enabled': self.enabled,
            'memory_num_slots': self.num_slots,
            'memory_write_mode': self.write_mode,
            'memory_read_mode': self.read_mode,
            'memory_reset_mode': self.reset_mode,
            'memory_gate_mean': float(self.gate_mean.detach().cpu().item()),
            'memory_norm': float(self.memory_norm.detach().cpu().item()),
            'memory_write_norm': float(
                self.write_norm.detach().cpu().item()
            ),
            'memory_reset_applied': self.reset_applied,
        }


class ResettableWorkingMemory(nn.Module):
    '''Read and update temporary slots without storing module-global state.'''

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not config.memory_enabled or not config.workspace_enabled:
            raise ValueError(
                'ResettableWorkingMemory requires enabled memory and workspace'
            )
        self.d_model = config.d_model
        self.num_slots = config.memory_num_slots
        self.write_mode = config.memory_write_mode
        self.read_mode = config.memory_read_mode
        self.reset_mode = config.memory_reset_mode
        self.decay = config.memory_decay
        self.gated_write = config.memory_gated_write

        self.write_router = nn.Linear(
            self.d_model, self.num_slots, bias=False
        )
        self.write_projection = nn.Linear(
            self.d_model, self.d_model, bias=False
        )
        self.write_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.read_query = nn.Linear(self.d_model, self.d_model, bias=False)
        self.read_key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.read_output = nn.Linear(self.d_model, self.d_model, bias=False)
        self.read_gate_logit = nn.Parameter(torch.tensor(-2.0))

    def _initial_memory(
        self, token_states: Tensor, past_memory: Tensor | None
    ) -> tuple[Tensor, bool]:
        batch_size = token_states.shape[0]
        expected = (batch_size, self.num_slots, self.d_model)
        reset_applied = past_memory is None or self.reset_mode == 'per_forward'
        if reset_applied:
            return token_states.new_zeros(expected), True
        if past_memory.shape != expected:
            raise ValueError(
                f'memory cache shape must be {expected}, '
                f'got {tuple(past_memory.shape)}'
            )
        if (
            past_memory.device != token_states.device
            or past_memory.dtype != token_states.dtype
        ):
            raise ValueError(
                'memory cache device/dtype must match token states'
            )
        return past_memory, False

    def _workspace_history(
        self, token_states: Tensor, workspace_slots: Tensor
    ) -> Tensor:
        if workspace_slots.ndim == 3:
            if (
                workspace_slots.shape[0] != token_states.shape[0]
                or workspace_slots.shape[-1] != self.d_model
            ):
                raise ValueError(
                    'workspace slots must match token batch/model dimension'
                )
            return workspace_slots.unsqueeze(1).expand(
                -1, token_states.shape[1], -1, -1
            )
        if workspace_slots.ndim != 4:
            raise ValueError(
                'workspace slots must have [batch, slots, model_dimension] '
                'or [batch, sequence, slots, model_dimension]'
            )
        if (
            workspace_slots.shape[:2] != token_states.shape[:2]
            or workspace_slots.shape[-1] != self.d_model
        ):
            raise ValueError(
                'workspace history must match token batch/sequence/model dimension'
            )
        return workspace_slots

    def forward(
        self,
        token_states: Tensor,
        workspace_slots: Tensor,
        past_memory: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, MemoryDiagnostics]:
        if token_states.ndim != 3 or token_states.shape[-1] != self.d_model:
            raise ValueError(
                'token_states must have [batch, sequence, model_dimension]'
            )
        workspace_history = self._workspace_history(
            token_states, workspace_slots
        )
        memory, reset_applied = self._initial_memory(
            token_states, past_memory
        )
        write_gate = torch.sigmoid(self.write_gate_logit).to(
            device=token_states.device, dtype=token_states.dtype
        )
        if not self.gated_write:
            write_gate = torch.ones_like(write_gate)
        read_gate = torch.sigmoid(self.read_gate_logit).to(
            device=token_states.device, dtype=token_states.dtype
        )

        updated_tokens: list[Tensor] = []
        write_norms: list[Tensor] = []
        for position in range(token_states.shape[1]):
            token = token_states[:, position]
            workspace_summary = workspace_history[:, position].mean(dim=-2)
            source = token + workspace_summary

            if self.write_mode == 'none':
                write = torch.zeros_like(memory)
            else:
                routing = F.softmax(self.write_router(source), dim=-1)
                candidate = torch.tanh(self.write_projection(source))
                if self.write_mode == 'hebbian_like':
                    candidate = torch.tanh(
                        candidate * torch.tanh(source)
                    )
                write = routing.unsqueeze(-1) * candidate.unsqueeze(-2)
                memory = ((1.0 - self.decay) * memory) + (
                    write_gate * write
                )
            write_norms.append(write.float().norm(dim=-1).mean())

            if self.read_mode == 'none':
                context = torch.zeros_like(token)
            elif self.read_mode == 'slot_mix':
                context = memory.mean(dim=-2)
            else:
                query = self.read_query(token)
                keys = self.read_key(memory)
                scores = torch.einsum('bd,bmd->bm', query, keys)
                weights = F.softmax(
                    scores / math.sqrt(self.d_model), dim=-1
                )
                context = torch.einsum('bm,bmd->bd', weights, memory)
            token_update = read_gate * self.read_output(context)
            updated_tokens.append(token + token_update)

        output = torch.stack(updated_tokens, dim=1)
        zero = token_states.new_zeros(())
        reported_write_gate = (
            write_gate if self.write_mode != 'none' else zero
        )
        diagnostics = MemoryDiagnostics(
            num_slots=self.num_slots,
            write_mode=self.write_mode,
            read_mode=self.read_mode,
            reset_mode=self.reset_mode,
            gate_mean=reported_write_gate,
            memory_norm=memory.float().norm(dim=-1).mean(),
            write_norm=torch.stack(write_norms).mean(),
            reset_applied=reset_applied,
        )
        return output, workspace_slots, memory, diagnostics

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def active_parameter_count(self) -> int:
        total = 0
        if self.write_mode != 'none':
            total += sum(
                parameter.numel()
                for parameter in (
                    self.write_router.weight,
                    self.write_projection.weight,
                    self.write_gate_logit,
                )
            )
        if self.read_mode != 'none':
            total += sum(
                parameter.numel()
                for parameter in (
                    self.read_query.weight,
                    self.read_key.weight,
                    self.read_output.weight,
                    self.read_gate_logit,
                )
            )
        return total
