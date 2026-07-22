'''Bounded latent-slot routing for the experimental Phase 5 workspace path.'''

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from aegis_sarn.config import ModelConfig


@dataclass(frozen=True, slots=True)
class WorkspaceDiagnostics:
    gate_mean: Tensor
    workspace_norm: Tensor
    num_slots: int
    enabled: bool = True

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            'workspace_enabled': self.enabled,
            'workspace_num_slots': self.num_slots,
            'workspace_gate_mean': float(self.gate_mean.detach().cpu().item()),
            'workspace_norm': float(self.workspace_norm.detach().cpu().item()),
        }


class LatentWorkspace(nn.Module):
    '''Causally route token states through a fixed number of latent slots.'''

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not config.workspace_enabled or config.workspace_num_slots <= 0:
            raise ValueError('LatentWorkspace requires an enabled workspace config')
        self.d_model = config.d_model
        self.num_slots = config.workspace_num_slots
        self.gated_writeback = config.workspace_gated_writeback
        self.read_mode = config.workspace_read_mode
        self.learned_slots = nn.Parameter(
            torch.empty(self.num_slots, self.d_model)
        )
        self.router = nn.Linear(self.d_model, self.num_slots, bias=False)
        self.token_update = nn.Linear(self.d_model, self.d_model, bias=False)
        self.read_query = nn.Linear(self.d_model, self.d_model, bias=False)
        self.read_key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.writeback = nn.Linear(self.d_model, self.d_model, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(config.workspace_dropout)
        nn.init.normal_(self.learned_slots, mean=0.0, std=0.02)

    def _initial_slots(self, inputs: Tensor, past_slots: Tensor | None) -> Tensor:
        batch_size = inputs.shape[0]
        if past_slots is None:
            return self.learned_slots.unsqueeze(0).expand(batch_size, -1, -1)
        expected = (batch_size, self.num_slots, self.d_model)
        if past_slots.shape != expected:
            raise ValueError(
                f'workspace cache shape must be {expected}, got {tuple(past_slots.shape)}'
            )
        if past_slots.device != inputs.device or past_slots.dtype != inputs.dtype:
            raise ValueError('workspace cache device/dtype must match token states')
        return past_slots

    def forward(
        self,
        token_states: Tensor,
        past_slots: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, WorkspaceDiagnostics]:
        slot_history = self.accumulate_slots(token_states, past_slots)
        updated, diagnostics = self.read_and_writeback(token_states, slot_history)
        return updated, slot_history[:, -1], diagnostics

    def accumulate_slots(
        self,
        token_states: Tensor,
        past_slots: Tensor | None = None,
    ) -> Tensor:
        '''Return causal pre-read slot states for every input position.'''
        if token_states.ndim != 3 or token_states.shape[-1] != self.d_model:
            raise ValueError(
                'token_states must have [batch, sequence, model_dimension]'
            )
        initial_slots = self._initial_slots(token_states, past_slots)

        # Each token is softly routed across slots. Cumulative updates make the
        # workspace causal and let incremental decoding carry only final slots.
        routing_weights = F.softmax(self.router(token_states), dim=-1)
        token_updates = torch.tanh(self.token_update(token_states))
        contributions = (
            routing_weights.unsqueeze(-1) * token_updates.unsqueeze(-2)
        )
        return initial_slots.unsqueeze(1) + contributions.cumsum(dim=1)

    def read_and_writeback(
        self,
        token_states: Tensor,
        slot_history: Tensor,
    ) -> tuple[Tensor, WorkspaceDiagnostics]:
        '''Read causal slot states and optionally write context to tokens.'''
        expected = (
            token_states.shape[0],
            token_states.shape[1],
            self.num_slots,
            self.d_model,
        )
        if slot_history.shape != expected:
            raise ValueError(
                f'slot_history must have shape {expected}, '
                f'got {tuple(slot_history.shape)}'
            )

        if self.read_mode == 'mean_pool':
            workspace_context = slot_history.mean(dim=-2)
        else:
            queries = self.read_query(token_states)
            keys = self.read_key(slot_history)
            read_scores = torch.einsum('bsd,bskd->bsk', queries, keys)
            read_weights = F.softmax(read_scores / math.sqrt(self.d_model), dim=-1)
            workspace_context = torch.einsum(
                'bsk,bskd->bsd', read_weights, slot_history
            )

        workspace_update = self.dropout(self.writeback(workspace_context))
        gate = torch.sigmoid(self.gate_logit).to(
            device=token_states.device, dtype=token_states.dtype
        )
        updated = token_states + (gate * workspace_update)
        if not self.gated_writeback:
            updated = token_states
            reported_gate = torch.zeros_like(gate)
        else:
            reported_gate = gate

        diagnostics = WorkspaceDiagnostics(
            gate_mean=reported_gate,
            workspace_norm=slot_history[:, -1].float().norm(dim=-1).mean(),
            num_slots=self.num_slots,
        )
        return updated, diagnostics

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
