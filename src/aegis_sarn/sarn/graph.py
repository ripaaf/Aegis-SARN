'''Bounded message passing over Phase 5 latent workspace slots.'''

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from aegis_sarn.config import ModelConfig


@dataclass(frozen=True, slots=True)
class GraphDiagnostics:
    num_cycles: int
    edge_mode: str
    gate_mean: Tensor
    message_norm: Tensor
    slot_norm: Tensor
    top_k: int | None
    enabled: bool = True

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        return {
            'graph_enabled': self.enabled,
            'graph_num_cycles': self.num_cycles,
            'graph_edge_mode': self.edge_mode,
            'graph_top_k': self.top_k,
            'graph_gate_mean': float(self.gate_mean.detach().cpu().item()),
            'graph_message_norm': float(
                self.message_norm.detach().cpu().item()
            ),
            'graph_slot_norm': float(self.slot_norm.detach().cpu().item()),
        }


class GraphMessagePassing(nn.Module):
    '''Apply a fixed number of shared-weight graph updates to latent slots.'''

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not config.graph_enabled or not config.workspace_enabled:
            raise ValueError(
                'GraphMessagePassing requires enabled graph and workspace config'
            )
        self.d_model = config.d_model
        self.num_slots = config.workspace_num_slots
        self.num_cycles = config.graph_num_cycles
        self.edge_mode = config.graph_edge_mode
        self.top_k = config.graph_top_k
        self.gated_update = config.graph_gated_update
        self.residual_scale = config.graph_residual_scale

        self.edge_logits = nn.Parameter(torch.zeros(self.num_slots, self.num_slots))
        self.message_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.update_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(config.graph_dropout)

        identity = torch.eye(self.num_slots)
        self.register_buffer('identity_edges', identity, persistent=False)
        self.register_buffer(
            'shuffled_edges',
            torch.roll(identity, shifts=1, dims=1),
            persistent=False,
        )

    def _adjacency(self, slots: Tensor) -> Tensor:
        if self.edge_mode == 'frozen_identity':
            return self.identity_edges.to(device=slots.device, dtype=slots.dtype)
        if self.edge_mode == 'shuffled':
            return self.shuffled_edges.to(device=slots.device, dtype=slots.dtype)

        logits = self.edge_logits.to(device=slots.device, dtype=slots.dtype)
        if self.edge_mode == 'learned_sparse':
            if self.top_k is None:
                raise RuntimeError('learned_sparse graph is missing graph_top_k')
            top_values, top_indices = logits.topk(self.top_k, dim=-1)
            sparse_logits = torch.full_like(logits, -torch.inf)
            logits = sparse_logits.scatter(-1, top_indices, top_values)
        return F.softmax(logits, dim=-1)

    def forward(self, slots: Tensor) -> tuple[Tensor, GraphDiagnostics]:
        if slots.ndim < 3:
            raise ValueError('slots must end with [slots, model_dimension]')
        if slots.shape[-2:] != (self.num_slots, self.d_model):
            raise ValueError(
                'slots must end with '
                f'[{self.num_slots}, {self.d_model}], got {tuple(slots.shape)}'
            )

        original_shape = slots.shape
        flat_slots = slots.reshape(-1, self.num_slots, self.d_model)
        zero = flat_slots.new_zeros(())
        if self.edge_mode == 'none':
            diagnostics = GraphDiagnostics(
                num_cycles=self.num_cycles,
                edge_mode=self.edge_mode,
                gate_mean=zero,
                message_norm=zero,
                slot_norm=flat_slots.float().norm(dim=-1).mean(),
                top_k=self.top_k,
            )
            return slots, diagnostics

        adjacency = self._adjacency(flat_slots)
        gate = torch.sigmoid(self.gate_logit).to(
            device=flat_slots.device, dtype=flat_slots.dtype
        )
        if not self.gated_update:
            gate = torch.ones_like(gate)
        message_update = flat_slots
        updated = flat_slots
        for _ in range(self.num_cycles):
            values = self.message_projection(updated)
            messages = torch.einsum('ks,bsd->bkd', adjacency, values)
            message_update = torch.tanh(self.update_projection(messages))
            updated = updated + (
                self.residual_scale * gate * self.dropout(message_update)
            )

        diagnostics = GraphDiagnostics(
            num_cycles=self.num_cycles,
            edge_mode=self.edge_mode,
            gate_mean=gate,
            message_norm=message_update.float().norm(dim=-1).mean(),
            slot_norm=updated.float().norm(dim=-1).mean(),
            top_k=self.top_k,
        )
        return updated.reshape(original_shape), diagnostics

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def active_parameter_count(self) -> int:
        if self.edge_mode == 'none':
            return 0
        total = self.count_parameters()
        if self.edge_mode in ('frozen_identity', 'shuffled'):
            total -= self.edge_logits.numel()
        return total
