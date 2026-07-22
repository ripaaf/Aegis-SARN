'''Small CPU reference implementation of Phase 8 sparse FFN routing.'''

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from aegis_sarn.config import ModelConfig


@dataclass(frozen=True, slots=True)
class ExpertDiagnostics:
    num_experts: int
    top_k: int
    active_experts: int
    router_entropy: Tensor
    load_balance_score: Tensor
    max_load_fraction: Tensor
    min_load_fraction: Tensor
    dropped_token_fraction: Tensor

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            'experts_enabled': True,
            'expert_num_experts': self.num_experts,
            'expert_top_k': self.top_k,
            'expert_active_experts': self.active_experts,
            'expert_router_entropy': float(
                self.router_entropy.detach().cpu().item()
            ),
            'expert_load_balance_score': float(
                self.load_balance_score.detach().cpu().item()
            ),
            'expert_max_load_fraction': float(
                self.max_load_fraction.detach().cpu().item()
            ),
            'expert_min_load_fraction': float(
                self.min_load_fraction.detach().cpu().item()
            ),
            'expert_dropped_token_fraction': float(
                self.dropped_token_fraction.detach().cpu().item()
            ),
        }


class _ExpertMLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = config.resolved_expert_hidden_dim
        self.ffn_type = config.ffn_type
        if self.ffn_type == 'gated':
            self.gate = nn.Linear(config.d_model, hidden_dim, bias=False)
        else:
            self.gate = None
        self.up = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.gate is None:
            hidden = F.gelu(self.up(inputs))
        else:
            hidden = F.silu(self.gate(inputs)) * self.up(inputs)
        return self.dropout(self.down(hidden))


class SparseExpertFFN(nn.Module):
    '''Route each token to top-k local FFNs without distributed machinery.'''

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not config.experts_enabled or not config.expert_replaces_ffn:
            raise ValueError(
                'SparseExpertFFN requires enabled replacing experts'
            )
        self.d_model = config.d_model
        self.num_experts = config.expert_num_experts
        self.top_k = config.expert_top_k
        self.router_noise = config.expert_router_noise
        self.load_balance_weight = config.expert_load_balance_weight
        self.router = nn.Linear(
            config.d_model, config.expert_num_experts, bias=False
        )
        self.experts = nn.ModuleList(
            _ExpertMLP(config) for _ in range(config.expert_num_experts)
        )
        self.shared_expert = (
            _ExpertMLP(config) if config.expert_use_shared_expert else None
        )
        self.last_diagnostics: ExpertDiagnostics | None = None
        self.last_top_k_indices: Tensor | None = None
        self.last_auxiliary_loss: Tensor | None = None

    def forward(
        self, inputs: Tensor
    ) -> tuple[Tensor, ExpertDiagnostics]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError(
                'expert inputs must have [batch, sequence, model_dimension]'
            )
        flat_inputs = inputs.reshape(-1, self.d_model)
        router_logits = self.router(flat_inputs)
        if self.training and self.router_noise > 0.0:
            router_logits = router_logits + (
                torch.randn_like(router_logits) * self.router_noise
            )
        router_probabilities = F.softmax(router_logits, dim=-1)
        top_probabilities, top_indices = torch.topk(
            router_probabilities, self.top_k, dim=-1
        )
        routing_weights = top_probabilities / top_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(top_probabilities.dtype).eps)

        combined = torch.zeros_like(flat_inputs)
        assignment_counts = torch.zeros(
            self.num_experts,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        for expert_index, expert in enumerate(self.experts):
            selected = top_indices.eq(expert_index)
            token_weights = (routing_weights * selected).sum(dim=-1)
            token_positions = token_weights.nonzero(as_tuple=False).squeeze(-1)
            if token_positions.numel() == 0:
                continue
            expert_output = expert(flat_inputs.index_select(0, token_positions))
            weighted_output = expert_output * token_weights.index_select(
                0, token_positions
            ).unsqueeze(-1)
            combined.index_add_(0, token_positions, weighted_output)
            assignment_counts[expert_index] = selected.sum().to(inputs.dtype)

        if self.shared_expert is not None:
            combined = combined + self.shared_expert(flat_inputs)

        assignment_total = assignment_counts.sum().clamp_min(1.0)
        load_fractions = assignment_counts / assignment_total
        mean_probabilities = router_probabilities.mean(dim=0)
        auxiliary_loss = self.load_balance_weight * self.num_experts * torch.sum(
            mean_probabilities * load_fractions.detach()
        )
        entropy = -(
            router_probabilities
            * router_probabilities.clamp_min(
                torch.finfo(router_probabilities.dtype).eps
            ).log()
        ).sum(dim=-1).mean()
        balance_score = 1.0 / (
            self.num_experts * load_fractions.square().sum().clamp_min(1.0e-12)
        )
        diagnostics = ExpertDiagnostics(
            num_experts=self.num_experts,
            top_k=self.top_k,
            active_experts=int(assignment_counts.gt(0).sum().item()),
            router_entropy=entropy / math.log(self.num_experts),
            load_balance_score=balance_score,
            max_load_fraction=load_fractions.max(),
            min_load_fraction=load_fractions.min(),
            dropped_token_fraction=inputs.new_zeros(()),
        )
        self.last_diagnostics = diagnostics
        self.last_top_k_indices = top_indices.detach()
        self.last_auxiliary_loss = auxiliary_loss
        return combined.reshape_as(inputs), diagnostics

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def active_parameter_count(self) -> int:
        router_parameters = sum(
            parameter.numel() for parameter in self.router.parameters()
        )
        per_expert = sum(
            parameter.numel() for parameter in self.experts[0].parameters()
        )
        shared_parameters = (
            0
            if self.shared_expert is None
            else sum(
                parameter.numel()
                for parameter in self.shared_expert.parameters()
            )
        )
        return router_parameters + (self.top_k * per_expert) + shared_parameters
