'''SARN-Dense decoder-only language model.'''

from __future__ import annotations

import torch
from torch import Tensor, nn

from aegis_sarn.config import ModelConfig
from aegis_sarn.sarn.graph import GraphMessagePassing
from aegis_sarn.sarn.layers import DecoderBlock, KVCache, RMSNorm
from aegis_sarn.sarn.memory import ResettableWorkingMemory
from aegis_sarn.sarn.workspace import LatentWorkspace


class SARNDense(nn.Module):
    '''The Phase 1 causal Transformer control model.'''

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(config, layer_index)
            for layer_index in range(config.n_layers)
        )
        if config.workspace_enabled:
            # Workspace construction must not perturb initialization of the
            # dense control parameters used by matched/null experiments.
            rng_state = torch.random.get_rng_state()
            try:
                self.workspace = LatentWorkspace(config)
            finally:
                torch.random.set_rng_state(rng_state)
        else:
            self.workspace = None
        self.last_workspace_diagnostics: dict[str, float | int | bool] = {}
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.graph_enabled:
            # Graph construction must not perturb matched control initialization.
            rng_state = torch.random.get_rng_state()
            try:
                self.graph = GraphMessagePassing(config)
            finally:
                torch.random.set_rng_state(rng_state)
        else:
            self.graph = None
        self.last_graph_diagnostics: dict[
            str, float | int | str | bool | None
        ] = {}
        if config.memory_enabled:
            # Memory construction follows all earlier experimental modules so
            # matched dense/workspace/graph parameters initialize identically.
            rng_state = torch.random.get_rng_state()
            try:
                self.memory = ResettableWorkingMemory(config)
            finally:
                torch.random.set_rng_state(rng_state)
        else:
            self.memory = None
        self.last_memory_diagnostics: dict[
            str, float | int | str | bool
        ] = {}
        self.apply(self._initialize_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        logits, _ = self.forward_with_cache(input_ids, past_key_values=None, use_cache=False)
        return logits

    def forward_with_cache(
        self,
        input_ids: Tensor,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = True,
    ) -> tuple[Tensor, list[KVCache] | None]:
        if input_ids.ndim != 2:
            raise ValueError('input_ids must have shape [batch, sequence]')
        if input_ids.dtype != torch.long:
            raise TypeError('input_ids must use torch.long token IDs')
        if input_ids.shape[1] == 0:
            raise ValueError('input_ids cannot be empty')
        if past_key_values is not None and not use_cache:
            raise ValueError('past_key_values require use_cache=True')
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError('past_key_values must contain one cache per decoder block')

        past_length = 0
        if past_key_values:
            lengths = {cache.sequence_length for cache in past_key_values}
            if len(lengths) != 1:
                raise ValueError('all layer caches must have the same sequence length')
            past_length = lengths.pop()
            if (
                self.workspace is not None
                and past_key_values[0].workspace_slots is None
            ):
                raise ValueError(
                    'workspace-enabled generation requires workspace cache state'
                )
            if (
                self.memory is not None
                and self.config.memory_reset_mode == 'per_generation'
                and past_key_values[0].memory_slots is None
            ):
                raise ValueError(
                    'memory-enabled generation requires memory cache state'
                )
        if past_length + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError('input plus KV cache exceeds max_seq_len')

        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        cache_inputs: list[KVCache | None]
        if past_key_values is None:
            cache_inputs = [None] * len(self.blocks)
        else:
            cache_inputs = list(past_key_values)
        present_key_values: list[KVCache] = []
        for block, past in zip(self.blocks, cache_inputs, strict=True):
            hidden, present = block.forward_with_cache(hidden, past, use_cache)
            if present is not None:
                present_key_values.append(present)
        if self.workspace is not None:
            past_workspace = (
                None
                if not past_key_values
                else past_key_values[0].workspace_slots
            )
            raw_slot_history = self.workspace.accumulate_slots(
                hidden, past_workspace
            )
            read_slot_history = raw_slot_history
            if self.graph is not None:
                read_slot_history, graph_diagnostics = self.graph(raw_slot_history)
                self.last_graph_diagnostics = graph_diagnostics.to_dict()
            else:
                self.last_graph_diagnostics = {}
            present_memory = None
            if self.memory is not None:
                past_memory = (
                    None
                    if not past_key_values
                    else past_key_values[0].memory_slots
                )
                (
                    hidden,
                    read_slot_history,
                    present_memory,
                    memory_diagnostics,
                ) = self.memory(hidden, read_slot_history, past_memory)
                self.last_memory_diagnostics = memory_diagnostics.to_dict()
            else:
                self.last_memory_diagnostics = {}
            hidden, workspace_diagnostics = self.workspace.read_and_writeback(
                hidden, read_slot_history
            )
            self.last_workspace_diagnostics = workspace_diagnostics.to_dict()
            # Cache the pre-graph accumulator. Reapplying graph transforms to
            # cached graph outputs would break full/incremental causal parity.
            present_workspace = raw_slot_history[:, -1]
            if use_cache:
                first_cache = present_key_values[0]
                present_key_values[0] = KVCache(
                    key=first_cache.key,
                    value=first_cache.value,
                    workspace_slots=present_workspace,
                    memory_slots=present_memory,
                )
        else:
            self.last_workspace_diagnostics = {}
            self.last_graph_diagnostics = {}
            self.last_memory_diagnostics = {}
        logits = self.lm_head(self.final_norm(hidden))
        return logits, present_key_values if use_cache else None

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def active_parameter_count(self) -> int:
        total = self.count_parameters()
        if self.graph is not None:
            total -= self.graph.count_parameters()
            total += self.graph.active_parameter_count()
        if self.memory is not None:
            total -= self.memory.count_parameters()
            total += self.memory.active_parameter_count()
        for block in self.blocks:
            if block.expert_ffn is not None:
                total -= block.expert_ffn.count_parameters()
                total += block.expert_ffn.active_parameter_count()
        return total

    def expert_auxiliary_loss(self) -> Tensor:
        losses = [
            block.expert_ffn.last_auxiliary_loss
            for block in self.blocks
            if block.expert_ffn is not None
            and block.expert_ffn.last_auxiliary_loss is not None
        ]
        if losses:
            return torch.stack(losses).sum()
        return self.token_embedding.weight.new_zeros(())

    def workspace_metrics(self) -> dict[str, object]:
        base: dict[str, object] = {
            'workspace_enabled': self.config.workspace_enabled,
            'workspace_num_slots': (
                self.config.workspace_num_slots
                if self.config.workspace_enabled
                else 0
            ),
            'workspace_gated_writeback': (
                self.config.workspace_gated_writeback
                if self.config.workspace_enabled
                else False
            ),
            'workspace_parameter_count': (
                0 if self.workspace is None else self.workspace.count_parameters()
            ),
            'workspace_gate_mean': 0.0,
            'workspace_norm': 0.0,
        }
        base.update(self.last_workspace_diagnostics)
        return base

    def graph_metrics(self) -> dict[str, object]:
        base: dict[str, object] = {
            'graph_enabled': self.config.graph_enabled,
            'graph_num_cycles': (
                self.config.graph_num_cycles if self.config.graph_enabled else 0
            ),
            'graph_edge_mode': (
                self.config.graph_edge_mode if self.config.graph_enabled else 'none'
            ),
            'graph_top_k': (
                self.config.graph_top_k if self.config.graph_enabled else None
            ),
            'graph_gated_update': (
                self.config.graph_gated_update if self.config.graph_enabled else False
            ),
            'graph_parameter_count': (
                0 if self.graph is None else self.graph.count_parameters()
            ),
            'graph_gate_mean': 0.0,
            'graph_message_norm': 0.0,
            'graph_slot_norm': 0.0,
        }
        base.update(self.last_graph_diagnostics)
        return base

    def memory_metrics(self) -> dict[str, object]:
        base: dict[str, object] = {
            'memory_enabled': self.config.memory_enabled,
            'memory_num_slots': (
                self.config.memory_num_slots if self.config.memory_enabled else 0
            ),
            'memory_write_mode': (
                self.config.memory_write_mode
                if self.config.memory_enabled
                else 'none'
            ),
            'memory_read_mode': (
                self.config.memory_read_mode
                if self.config.memory_enabled
                else 'none'
            ),
            'memory_reset_mode': (
                self.config.memory_reset_mode
                if self.config.memory_enabled
                else 'per_generation'
            ),
            'memory_decay': (
                self.config.memory_decay if self.config.memory_enabled else 0.0
            ),
            'memory_parameter_count': (
                0 if self.memory is None else self.memory.count_parameters()
            ),
            'memory_gate_mean': 0.0,
            'memory_norm': 0.0,
            'memory_write_norm': 0.0,
            'memory_reset_applied': False,
        }
        base.update(self.last_memory_diagnostics)
        return base

    def expert_metrics(self) -> dict[str, object]:
        expert_modules = [
            block.expert_ffn
            for block in self.blocks
            if block.expert_ffn is not None
        ]
        diagnostics = [
            module.last_diagnostics
            for module in expert_modules
            if module.last_diagnostics is not None
        ]
        expert_parameter_count = sum(
            module.count_parameters() for module in expert_modules
        )
        expert_active_parameter_count = sum(
            module.active_parameter_count() for module in expert_modules
        )
        base: dict[str, object] = {
            'experts_enabled': self.config.experts_enabled,
            'expert_num_experts': (
                self.config.expert_num_experts
                if self.config.experts_enabled
                else 0
            ),
            'expert_top_k': (
                self.config.expert_top_k if self.config.experts_enabled else 0
            ),
            'expert_capacity_factor': (
                self.config.expert_capacity_factor
                if self.config.experts_enabled
                else 0.0
            ),
            'expert_hidden_dim': (
                self.config.resolved_expert_hidden_dim
                if self.config.experts_enabled
                else 0
            ),
            'expert_active_experts': 0,
            'expert_router_entropy': 0.0,
            'expert_load_balance_score': 0.0,
            'expert_max_load_fraction': 0.0,
            'expert_min_load_fraction': 0.0,
            'expert_dropped_token_fraction': 0.0,
            'expert_parameter_count': expert_parameter_count,
            'expert_active_parameter_count': expert_active_parameter_count,
            'expert_layer_count': len(expert_modules),
        }
        if diagnostics:
            count = float(len(diagnostics))
            base.update(
                {
                    'expert_active_experts': max(
                        item.active_experts for item in diagnostics
                    ),
                    'expert_router_entropy': sum(
                        float(item.router_entropy.detach().cpu().item())
                        for item in diagnostics
                    )
                    / count,
                    'expert_load_balance_score': sum(
                        float(
                            item.load_balance_score.detach().cpu().item()
                        )
                        for item in diagnostics
                    )
                    / count,
                    'expert_max_load_fraction': max(
                        float(item.max_load_fraction.detach().cpu().item())
                        for item in diagnostics
                    ),
                    'expert_min_load_fraction': min(
                        float(item.min_load_fraction.detach().cpu().item())
                        for item in diagnostics
                    ),
                    'expert_dropped_token_fraction': sum(
                        float(
                            item.dropped_token_fraction.detach().cpu().item()
                        )
                        for item in diagnostics
                    )
                    / count,
                }
            )
        return base
