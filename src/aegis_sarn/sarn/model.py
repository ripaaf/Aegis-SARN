'''SARN-Dense decoder-only language model.'''

from __future__ import annotations

import torch
from torch import Tensor, nn

from aegis_sarn.config import ModelConfig
from aegis_sarn.sarn.layers import DecoderBlock, KVCache, RMSNorm
from aegis_sarn.sarn.workspace import LatentWorkspace


class SARNDense(nn.Module):
    '''The Phase 1 causal Transformer control model.'''

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.n_layers)
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
            hidden, present_workspace, diagnostics = self.workspace(
                hidden, past_workspace
            )
            self.last_workspace_diagnostics = diagnostics.to_dict()
            if use_cache:
                first_cache = present_key_values[0]
                present_key_values[0] = KVCache(
                    key=first_cache.key,
                    value=first_cache.value,
                    workspace_slots=present_workspace,
                )
        else:
            self.last_workspace_diagnostics = {}
        logits = self.lm_head(self.final_norm(hidden))
        return logits, present_key_values if use_cache else None

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

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
