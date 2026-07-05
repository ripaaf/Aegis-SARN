'''SARN-Dense decoder-only language model.'''

from __future__ import annotations

import torch
from torch import Tensor, nn

from aegis_sarn.config import ModelConfig
from aegis_sarn.sarn.layers import DecoderBlock, KVCache, RMSNorm


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
        logits = self.lm_head(self.final_norm(hidden))
        return logits, present_key_values if use_cache else None

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
