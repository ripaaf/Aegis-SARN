'''SARN-Dense decoder-only language model.'''

from __future__ import annotations

import torch
from torch import Tensor, nn

from aegis_sarn.config import ModelConfig
from aegis_sarn.sarn.layers import DecoderBlock, RMSNorm


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
        if input_ids.ndim != 2:
            raise ValueError('input_ids must have shape [batch, sequence]')
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError('input sequence exceeds max_seq_len')
        if input_ids.dtype != torch.long:
            raise TypeError('input_ids must use torch.long token IDs')

        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
