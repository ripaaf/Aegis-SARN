'''Readable reference layers for the SARN-Dense baseline.'''

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from aegis_sarn.config import ModelConfig


@dataclass(frozen=True, slots=True)
class KVCache:
    key: Tensor
    value: Tensor

    def __post_init__(self) -> None:
        if self.key.ndim != 4 or self.value.ndim != 4:
            raise ValueError('KV cache tensors must have [batch, heads, sequence, head_dim]')
        if self.key.shape != self.value.shape:
            raise ValueError('KV cache key/value shapes must match')
        if self.key.device != self.value.device or self.key.dtype != self.value.dtype:
            raise ValueError('KV cache key/value device and dtype must match')

    @property
    def sequence_length(self) -> int:
        return self.key.shape[-2]

    def validate_for(self, tensor: Tensor, n_heads: int, head_dim: int) -> None:
        expected_prefix = (tensor.shape[0], n_heads)
        if self.key.shape[:2] != expected_prefix or self.key.shape[-1] != head_dim:
            raise ValueError('KV cache shape is incompatible with the current input')
        if self.key.device != tensor.device or self.key.dtype != tensor.dtype:
            raise ValueError('KV cache device/dtype is incompatible with the current input')


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=inputs.dtype) * self.weight


def rotate_half(inputs: Tensor) -> Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError('RoPE head_dim must be even')
        frequencies = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer('cos_cache', angles.cos(), persistent=False)
        self.register_buffer('sin_cache', angles.sin(), persistent=False)

    def forward(
        self, query: Tensor, key: Tensor, position_offset: int = 0
    ) -> tuple[Tensor, Tensor]:
        sequence_length = query.shape[-2]
        position_end = position_offset + sequence_length
        if position_offset < 0 or position_end > self.cos_cache.shape[0]:
            raise ValueError('sequence length exceeds the configured RoPE cache')
        cos = self.cos_cache[position_offset:position_end].to(
            device=query.device, dtype=query.dtype
        )
        sin = self.sin_cache[position_offset:position_end].to(
            device=query.device, dtype=query.dtype
        )
        cos = cos.view(1, 1, sequence_length, -1)
        sin = sin.view(1, 1, sequence_length, -1)
        return (
            (query * cos) + (rotate_half(query) * sin),
            (key * cos) + (rotate_half(key) * sin),
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(
            config.head_dim, config.max_seq_len, config.rope_base
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def _split_heads(self, inputs: Tensor) -> Tensor:
        batch, sequence, _ = inputs.shape
        return inputs.view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        output, _ = self.forward_with_cache(inputs, past_key_value=None, use_cache=False)
        return output

    def forward_with_cache(
        self,
        inputs: Tensor,
        past_key_value: KVCache | None,
        use_cache: bool,
    ) -> tuple[Tensor, KVCache | None]:
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        past_length = 0
        if past_key_value is not None:
            past_key_value.validate_for(query, self.n_heads, self.head_dim)
            past_length = past_key_value.sequence_length
        query, key = self.rope(query, key, position_offset=past_length)

        if past_key_value is not None:
            key = torch.cat((past_key_value.key, key), dim=-2)
            value = torch.cat((past_key_value.value, value), dim=-2)
        present = KVCache(key=key, value=value) if use_cache else None

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        query_length = inputs.shape[1]
        key_length = key.shape[-2]
        query_positions = torch.arange(
            past_length, past_length + query_length, device=inputs.device
        ).unsqueeze(-1)
        key_positions = torch.arange(key_length, device=inputs.device).unsqueeze(0)
        future_mask = key_positions > query_positions
        scores = scores.masked_fill(future_mask, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1)
        probabilities = F.dropout(
            probabilities, p=self.dropout, training=self.training
        )
        attended = probabilities @ value
        attended = attended.transpose(1, 2).contiguous().view_as(inputs)
        return self.residual_dropout(self.output(attended)), present


class StandardFFN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.down = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.dropout(self.down(F.gelu(self.up(inputs))))


class GatedFFN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.up = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.down = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.silu(self.gate(inputs)) * self.up(inputs)
        return self.dropout(self.down(hidden))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = GatedFFN(config) if config.ffn_type == 'gated' else StandardFFN(config)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.ffn(self.ffn_norm(hidden))

    def forward_with_cache(
        self,
        inputs: Tensor,
        past_key_value: KVCache | None,
        use_cache: bool,
    ) -> tuple[Tensor, KVCache | None]:
        attention_output, present = self.attention.forward_with_cache(
            self.attention_norm(inputs), past_key_value, use_cache
        )
        hidden = inputs + attention_output
        return hidden + self.ffn(self.ffn_norm(hidden)), present
