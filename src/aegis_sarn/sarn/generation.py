'''Deterministic greedy and stochastic Phase 1 decoding.'''

from __future__ import annotations

import torch
from torch import Tensor

from aegis_sarn.config import DecodingConfig
from aegis_sarn.sarn.layers import KVCache
from aegis_sarn.sarn.model import SARNDense


def _filter_logits(logits: Tensor, top_k: int | None, top_p: float | None) -> Tensor:
    filtered = logits
    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        threshold = torch.topk(filtered, k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float('-inf'))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
        filtered = torch.full_like(filtered, float('-inf'))
        filtered.scatter_(-1, sorted_indices, sorted_logits)
    return filtered


def _select_next_token(
    logits: Tensor,
    config: DecodingConfig,
    generator: torch.Generator,
) -> Tensor:
    if config.strategy == 'greedy':
        return logits.argmax(dim=-1, keepdim=True)
    filtered = _filter_logits(logits / config.temperature, config.top_k, config.top_p)
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator)


@torch.inference_mode()
def generate(model: SARNDense, input_ids: Tensor, config: DecodingConfig) -> Tensor:
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError('input_ids must be a non-empty [batch, sequence] tensor')
    if input_ids.dtype != torch.long:
        raise TypeError('input_ids must use torch.long token IDs')
    if config.stop_token_id is not None and config.stop_token_id >= model.config.vocab_size:
        raise ValueError('stop_token_id is outside the model vocabulary')

    was_training = model.training
    model.eval()
    generated = input_ids
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    generator = torch.Generator(device=input_ids.device).manual_seed(config.seed)
    past_key_values: list[KVCache] | None = None
    cached_logits: Tensor | None = None
    try:
        if config.use_kv_cache and config.max_new_tokens > 0:
            context = generated[:, -model.config.max_seq_len :]
            cached_logits, past_key_values = model.forward_with_cache(
                context, past_key_values=None, use_cache=True
            )

        for step in range(config.max_new_tokens):
            if config.use_kv_cache:
                if cached_logits is None:
                    raise RuntimeError('KV-cache generation was not initialized')
                next_logits = cached_logits[:, -1, :]
            else:
                context = generated[:, -model.config.max_seq_len :]
                next_logits = model(context)[:, -1, :]

            next_token = _select_next_token(next_logits, config, generator)
            if config.stop_token_id is not None:
                stop_fill = torch.full_like(next_token, config.stop_token_id)
                next_token = torch.where(finished.unsqueeze(-1), stop_fill, next_token)
                finished = finished | next_token.squeeze(-1).eq(config.stop_token_id)
            generated = torch.cat((generated, next_token), dim=1)
            if finished.all() or step + 1 == config.max_new_tokens:
                break

            if config.use_kv_cache:
                if past_key_values is None:
                    raise RuntimeError('KV cache was not returned by the model')
                cache_length = past_key_values[0].sequence_length
                if cache_length >= model.config.max_seq_len:
                    context = generated[:, -model.config.max_seq_len :]
                    cached_logits, past_key_values = model.forward_with_cache(
                        context, past_key_values=None, use_cache=True
                    )
                else:
                    cached_logits, past_key_values = model.forward_with_cache(
                        next_token, past_key_values=past_key_values, use_cache=True
                    )
    finally:
        model.train(was_training)
    return generated


def generate_greedy(
    model: SARNDense,
    input_ids: Tensor,
    max_new_tokens: int,
    use_kv_cache: bool = False,
    stop_token_id: int | None = None,
) -> Tensor:
    return generate(
        model,
        input_ids,
        DecodingConfig(
            strategy='greedy',
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv_cache,
            stop_token_id=stop_token_id,
        ),
    )


def generate_sample(
    model: SARNDense,
    input_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int = 7,
    use_kv_cache: bool = False,
    stop_token_id: int | None = None,
) -> Tensor:
    return generate(
        model,
        input_ids,
        DecodingConfig(
            strategy='sample',
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            use_kv_cache=use_kv_cache,
            stop_token_id=stop_token_id,
        ),
    )
