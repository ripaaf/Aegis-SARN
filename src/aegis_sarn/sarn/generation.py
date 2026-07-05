'''Reference greedy generation without KV caching.'''

from __future__ import annotations

import torch
from torch import Tensor

from aegis_sarn.sarn.model import SARNDense


@torch.inference_mode()
def generate_greedy(
    model: SARNDense, input_ids: Tensor, max_new_tokens: int
) -> Tensor:
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError('input_ids must be a non-empty [batch, sequence] tensor')
    if max_new_tokens < 0:
        raise ValueError('max_new_tokens cannot be negative')

    was_training = model.training
    model.eval()
    generated = input_ids
    for _ in range(max_new_tokens):
        context = generated[:, -model.config.max_seq_len :]
        next_token = model(context)[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    model.train(was_training)
    return generated
