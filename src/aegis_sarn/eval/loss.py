'''Language-model loss and evaluation.'''

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from aegis_sarn.sarn.model import SARNDense


def language_model_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.shape[:-1] != labels.shape:
        raise ValueError('logits and labels have incompatible shapes')
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


@torch.inference_mode()
def evaluate_loss(model: SARNDense, input_ids: Tensor, labels: Tensor) -> float:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    loss = language_model_loss(model(input_ids.to(device)), labels.to(device))
    model.train(was_training)
    return float(loss.item())
