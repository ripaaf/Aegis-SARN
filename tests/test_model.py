from __future__ import annotations

import torch

from aegis_sarn.config import ModelConfig, SeedConfig
from aegis_sarn.eval import language_model_loss
from aegis_sarn.sarn import SARNDense, generate_greedy
from aegis_sarn.sarn.layers import RotaryEmbedding
from aegis_sarn.utils import set_global_seed


def test_output_shape_and_tied_head(tiny_model_config: ModelConfig) -> None:
    model = SARNDense(tiny_model_config)
    inputs = torch.randint(0, tiny_model_config.vocab_size, (3, 7))
    assert model(inputs).shape == (3, 7, tiny_model_config.vocab_size)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_causal_mask_prevents_future_token_leakage(
    tiny_model_config: ModelConfig,
) -> None:
    set_global_seed(SeedConfig(seed=11))
    model = SARNDense(tiny_model_config).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed_future = torch.tensor([[1, 2, 3, 9, 8, 7]])
    with torch.inference_mode():
        first_logits = model(first)
        changed_logits = model(changed_future)
    torch.testing.assert_close(first_logits[:, :3], changed_logits[:, :3])


def test_rope_preserves_shape_device_and_dtype() -> None:
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16, base=10_000.0)
    query = torch.randn(2, 4, 7, 8, dtype=torch.float32)
    key = torch.randn_like(query)
    rotated_query, rotated_key = rope(query, key)
    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert rotated_query.device == query.device
    assert rotated_query.dtype == query.dtype


def test_seeded_initialization_is_deterministic(
    tiny_model_config: ModelConfig,
) -> None:
    set_global_seed(17)
    first = SARNDense(tiny_model_config)
    set_global_seed(17)
    second = SARNDense(tiny_model_config)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_forward_backward_loss_is_finite(tiny_model_config: ModelConfig) -> None:
    model = SARNDense(tiny_model_config)
    inputs = torch.randint(0, tiny_model_config.vocab_size, (2, 8))
    labels = torch.randint(0, tiny_model_config.vocab_size, (2, 8))
    loss = language_model_loss(model(inputs), labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_tiny_greedy_generation(tiny_model_config: ModelConfig) -> None:
    model = SARNDense(tiny_model_config)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generated = generate_greedy(model, prompt, max_new_tokens=5)
    assert generated.shape == (1, 8)
    torch.testing.assert_close(generated[:, :3], prompt)
