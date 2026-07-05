from __future__ import annotations

import torch

from aegis_sarn.config import DecodingConfig, ModelConfig
from aegis_sarn.sarn.generation import generate, generate_greedy, generate_sample
from aegis_sarn.sarn.model import SARNDense
from aegis_sarn.utils import set_global_seed


def test_cached_and_uncached_greedy_generation_match(
    tiny_model_config: ModelConfig,
) -> None:
    set_global_seed(41)
    model = SARNDense(tiny_model_config).eval()
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    uncached = generate_greedy(model, prompt, max_new_tokens=14, use_kv_cache=False)
    cached = generate_greedy(model, prompt, max_new_tokens=14, use_kv_cache=True)
    torch.testing.assert_close(uncached, cached, rtol=0.0, atol=0.0)


def test_kv_cache_shape_device_dtype_and_growth(
    tiny_model_config: ModelConfig,
) -> None:
    model = SARNDense(tiny_model_config).eval()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logits, caches = model.forward_with_cache(prompt, use_cache=True)
    assert caches is not None
    assert len(caches) == tiny_model_config.n_layers
    cache = caches[0]
    assert cache.key.shape == (
        1,
        tiny_model_config.n_heads,
        3,
        tiny_model_config.head_dim,
    )
    assert cache.key.device == logits.device
    assert cache.key.dtype == logits.dtype

    _, grown = model.forward_with_cache(
        torch.tensor([[4]], dtype=torch.long), past_key_values=caches, use_cache=True
    )
    assert grown is not None
    assert grown[0].sequence_length == 4


def test_top_k_and_top_p_sampling_are_seeded(
    tiny_model_config: ModelConfig,
) -> None:
    model = SARNDense(tiny_model_config).eval()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    first = generate_sample(
        model,
        prompt,
        max_new_tokens=6,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        seed=53,
        use_kv_cache=True,
    )
    second = generate_sample(
        model,
        prompt,
        max_new_tokens=6,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        seed=53,
        use_kv_cache=True,
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first.shape == (1, 9)


def test_stop_token_ends_generation(tiny_model_config: ModelConfig) -> None:
    model = SARNDense(tiny_model_config).eval()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    stop_token = int(model(prompt)[:, -1, :].argmax().item())
    output = generate(
        model,
        prompt,
        DecodingConfig(
            strategy='greedy', max_new_tokens=8, stop_token_id=stop_token
        ),
    )
    assert output.shape[1] == prompt.shape[1] + 1
    assert output[0, -1].item() == stop_token
