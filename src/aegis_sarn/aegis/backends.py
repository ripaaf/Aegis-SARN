'''Backend contract plus fake and SARN-Dense implementations.'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from aegis_sarn.aegis.schemas import RunRequest
from aegis_sarn.config import DecodingConfig
from aegis_sarn.sarn import ByteTokenizer, SARNDense, generate
from aegis_sarn.utils import set_global_seed


@dataclass(slots=True)
class BackendOutput:
    text: str
    prompt_tokens: int
    generated_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


class GenerationBackend(Protocol):
    name: str

    def generate(self, request: RunRequest) -> BackendOutput: ...


class FakeBackend:
    name = 'fake'

    def __init__(self, response: str = 'fake-backend-response') -> None:
        self.response = response

    def generate(self, request: RunRequest) -> BackendOutput:
        generated = self.response[: request.max_new_tokens]
        return BackendOutput(
            text=generated,
            prompt_tokens=min(len(request.prompt), request.max_prompt_tokens),
            generated_tokens=len(generated),
            metadata={'deterministic': True},
        )


class SARNBackend:
    name = 'sarn-dense'

    def __init__(
        self,
        model: SARNDense,
        tokenizer: ByteTokenizer | None = None,
        device: str = 'cpu',
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer or ByteTokenizer()
        self.device = torch.device(device)
        if self.model.config.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                'model vocab_size must match the Phase 1 byte tokenizer vocabulary'
            )

    def generate(self, request: RunRequest) -> BackendOutput:
        set_global_seed(request.seed)
        encoded = self.tokenizer.encode(request.prompt)
        encoded = encoded[-request.max_prompt_tokens :]
        if not encoded:
            raise ValueError('prompt produced no tokens')
        input_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)
        decoding = DecodingConfig(
            strategy=request.decoding_strategy,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            stop_token_id=request.stop_token_id,
            use_kv_cache=request.use_kv_cache,
            seed=request.seed,
        )
        generated = generate(self.model, input_ids, decoding)
        new_ids = generated[0, input_ids.shape[1] :].tolist()
        return BackendOutput(
            text=self.tokenizer.decode(new_ids),
            prompt_tokens=input_ids.shape[1],
            generated_tokens=len(new_ids),
            metadata={
                'parameter_count': self.model.count_parameters(),
                'device': str(self.device),
                'decoding': decoding.to_dict(),
            },
        )
