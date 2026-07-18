'''Deterministic generated datasets for Phase 1 correctness and smoke runs.'''

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from aegis_sarn.sarn.tokenizer import ByteTokenizer


@dataclass(frozen=True, slots=True)
class ToyBatch:
    input_ids: Tensor
    labels: Tensor
    task: str


class GeneratedTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, input_ids: Tensor, labels: Tensor) -> None:
        if input_ids.shape != labels.shape or input_ids.ndim != 2:
            raise ValueError('input_ids and labels must share [examples, sequence] shape')
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.input_ids[index], self.labels[index]


def _repeat_to_length(values: list[int], length: int) -> list[int]:
    if not values:
        raise ValueError('values cannot be empty')
    repeats = (length + len(values) - 1) // len(values)
    return (values * repeats)[:length]


def repeated_pattern_batch(
    batch_size: int = 8,
    sequence_length: int = 32,
    pattern: tuple[int, ...] = (1, 2, 3, 4),
) -> ToyBatch:
    sequence = torch.tensor(
        _repeat_to_length(list(pattern), sequence_length + 1), dtype=torch.long
    )
    input_ids = sequence[:-1].repeat(batch_size, 1)
    labels = sequence[1:].repeat(batch_size, 1)
    return ToyBatch(input_ids=input_ids, labels=labels, task='repeated_pattern')


def copy_task_batch(
    batch_size: int = 8,
    sequence_length: int = 32,
    vocab_size: int = 256,
    seed: int = 7,
) -> ToyBatch:
    if vocab_size < 8:
        raise ValueError('copy task requires vocab_size >= 8')
    generator = torch.Generator().manual_seed(seed)
    rows: list[Tensor] = []
    payload_length = max(2, (sequence_length - 2) // 2)
    for _ in range(batch_size):
        payload = torch.randint(
            0, vocab_size - 2, (payload_length,), generator=generator
        ).tolist()
        values = [vocab_size - 2, *payload, vocab_size - 1, *payload]
        rows.append(torch.tensor(_repeat_to_length(values, sequence_length + 1)))
    sequences = torch.stack(rows).long()
    return ToyBatch(
        input_ids=sequences[:, :-1], labels=sequences[:, 1:], task='copy'
    )


def toy_text_batch(
    batch_size: int = 8,
    sequence_length: int = 32,
    text: str = 'aegis sarn dense baseline. ',
) -> ToyBatch:
    tokenizer = ByteTokenizer()
    encoded = tokenizer.encode(text)
    required = sequence_length + batch_size + 1
    stream = _repeat_to_length(encoded, required)
    rows = [
        torch.tensor(stream[index : index + sequence_length + 1], dtype=torch.long)
        for index in range(batch_size)
    ]
    sequences = torch.stack(rows)
    return ToyBatch(
        input_ids=sequences[:, :-1], labels=sequences[:, 1:], task='toy_text'
    )


def as_dataset(batch: ToyBatch) -> GeneratedTokenDataset:
    return GeneratedTokenDataset(batch.input_ids, batch.labels)
