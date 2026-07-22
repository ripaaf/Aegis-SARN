'''Deterministic generated datasets for baseline correctness and smoke runs.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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


def _split_seed(seed: int, split: str) -> int:
    return seed + {'train': 0, 'validation': 1_000, 'eval': 1_000}.get(split, 2_000)


def _validate_task_shape(batch_size: int, sequence_length: int, vocab_size: int, minimum_vocab: int = 16) -> None:
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if sequence_length < 2:
        raise ValueError('sequence_length must be at least 2')
    if vocab_size < minimum_vocab:
        raise ValueError(f'toy task requires vocab_size >= {minimum_vocab}')


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
    split: str = 'train',
) -> ToyBatch:
    _validate_task_shape(batch_size, sequence_length, vocab_size, minimum_vocab=8)
    generator = torch.Generator().manual_seed(_split_seed(seed, split))
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


def alternating_pattern_batch(
    batch_size: int = 8,
    sequence_length: int = 32,
    vocab_size: int = 256,
    seed: int = 7,
    split: str = 'train',
) -> ToyBatch:
    _validate_task_shape(batch_size, sequence_length, vocab_size)
    generator = torch.Generator().manual_seed(_split_seed(seed, split))
    rows: list[Tensor] = []
    for _ in range(batch_size):
        a = int(torch.randint(1, min(vocab_size, 8), (1,), generator=generator).item())
        b = int(torch.randint(8, min(vocab_size, 16), (1,), generator=generator).item())
        phase = int(torch.randint(0, 2, (1,), generator=generator).item())
        pattern = [a, b] if phase == 0 else [b, a]
        rows.append(torch.tensor(_repeat_to_length(pattern, sequence_length + 1)))
    sequences = torch.stack(rows).long()
    return ToyBatch(sequences[:, :-1], sequences[:, 1:], 'alternating_pattern')


def modular_counting_batch(
    batch_size: int = 8,
    sequence_length: int = 32,
    vocab_size: int = 256,
    seed: int = 7,
    split: str = 'train',
    modulus: int = 8,
) -> ToyBatch:
    _validate_task_shape(batch_size, sequence_length, vocab_size)
    modulus = min(modulus, vocab_size - 1)
    generator = torch.Generator().manual_seed(_split_seed(seed, split))
    rows: list[Tensor] = []
    for _ in range(batch_size):
        start = int(torch.randint(0, modulus, (1,), generator=generator).item())
        values = [(start + index) % modulus for index in range(sequence_length + 1)]
        rows.append(torch.tensor(values, dtype=torch.long))
    sequences = torch.stack(rows)
    return ToyBatch(sequences[:, :-1], sequences[:, 1:], 'modular_counting')


def bracket_structure_batch(
    batch_size: int = 8,
    sequence_length: int = 32,
    vocab_size: int = 256,
    seed: int = 7,
    split: str = 'train',
) -> ToyBatch:
    _validate_task_shape(batch_size, sequence_length, vocab_size)
    generator = torch.Generator().manual_seed(_split_seed(seed, split))
    open_token, close_token, sep_token = 10, 11, 12
    rows: list[Tensor] = []
    for _ in range(batch_size):
        payload = int(torch.randint(1, 10, (1,), generator=generator).item())
        pattern = [open_token, payload, close_token, sep_token]
        rows.append(torch.tensor(_repeat_to_length(pattern, sequence_length + 1)))
    sequences = torch.stack(rows).long()
    return ToyBatch(sequences[:, :-1], sequences[:, 1:], 'bracket_structure')


def rule_following_batch(
    batch_size: int = 8,
    sequence_length: int = 32,
    vocab_size: int = 256,
    seed: int = 7,
    split: str = 'train',
) -> ToyBatch:
    _validate_task_shape(batch_size, sequence_length, vocab_size)
    generator = torch.Generator().manual_seed(_split_seed(seed, split))
    rules = ((2, 3), (4, 5), (6, 7), (8, 9))
    rows: list[Tensor] = []
    for _ in range(batch_size):
        values: list[int] = []
        while len(values) < sequence_length + 1:
            trigger, response = rules[int(torch.randint(0, len(rules), (1,), generator=generator).item())]
            values.extend([trigger, response])
        rows.append(torch.tensor(values[: sequence_length + 1], dtype=torch.long))
    sequences = torch.stack(rows)
    return ToyBatch(sequences[:, :-1], sequences[:, 1:], 'rule_following')


def relation_chain_batch(
    batch_size: int = 8,
    sequence_length: int = 32,
    vocab_size: int = 256,
    seed: int = 7,
    split: str = 'train',
) -> ToyBatch:
    '''Encode two linked relations followed by a start/endpoint query.'''
    _validate_task_shape(batch_size, sequence_length, vocab_size, minimum_vocab=32)
    generator = torch.Generator().manual_seed(_split_seed(seed, split))
    edge_token, query_token, answer_token = 24, 25, 26
    rows: list[Tensor] = []
    for _ in range(batch_size):
        nodes = (torch.randperm(8, generator=generator)[:3] + 1).tolist()
        a, b, c = (int(node) for node in nodes)
        pattern = [
            a, edge_token, b,
            b, edge_token, c,
            query_token, a, answer_token, c,
        ]
        rows.append(
            torch.tensor(_repeat_to_length(pattern, sequence_length + 1))
        )
    sequences = torch.stack(rows).long()
    return ToyBatch(sequences[:, :-1], sequences[:, 1:], 'relation_chain')


def route_propagation_batch(
    batch_size: int = 8,
    sequence_length: int = 32,
    vocab_size: int = 256,
    seed: int = 7,
    split: str = 'train',
) -> ToyBatch:
    '''Encode a three-hop route and query its reachable endpoint.'''
    _validate_task_shape(batch_size, sequence_length, vocab_size, minimum_vocab=32)
    generator = torch.Generator().manual_seed(_split_seed(seed, split))
    route_token, query_token, answer_token = 24, 25, 26
    rows: list[Tensor] = []
    for _ in range(batch_size):
        nodes = (torch.randperm(10, generator=generator)[:4] + 1).tolist()
        a, b, c, d = (int(node) for node in nodes)
        pattern = [
            a, route_token, b,
            b, route_token, c,
            c, route_token, d,
            query_token, a, answer_token, d,
        ]
        rows.append(
            torch.tensor(_repeat_to_length(pattern, sequence_length + 1))
        )
    sequences = torch.stack(rows).long()
    return ToyBatch(sequences[:, :-1], sequences[:, 1:], 'route_propagation')


def slot_binding_batch(
    batch_size: int = 8,
    sequence_length: int = 32,
    vocab_size: int = 256,
    seed: int = 7,
    split: str = 'train',
) -> ToyBatch:
    '''Encode key/value bindings, a distractor, and a selected-key query.'''
    _validate_task_shape(batch_size, sequence_length, vocab_size, minimum_vocab=32)
    generator = torch.Generator().manual_seed(_split_seed(seed, split))
    bind_token, query_token, answer_token = 24, 25, 26
    rows: list[Tensor] = []
    for _ in range(batch_size):
        keys = (torch.randperm(6, generator=generator)[:3] + 1).tolist()
        values = (torch.randperm(6, generator=generator)[:3] + 10).tolist()
        target_index = int(
            torch.randint(0, 3, (1,), generator=generator).item()
        )
        pattern: list[int] = []
        for key, value in zip(keys, values, strict=True):
            pattern.extend([int(key), bind_token, int(value)])
        pattern.extend(
            [
                query_token,
                int(keys[target_index]),
                answer_token,
                int(values[target_index]),
            ]
        )
        rows.append(
            torch.tensor(_repeat_to_length(pattern, sequence_length + 1))
        )
    sequences = torch.stack(rows).long()
    return ToyBatch(sequences[:, :-1], sequences[:, 1:], 'slot_binding')


def length_extrapolation_batch(
    batch_size: int = 8,
    sequence_length: int = 32,
    vocab_size: int = 256,
    seed: int = 7,
    split: str = 'train',
) -> ToyBatch:
    '''Use a longer deterministic route on validation/eval than on train.'''
    _validate_task_shape(batch_size, sequence_length, vocab_size, minimum_vocab=32)
    generator = torch.Generator().manual_seed(_split_seed(seed, split))
    edge_token, query_token, answer_token = 24, 25, 26
    edge_count = 2 if split == 'train' else 3
    rows: list[Tensor] = []
    for _ in range(batch_size):
        nodes = (
            torch.randperm(10, generator=generator)[: edge_count + 1] + 1
        ).tolist()
        pattern: list[int] = []
        for source, target in zip(nodes[:-1], nodes[1:], strict=True):
            pattern.extend([int(source), edge_token, int(target)])
        pattern.extend(
            [query_token, int(nodes[0]), answer_token, int(nodes[-1])]
        )
        rows.append(
            torch.tensor(_repeat_to_length(pattern, sequence_length + 1))
        )
    sequences = torch.stack(rows).long()
    return ToyBatch(
        sequences[:, :-1], sequences[:, 1:], 'length_extrapolation'
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


TaskFactory = Callable[[int, int, int, int, str], ToyBatch]


def _repeated_factory(batch_size: int, sequence_length: int, vocab_size: int, seed: int, split: str) -> ToyBatch:
    del vocab_size, seed, split
    return repeated_pattern_batch(batch_size, sequence_length)


TOY_TASK_FACTORIES: dict[str, TaskFactory] = {
    'repeated_pattern': _repeated_factory,
    'copy': copy_task_batch,
    'alternating_pattern': alternating_pattern_batch,
    'modular_counting': modular_counting_batch,
    'bracket_structure': bracket_structure_batch,
    'rule_following': rule_following_batch,
    'relation_chain': relation_chain_batch,
    'route_propagation': route_propagation_batch,
    'slot_binding': slot_binding_batch,
    'length_extrapolation': length_extrapolation_batch,
}


TOY_TASK_NAMES = tuple(TOY_TASK_FACTORIES.keys())
GRAPH_TASK_NAMES = (
    'relation_chain',
    'route_propagation',
    'slot_binding',
    'length_extrapolation',
)


def make_toy_task_batch(
    task_name: str,
    batch_size: int = 8,
    sequence_length: int = 32,
    vocab_size: int = 256,
    seed: int = 7,
    split: str = 'validation',
) -> ToyBatch:
    try:
        factory = TOY_TASK_FACTORIES[task_name]
    except KeyError as error:
        raise ValueError(f'unknown toy task: {task_name}') from error
    return factory(batch_size, sequence_length, vocab_size, seed, split)


def as_dataset(batch: ToyBatch) -> GeneratedTokenDataset:
    return GeneratedTokenDataset(batch.input_ids, batch.labels)
