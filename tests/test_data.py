import torch

from aegis_sarn.sarn.data import copy_task_batch, repeated_pattern_batch, toy_text_batch


def test_generated_tasks_have_expected_shapes_and_are_deterministic() -> None:
    first = copy_task_batch(batch_size=3, sequence_length=12, vocab_size=32, seed=5)
    second = copy_task_batch(batch_size=3, sequence_length=12, vocab_size=32, seed=5)
    torch.testing.assert_close(first.input_ids, second.input_ids)
    torch.testing.assert_close(first.labels, second.labels)
    assert first.input_ids.shape == (3, 12)


def test_pattern_and_text_labels_are_shifted() -> None:
    for batch in (
        repeated_pattern_batch(batch_size=2, sequence_length=8),
        toy_text_batch(batch_size=2, sequence_length=8),
    ):
        torch.testing.assert_close(batch.input_ids[:, 1:], batch.labels[:, :-1])
