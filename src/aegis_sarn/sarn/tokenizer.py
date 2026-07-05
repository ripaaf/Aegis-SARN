'''Dependency-free byte tokenizer used by the Phase 1 CLI and toy data.'''

from __future__ import annotations


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode('utf-8'))

    def decode(self, token_ids: list[int]) -> str:
        values = bytes(token_id % self.vocab_size for token_id in token_ids)
        return values.decode('utf-8', errors='replace')
