"""Exact re-implementation of IgLM's tokenizer, independent of transformers.

Why this exists
---------------
Upstream IgLM builds its tokenizer with ``transformers.BertTokenizerFast(vocab_file=...)``
(see ``iglm/model/IgLM.py``). Under transformers 5.x that constructor silently ignores
``vocab_file``: it yields ``vocab_size == 5`` and maps every residue to ``[UNK]``.
That breaks both generation and any training built on it.

``iglm/trained_models/vocab.txt`` is a flat 33-line list, so the mapping is just
"line number minus one". We read that file directly, which is exact, deterministic and
immune to tokenizer API churn. The ids produced here are identical to what
transformers 4.5.1 produced, which is what the released weights were trained with.

Sanity anchor: ``iglm/model/tokens.py`` declares
``BAD_WORD_IDS = [[0], [1], [25] ... [32]]`` -- i.e. PAD, UNK and the eight
conditioning tokens. That is only consistent with PAD=0, UNK=1, [CAMEL]=25 ...
[LIGHT]=32, which is exactly the ordering in vocab.txt.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence

import numpy as np

import iglm

VOCAB_FILE = os.path.join(
    os.path.dirname(os.path.realpath(iglm.__file__)), "trained_models", "vocab.txt"
)

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

#: Amino acids, in the order they appear in vocab.txt.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class IgLMTokenizer:
    """Character-level tokenizer over IgLM's 33-token vocabulary."""

    def __init__(self, vocab_file: str = VOCAB_FILE) -> None:
        with open(vocab_file) as handle:
            self.tokens: List[str] = [line.strip() for line in handle if line.strip()]

        self.vocab = {token: idx for idx, token in enumerate(self.tokens)}
        if len(self.vocab) != len(self.tokens):
            raise ValueError(f"duplicate tokens in {vocab_file}")

        self.pad_token_id = self.vocab[PAD_TOKEN]
        self.unk_token_id = self.vocab[UNK_TOKEN]
        self.cls_token_id = self.vocab[CLS_TOKEN]
        self.sep_token_id = self.vocab[SEP_TOKEN]
        self.mask_token_id = self.vocab[MASK_TOKEN]

        # Guard against a reordered vocab.txt silently corrupting training.
        expected = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            CLS_TOKEN: 2,
            SEP_TOKEN: 3,
            MASK_TOKEN: 4,
            "A": 5,
            "Y": 24,
            "[CAMEL]": 25,
            "[HUMAN]": 26,
            "[HEAVY]": 31,
            "[LIGHT]": 32,
        }
        for token, idx in expected.items():
            if self.vocab.get(token) != idx:
                raise ValueError(
                    f"unexpected vocab layout: {token!r} is "
                    f"{self.vocab.get(token)}, expected {idx}. "
                    "The released IgLM weights assume the original ordering."
                )

        self.residue_ids = np.array(
            [self.vocab[aa] for aa in AMINO_ACIDS], dtype=np.int64
        )

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> List[int]:
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
        return [self.tokens[int(i)] for i in ids]

    def encode_residues(self, sequence: str) -> List[int]:
        """Encode a plain amino-acid string. Raises on anything unrecognised."""
        ids = []
        for position, residue in enumerate(sequence):
            if residue not in self.vocab:
                raise ValueError(
                    f"residue {residue!r} at position {position} is not in the "
                    "IgLM vocabulary"
                )
            ids.append(self.vocab[residue])
        return ids

    def decode_residues(self, ids: Iterable[int]) -> str:
        """Decode ids back to an amino-acid string, dropping special tokens."""
        return "".join(
            self.tokens[int(i)] for i in ids if 5 <= int(i) <= 24
        )
