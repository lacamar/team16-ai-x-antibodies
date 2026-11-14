"""In-house FACS binder classifier: CDRH3 10-mer -> P(binding class).

Trained on the 524,346-variant trastuzumab FACS screen (the provided multi-modal data),
this is the cheap, dense scorer the RL loop runs on every surviving candidate. It predicts
the three measured affinity classes (low / medium / high); we rank designs by P(high).

Architecture: one-hot (10 x 20) -> MLP. Tiny and exact; trains in well under a minute on
the H100 over the full screen. This is a model we *build from the provided data*, not a
guessed score.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from her2_iglm import data as d
from her2_iglm.tokenizer import IgLMTokenizer

NUM_AA = 20
NUM_CLASSES = len(d.CLASS_NAMES)
HIGH_INDEX = d.CLASS_TO_INDEX["high"]


class FacsClassifier(nn.Module):
    def __init__(self, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d.CDRH3_LENGTH * NUM_AA, hidden),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, NUM_CLASSES),
        )

    def forward(self, x):  # x: (B, 10*20)
        return self.net(x)


def onehot_from_aaindex(aaidx: np.ndarray) -> np.ndarray:
    """(n, 10) int aa-index -> (n, 200) float32 one-hot."""
    n, L = aaidx.shape
    oh = np.zeros((n, L, NUM_AA), dtype=np.float32)
    rows = np.arange(n)[:, None]
    cols = np.arange(L)[None, :]
    oh[rows, cols, aaidx] = 1.0
    return oh.reshape(n, L * NUM_AA)


def strings_to_aaindex(seqs: List[str]) -> np.ndarray:
    aa_order = "ACDEFGHIKLMNPQRSTVWY"
    lut = {aa: i for i, aa in enumerate(aa_order)}
    out = np.empty((len(seqs), d.CDRH3_LENGTH), dtype=np.int64)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            out[i, j] = lut[c]
    return out


def spans_to_aaindex(spans: np.ndarray, tokenizer: IgLMTokenizer) -> np.ndarray:
    """(n, 10) token-id spans -> (n, 10) 0..19 aa-index."""
    aa_order = "ACDEFGHIKLMNPQRSTVWY"
    table = np.full(256, -1, dtype=np.int64)
    for i, aa in enumerate(aa_order):
        table[tokenizer.vocab[aa]] = i
    idx = table[spans.astype(np.int64)]
    if (idx < 0).any():
        raise ValueError("non-residue token in span")
    return idx


class ClassifierBundle:
    """A loaded classifier ready to score CDRH3 strings for P(high)."""

    def __init__(self, model: FacsClassifier, device: torch.device):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def p_class(self, seqs: List[str], batch: int = 65536) -> np.ndarray:
        aaidx = strings_to_aaindex(seqs)
        probs = np.empty((len(seqs), NUM_CLASSES), dtype=np.float32)
        for s in range(0, len(seqs), batch):
            oh = onehot_from_aaindex(aaidx[s : s + batch])
            x = torch.from_numpy(oh).to(self.device)
            p = torch.softmax(self.model(x), dim=-1).cpu().numpy()
            probs[s : s + batch] = p
        return probs

    def p_high(self, seqs: List[str], batch: int = 65536) -> np.ndarray:
        return self.p_class(seqs, batch=batch)[:, HIGH_INDEX]


def load_classifier(path: str, device: torch.device, hidden: int = 256) -> ClassifierBundle:
    meta_path = os.path.join(path, "classifier_meta.json")
    hidden_ = hidden
    if os.path.exists(meta_path):
        hidden_ = json.load(open(meta_path)).get("hidden", hidden)
    model = FacsClassifier(hidden=hidden_)
    sd = torch.load(os.path.join(path, "classifier.pt"), map_location="cpu")
    model.load_state_dict(sd)
    model.to(device)
    return ClassifierBundle(model, device)
