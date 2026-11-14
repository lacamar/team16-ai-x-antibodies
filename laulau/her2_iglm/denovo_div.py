"""Shared primitives for the *diversified* de-novo path (new, additive module).

The original de-novo path (``her2_iglm.denovo`` + ``train_denovo`` + ``generate_denovo``)
collapsed onto trastuzumab's single framework: fine-tuning on 524k chains that all share
one framework taught the model that framework, so generation reproduced it (mean VH
identity to trastuzumab 0.943, ~20/19101 eligible). Novelty needs *framework* diversity,
which that corpus cannot supply.

This module adds exactly the pieces the diversified path needs, without touching any
existing file:

* ``locate_cdrh3_general`` -- a framework-agnostic CDRH3 locator. ``denovo.cdrh3_from_vh``
  keys off trastuzumab's *literal* flanks (``CSR`` / ``YWGQ``), so it cannot find the loop
  in a *different* framework. Here we anchor on the two invariants every human VH shares:
  the conserved FR3-terminal cysteine and the FR4 ``WG.G`` tryptophan motif. The 10-residue
  slot between them is the CDRH3.
* ``vh_identity_to_trastuzumab`` -- the same positional identity ``generate_denovo`` prints.
* ``DiverseSplits`` + save/load -- variable-length, padded full-generation examples
  (``[HEAVY][HUMAN] <VH> [CLS]``) built from diverse host frameworks grafted with FACS
  binder 10-mers, carrying each 10-mer's binding class.
* ``PaddedFullGenBatcher`` + ``padded_both_nll`` -- the variable-length analogue of
  ``denovo.FullGenBatcher`` / ``denovo.both_nll``: right-padding with an attention mask and
  per-example loss masks, so a batch can mix chain lengths. Returns both the full-chain NLL
  (training objective) and the CDRH3-region NLL (model-selection signal), exactly as the
  fixed-template path does.

Geometry (trastuzumab, for reference)::

    ...Q M N S L R A E D T A V Y Y C | S R W G G D G F Y A M D Y | W G Q G T L V T V S S
                                    Cys        (core, 13)          FR4 'WG.G'

The dataset's 10-mer (``WGGDGFYAMD``) is a trimmed window of that 13-residue core. For a
*diverse* host we do not try to reproduce trastuzumab's idiosyncratic trim: we replace the
entire Cys->FR4 core with a FACS 10-mer, so in every training chain (and therefore in every
generated chain) the conserved Cys and FR4 motif sit *immediately* around a clean 10-mer and
``locate_cdrh3_general`` recovers it exactly.
"""

from __future__ import annotations

import dataclasses
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from her2_iglm import data as d
from her2_iglm.tokenizer import IgLMTokenizer

PROMPT_LEN = 2          # [chain][species]
SPAN_LEN = d.CDRH3_LENGTH  # 10
FG_FIRST_LABEL = PROMPT_LEN  # first residue, predicted from [HEAVY][HUMAN]
#: Canonical human-IgG FR4 (== trastuzumab's own FR4, ``suffix[1:]``). FR4 is near-constant
#: across antibodies, so we supply it rather than depend on the base model emitting a clean
#: one: this keeps every chain's diversity in FR1-FR3 while guaranteeing a clean terminator.
CANONICAL_FR4 = "WGQGTLVTVSS"


# --- framework-agnostic CDRH3 location -------------------------------------------


def locate_cdrh3_general(
    vh: str, min_core: int = 5, max_core: int = 30
) -> Optional[Tuple[int, int, str]]:
    """Locate a CDRH3 core in an arbitrary human VH by its two conserved anchors.

    Anchors: the conserved FR3-terminal cysteine and the FR4 ``WG.G`` tryptophan. We take
    the **first** ``WG.G`` that has a cysteine ``min_core..max_core`` residues upstream --
    i.e. the first properly Cys-anchored FR4. Base IgLM frequently fails to emit [CLS] after
    FR4 and runs on into the start of a *second* VH; taking the first (not the rightmost)
    Cys-anchored FR4 ignores that run-on tail instead of mislocating onto it.

    Returns ``(start, fr4_w, core)`` 0-indexed into ``vh`` (``vh[start:fr4_w] == core``), or
    ``None`` if no clean anchor pair is found. Only the FR4 *start* (the W) is needed to mark
    the CDR3 end; callers keep the diverse prefix ``vh[:start]`` and append a canonical FR4,
    so the host's FR4 need not even be complete.
    """
    n = len(vh)
    i = 1
    while i <= n - 4:
        if vh[i] == "W" and vh[i + 1] == "G" and vh[i + 3] == "G":
            lo = max(0, i - max_core - 1)
            c = -1
            for j in range(i - 1, lo - 1, -1):
                if vh[j] == "C":
                    c = j
                    break
            if c >= 0 and min_core <= i - (c + 1) <= max_core:
                return c + 1, i, vh[c + 1 : i]
        i += 1
    return None


def design_vh(prefix: str, cdrh3: str) -> str:
    """Assemble a clean design VH: diverse ``prefix`` (FR1-FR3 ending at the Cys) + loop + FR4."""
    return prefix + cdrh3 + CANONICAL_FR4


def host_clean_vh(vh: str, fr4_w: int) -> str:
    """Host VH with its run-on tail (if any) dropped and a canonical FR4: for identity/report."""
    return vh[:fr4_w] + CANONICAL_FR4


def vh_identity_to_trastuzumab(vh: str, trast_vh: str) -> float:
    """Positional identity to trastuzumab's VH (same convention as ``generate_denovo``)."""
    match = sum(a == b for a, b in zip(vh, trast_vh))
    return match / max(len(vh), len(trast_vh))


def trastuzumab_vh(splits: d.Splits) -> str:
    return splits.framework_prefix + d.TRASTUZUMAB_CDRH3 + splits.framework_suffix


def graft_cdrh3(vh: str, start: int, end: int, cdrh3: str) -> str:
    """Replace the located core ``vh[start:end]`` with ``cdrh3`` (keeps both anchors)."""
    return vh[:start] + cdrh3 + vh[end:]


# --- variable-length padded dataset ----------------------------------------------


@dataclasses.dataclass
class DiverseSplits:
    """Padded full-generation examples for the diversified de-novo fine-tune.

    ``tokens`` is ``[HEAVY][HUMAN] <VH> [CLS]`` right-padded with [PAD] to ``tokens.shape[1]``.
    ``length`` is the number of real tokens (incl. the terminating [CLS]); ``span_start`` is
    the token index of the first CDRH3 residue (the grafted 10-mer occupies
    ``span_start .. span_start+9``). ``classes`` is the grafted 10-mer's FACS binding class.
    """

    train_tokens: np.ndarray  # (n, Lmax) uint8
    train_length: np.ndarray  # (n,) uint8
    train_span: np.ndarray    # (n,) uint8
    train_classes: np.ndarray  # (n,) uint8
    val_tokens: np.ndarray
    val_length: np.ndarray
    val_span: np.ndarray
    val_classes: np.ndarray
    trast_vh: str
    pad_id: int
    cls_id: int
    meta: dict

    def counts(self) -> dict:
        out = {}
        for name, cls in (("train", self.train_classes), ("val", self.val_classes)):
            out[name] = {
                d.CLASS_NAMES[i]: int((cls == i).sum()) for i in range(len(d.CLASS_NAMES))
            }
            out[name]["total"] = int(cls.size)
        return out


def save_diverse_splits(splits: DiverseSplits, path: str) -> None:
    import json

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        train_tokens=splits.train_tokens,
        train_length=splits.train_length,
        train_span=splits.train_span,
        train_classes=splits.train_classes,
        val_tokens=splits.val_tokens,
        val_length=splits.val_length,
        val_span=splits.val_span,
        val_classes=splits.val_classes,
        trast_vh=np.array(splits.trast_vh),
        pad_id=np.array(splits.pad_id),
        cls_id=np.array(splits.cls_id),
        meta=np.array(json.dumps(splits.meta)),
    )


def load_diverse_splits(path: str) -> DiverseSplits:
    import json

    with np.load(path, allow_pickle=False) as b:
        return DiverseSplits(
            train_tokens=b["train_tokens"],
            train_length=b["train_length"],
            train_span=b["train_span"],
            train_classes=b["train_classes"],
            val_tokens=b["val_tokens"],
            val_length=b["val_length"],
            val_span=b["val_span"],
            val_classes=b["val_classes"],
            trast_vh=str(b["trast_vh"]),
            pad_id=int(b["pad_id"]),
            cls_id=int(b["cls_id"]),
            meta=json.loads(str(b["meta"])),
        )


class PaddedFullGenBatcher:
    """Variable-length analogue of ``denovo.FullGenBatcher``.

    Holds the whole (already padded) token matrix on the GPU as int64 and gathers batches
    there. ``gather`` returns tokens, the attention mask, per-example length and CDRH3 span
    start, and the binding class -- everything ``padded_both_nll`` needs.
    """

    def __init__(self, tokens, length, span, classes, pad_id, device) -> None:
        self.device = device
        self.pad_id = int(pad_id)
        self.tokens = torch.from_numpy(tokens.astype(np.int64)).to(device)
        self.length = torch.from_numpy(length.astype(np.int64)).to(device)
        self.span = torch.from_numpy(span.astype(np.int64)).to(device)
        self.classes = torch.from_numpy(classes.astype(np.int64)).to(device)
        self.size = self.tokens.shape[0]
        self.lmax = self.tokens.shape[1]
        self._pos = torch.arange(self.lmax, device=device)

    def gather(self, index: torch.Tensor):
        tok = self.tokens[index]
        length = self.length[index]
        attn = (self._pos.unsqueeze(0) < length.unsqueeze(1)).long()  # (b, Lmax)
        return tok, attn, length, self.span[index], self.classes[index]


def padded_both_nll(model, tokens, attn, length, span_start):
    """One forward -> (full-chain NLL, CDRH3-region NLL), per example, pad-masked.

    Full region: label positions ``FG_FIRST_LABEL .. length-1`` (all residues + [CLS]).
    CDRH3 region: label positions ``span_start .. span_start+9`` (the grafted 10-mer).
    Teacher forcing: ``logits[:, t-1]`` predicts token ``t``.
    """
    logits = model(tokens, attention_mask=attn).logits  # (b, Lmax, V)
    b, lmax, v = logits.shape
    shift_logits = logits[:, :-1, :]       # predicts tokens[:, 1:]
    shift_labels = tokens[:, 1:]           # (b, Lmax-1), label at token index t=pos+1
    ce = F.cross_entropy(
        shift_logits.reshape(-1, v).float(),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(b, lmax - 1)

    pos = torch.arange(1, lmax, device=tokens.device).unsqueeze(0)  # (1, Lmax-1): label index t
    last = (length - 1).unsqueeze(1)
    full_mask = (pos >= FG_FIRST_LABEL) & (pos <= last)
    s0 = span_start.unsqueeze(1)
    reg_mask = (pos >= s0) & (pos <= s0 + (SPAN_LEN - 1))

    full = (ce * full_mask).sum(1) / full_mask.sum(1).clamp(min=1)
    region = (ce * reg_mask).sum(1) / reg_mask.sum(1).clamp(min=1)
    return full, region
