"""De-novo path primitives: full-antibody generation + full-sequence fine-tuning.

The optimisation path (``data``/``train``/``generate``) rewrites only the 10-residue
CDRH3 inside trastuzumab's fixed framework, via the ``IgLM.infill`` framing. The de-novo
path instead uses the ``IgLM.generate`` framing to emit a *whole* heavy chain, and
fine-tunes the model on *full-sequence* examples so that generation itself is
contextualised to HER2.

How HER2 enters the model
-------------------------
IgLM has no antigen input -- it conditions only on two control tokens, [HEAVY] and
[HUMAN]. HER2 can therefore only enter through the *weights*. We fine-tune on the 524k
trastuzumab VH variants, each FACS-labelled by measured HER2 binding, with a class
weighting that pulls probability mass toward the strong binders. The model never "sees"
HER2; it learns which full heavy chains are the likely / strong-binding ones, and because
that label *was* HER2 binding, its generative prior becomes implicitly HER2-conditioned.

Token layout for a full-generation example (length 123), matching ``IgLM.generate``'s
``[chain][species] <residues> [CLS]`` framing (``IgLM._generate`` uses [CLS] as EOS):

    [HEAVY] [HUMAN] <vh[0:120]> [CLS]
    0       1        2..121      122

The CDRH3 sits at vh[98:108] -> token positions 100..110. Training loss is the
class-weighted mean cross-entropy over the whole chain + terminating [CLS] (so the model
stays a coherent full-chain generator that terminates); model *selection* uses the
held-out CDRH3-region NLL of strong binders -- the slice that actually carries the
HER2-binder signal, directly comparable to the optimisation run's ``val_nll_high``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from her2_iglm import data as d
from her2_iglm.tokenizer import IgLMTokenizer

PROMPT_LEN = 2  # [chain][species]
FG_SPAN_START = PROMPT_LEN + d.CDRH3_RANGE[0]   # 100
FG_SPAN_END = PROMPT_LEN + d.CDRH3_RANGE[1]     # 110
FG_CLS_POS = PROMPT_LEN + d.VH_LENGTH           # 122
FG_SEQ_LENGTH = FG_CLS_POS + 1                  # 123
FG_FIRST_LABEL = PROMPT_LEN                     # 2: first residue, predicted from [HEAVY][HUMAN]


def build_fullgen_template(
    splits: d.Splits,
    tokenizer: IgLMTokenizer,
    chain_token: str = "[HEAVY]",
    species_token: str = "[HUMAN]",
) -> np.ndarray:
    """Constant 123-token full-generation row; CDRH3 span (100..110) overwritten per example."""
    t = np.zeros(FG_SEQ_LENGTH, dtype=np.int64)
    t[0] = tokenizer.vocab[chain_token]
    t[1] = tokenizer.vocab[species_token]
    t[PROMPT_LEN:FG_SPAN_START] = tokenizer.encode_residues(splits.framework_prefix)
    t[FG_SPAN_START:FG_SPAN_END] = tokenizer.unk_token_id  # per-row CDRH3
    t[FG_SPAN_END:FG_CLS_POS] = tokenizer.encode_residues(splits.framework_suffix)
    t[FG_CLS_POS] = tokenizer.cls_token_id
    return t


class FullGenBatcher:
    """Builds full-sequence batches by writing CDRH3 spans into the constant template.

    Reuses the same cached (n,10) spans + classes the optimisation path uses; only the
    template and span positions differ. The whole dataset lives on the GPU as int64.
    """

    def __init__(self, template, spans, classes, device) -> None:
        self.device = device
        self.template = torch.from_numpy(template.astype(np.int64)).to(device)
        self.spans = torch.from_numpy(spans.astype(np.int64)).to(device)
        self.classes = torch.from_numpy(classes.astype(np.int64)).to(device)
        self.size = self.spans.shape[0]

    def gather(self, index: torch.Tensor):
        tokens = self.template.unsqueeze(0).repeat(index.numel(), 1)
        tokens[:, FG_SPAN_START:FG_SPAN_END] = self.spans[index]
        return tokens, self.classes[index]


def _ce_region(logits, tokens, first_label: int, last_label: int) -> torch.Tensor:
    """Per-example mean CE over labels[first_label..last_label] (inclusive)."""
    shift_logits = logits[:, first_label - 1 : last_label, :]
    shift_labels = tokens[:, first_label : last_label + 1]
    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(shift_labels.shape)
    return nll.mean(dim=1)


def fullseq_nll(model, tokens) -> torch.Tensor:
    """Training objective: mean CE over the whole chain + terminating [CLS] (121 targets)."""
    logits = model(tokens).logits
    return _ce_region(logits, tokens, FG_FIRST_LABEL, FG_CLS_POS)


def both_nll(model, tokens) -> Tuple[torch.Tensor, torch.Tensor]:
    """One forward -> (full-sequence NLL, CDRH3-region NLL). Region is the selection signal."""
    logits = model(tokens).logits
    full = _ce_region(logits, tokens, FG_FIRST_LABEL, FG_CLS_POS)
    region = _ce_region(logits, tokens, FG_SPAN_START, FG_SPAN_END - 1)  # 10 CDRH3 residues
    return full, region


# --- generation -----------------------------------------------------------------


@torch.no_grad()
def generate_full_vh(
    model,
    tokenizer: IgLMTokenizer,
    n: int,
    device: torch.device,
    prompt_residues: str = "",
    chain_token: str = "[HEAVY]",
    species_token: str = "[HUMAN]",
    temperature: float = 1.0,
    top_p: float = 0.95,
    batch_size: int = 512,
    seed: Optional[int] = None,
    min_len: int = 100,
    max_len: int = 140,
) -> List[str]:
    """Draw ``n`` full heavy-chain samples (with duplicates) via the ``.generate`` framing.

    Prompt = [chain][species] (+ optional seed residues). Generation terminates on [CLS];
    a sample is kept if it terminated and decodes to ``min_len..max_len`` residues.
    """
    start = [tokenizer.vocab[chain_token], tokenizer.vocab[species_token]]
    start += tokenizer.encode_residues(prompt_residues) if prompt_residues else []
    prompt = torch.tensor([start], dtype=torch.int64, device=device)

    if seed is not None:
        torch.manual_seed(seed)

    out: List[str] = []
    model.eval()
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        gen = model.generate(
            prompt,
            max_new_tokens=max_len + 2,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=b,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.cls_token_id,
        )
        for row in gen:
            tail = row[PROMPT_LEN:].tolist()
            if tokenizer.cls_token_id not in tail:
                continue  # never terminated
            body = tail[: tail.index(tokenizer.cls_token_id)]
            seq = (prompt_residues or "") + tokenizer.decode_residues(body)
            if min_len <= len(seq) <= max_len:
                out.append(seq)
        remaining -= b
    return out


def cdrh3_from_vh(vh: str, splits: d.Splits) -> Optional[str]:
    """Locate the 10-residue CDRH3 in a generated VH.

    Fast path: an on-template VH (same framework, length 120) -> vh[98:108]. Otherwise
    locate by the conserved flanks (…CSR | YWGQG here). Returns None if no clean 10-mer
    is found -- such a design is treated as ineligible rather than guessed.
    """
    start, end = d.CDRH3_RANGE
    if (
        len(vh) == d.VH_LENGTH
        and vh[:start] == splits.framework_prefix
        and vh[end:] == splits.framework_suffix
    ):
        return vh[start:end]
    left = splits.framework_prefix[-3:]   # CSR
    right = splits.framework_suffix[:4]   # YWGQ
    li = vh.find(left)
    if li < 0:
        return None
    ri = vh.find(right, li + len(left))
    if ri < 0:
        return None
    cd = vh[li + len(left) : ri]
    return cd if len(cd) == d.CDRH3_LENGTH else None
