"""Sample CDRH3 10-mers from a (fine-tuned) IgLM checkpoint.

Reuses the exact infilling prompt the model was trained on (``data`` builds the template)
and ``model.generate`` -- the generation path validated in ``scripts/_reload_test.py``.
Returns distinct, valid 10-residue spans that terminate on [CLS].
"""
from __future__ import annotations

from typing import List, Optional, Set

import numpy as np
import torch

from her2_iglm import data as d
from her2_iglm.tokenizer import IgLMTokenizer


@torch.no_grad()
def sample_cdrh3(
    model,
    tokenizer: IgLMTokenizer,
    n: int,
    device: torch.device,
    temperature: float = 1.0,
    top_p: float = 0.95,
    batch_size: int = 2048,
    seed: Optional[int] = None,
    splits: Optional[d.Splits] = None,
) -> List[str]:
    """Draw ``n`` raw CDRH3 samples (with duplicates); each a valid terminating 10-mer or dropped."""
    if splits is None:
        splits = d.load_splits("data/processed/facs_splits.npz")
    prompt = torch.from_numpy(splits.template[: d.SPAN_START].astype("int64")).unsqueeze(0).to(device)

    if seed is not None:
        torch.manual_seed(seed)

    out: List[str] = []
    model.eval()
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        gen = model.generate(
            prompt,
            max_new_tokens=d.CDRH3_LENGTH + 2,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=b,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.cls_token_id,
        )
        for row in gen:
            tail = row[d.SPAN_START:].tolist()
            if tokenizer.cls_token_id in tail:
                span = tail[: tail.index(tokenizer.cls_token_id)]
            else:
                continue  # never terminated -> invalid
            seq = tokenizer.decode_residues(span)
            if len(seq) == d.CDRH3_LENGTH:
                out.append(seq)
        remaining -= b
    return out


def generate_distinct_novel(
    model,
    tokenizer: IgLMTokenizer,
    target: int,
    device: torch.device,
    db_set: Set[str],
    exclude: Optional[Set[str]] = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    batch_size: int = 2048,
    max_rounds: int = 400,
    seed: int = 0,
    splits: Optional[d.Splits] = None,
    log=print,
) -> List[str]:
    """Sample until ``target`` distinct CDRH3s are collected that are not in ``db_set``
    (the 524k screen) nor ``exclude`` (e.g. candidates already seen in earlier cycles).

    Returns the list of distinct novel CDRH3 strings (order = discovery order).
    """
    if splits is None:
        splits = d.load_splits("data/processed/facs_splits.npz")
    exclude = exclude or set()
    seen: Set[str] = set()
    kept: List[str] = []
    raw_total = 0
    for rnd in range(max_rounds):
        if len(kept) >= target:
            break
        batch = sample_cdrh3(
            model, tokenizer, batch_size, device,
            temperature=temperature, top_p=top_p, batch_size=batch_size,
            seed=seed + rnd, splits=splits,
        )
        raw_total += batch_size
        for s in batch:
            if s in seen or s in db_set or s in exclude:
                continue
            seen.add(s)
            kept.append(s)
        if rnd % 10 == 0 or len(kept) >= target:
            log(f"  gen round {rnd}: {len(kept)}/{target} distinct-novel "
                f"(raw {raw_total}, yield {len(kept)/max(raw_total,1):.3f})")
    return kept[:target]
