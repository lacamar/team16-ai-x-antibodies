"""RAFT / expert-iteration fine-tune step for the RL loop.

Reward-rAnked FineTuning (Dong et al. 2023): each cycle we sample from the current policy,
keep the eligible high-reward "elite" CDRH3s, and fine-tune the policy on them. Repeated,
this is a stable policy-improvement loop that moves the model toward generating novel,
eligible, predicted-strong HER2 binders -- without PPO's instability and without needing a
differentiable reward (our reward comes from the FACS classifier and co-folding).

KL anchoring to the previous policy is implicit and practical: we initialise from the
previous checkpoint, use a small learning rate and only a few passes over the elite set,
and early-stop. That keeps the policy close to its prior (a real antibody distribution)
rather than collapsing onto a few sequences.

The infilling loss is identical to ``train.example_nll`` -- the span + [CLS] slice that
``IgLM.log_likelihood`` scores -- so the RL-tuned checkpoint stays a drop-in IgLM infiller.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from her2_iglm import data as data_mod
from her2_iglm.model import load_model
from her2_iglm.tokenizer import IgLMTokenizer
from her2_iglm.train import Batcher, example_nll, evaluate, save_checkpoint, TrainConfig


def spans_from_strings(seqs: List[str], tokenizer: IgLMTokenizer) -> np.ndarray:
    """(K,10) uint8 token-id spans from CDRH3 strings."""
    out = np.empty((len(seqs), data_mod.CDRH3_LENGTH), dtype=np.uint8)
    for i, s in enumerate(seqs):
        out[i] = tokenizer.encode_residues(s)
    return out


def raft_finetune(
    from_ckpt: str,
    elite_seqs: List[str],
    elite_weights: Optional[np.ndarray],
    out_dir: str,
    splits,
    device: torch.device,
    epochs: int = 4,
    batch_size: int = 256,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    warmup_frac: float = 0.1,
    bf16: bool = True,
    seed: int = 0,
    log=print,
) -> Dict:
    """Fine-tune the policy at ``from_ckpt`` on the elite CDRH3 set; save to ``out_dir``.

    elite_weights: per-elite reward weight (e.g. classifier P(high)); normalised internally.
    Returns a summary dict with before/after val NLL of the standard high-binder val set
    (a drift diagnostic, not the selection metric here).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    tokenizer = IgLMTokenizer()
    autocast_dtype = (
        torch.bfloat16 if bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )

    spans = spans_from_strings(elite_seqs, tokenizer)
    K = len(elite_seqs)
    if elite_weights is None:
        elite_weights = np.ones(K, dtype=np.float32)
    w = torch.from_numpy(elite_weights.astype(np.float32)).to(device)
    w = w / w.mean().clamp(min=1e-6)  # mean-1 normalise so LR scale is stable

    template = torch.from_numpy(splits.template.astype(np.int64)).to(device)
    span_t = torch.from_numpy(spans.astype(np.int64)).to(device)

    model = load_model(from_ckpt, tokenizer=tokenizer, device=device)
    model.train()

    cfg = TrainConfig(bf16=bf16)
    val_batcher = Batcher(splits.template, splits.val_spans, splits.val_classes, device)
    before = evaluate(model, val_batcher, cfg, autocast_dtype)

    steps_per_epoch = max(1, K // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * warmup_frac))
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay, betas=(0.9, 0.95))

    def lr_at(step):
        if step < warmup:
            return learning_rate * step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return learning_rate * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    gen = torch.Generator(device="cpu").manual_seed(seed)
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        order = torch.randperm(K, generator=gen)
        for bs in range(0, steps_per_epoch * batch_size, batch_size):
            idx = order[bs : bs + batch_size].to(device)
            tokens = template.unsqueeze(0).repeat(idx.numel(), 1)
            tokens[:, data_mod.SPAN_START : data_mod.SPAN_END] = span_t[idx]
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                nll = example_nll(model, tokens)
            bw = w[idx]
            loss = (nll * bw).sum() / bw.sum().clamp(min=1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        log(f"    raft epoch {epoch+1}/{epochs} loss {loss.item():.4f} lr {lr:.2e}")

    after = evaluate(model, val_batcher, cfg, autocast_dtype)
    meta = {
        "from_ckpt": from_ckpt,
        "n_elite": K,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "val_nll_high_before": before["val_nll_high"],
        "val_nll_high_after": after["val_nll_high"],
        "elapsed_s": time.time() - t0,
        "infill_range": list(data_mod.CDRH3_RANGE),
        "framework_prefix": splits.framework_prefix,
        "framework_suffix": splits.framework_suffix,
        "vl_sequence": splits.vl_sequence,
    }
    save_checkpoint(model, tokenizer, out_dir, meta)
    log(f"    raft done: {K} elites, val_nll_high {before['val_nll_high']:.3f}"
        f"->{after['val_nll_high']:.3f}, saved {out_dir}")
    return meta
