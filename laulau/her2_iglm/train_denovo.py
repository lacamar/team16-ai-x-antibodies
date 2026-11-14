"""Stage 2 (de-novo path): class-weighted *full-sequence* fine-tune of IgLM.

Same data, class weights, cosine-LR-with-warmup, early stopping and checkpoint layout as
``her2_iglm.train`` (the optimisation path), with two differences that define the de-novo
path:

* the training objective is the full-chain NLL (``denovo.fullseq_nll``) over the whole
  123-token ``[HEAVY][HUMAN] <VH> [CLS]`` example, so the fine-tuned model stays a
  coherent full-antibody generator for ``IgLM.generate`` (not just a CDRH3 infiller);
* model selection / early stopping use the held-out **CDRH3-region** NLL of strong
  binders (``val_nll_high``) -- the slice that carries the HER2-binder signal. Selecting
  on the full-sequence NLL would be dominated by the (class-invariant) framework and give
  no usable binder signal, so we isolate the region exactly as the optimisation run does.

Early stopping is load-bearing here for two reasons: it stops at maximal strong-binder
preference before the model overfits the narrow single-framework corpus, and it preserves
the pretrained framework diversity that the de-novo <70%-identity novelty filter depends
on. Over-train and generation collapses onto trastuzumab's framework.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from her2_iglm import data as data_mod
from her2_iglm import denovo
from her2_iglm.model import load_model, parameter_count
from her2_iglm.tokenizer import IgLMTokenizer
from her2_iglm.train import TrainConfig, RunLogger, save_checkpoint


@torch.no_grad()
def evaluate(model, batcher, config, autocast_dtype) -> Dict[str, float]:
    """Held-out NLL by binding class: CDRH3-region (selection) and full-sequence (diagnostic)."""
    model.eval()
    nclass = len(data_mod.CLASS_NAMES)
    tot_region = torch.zeros(nclass, device=batcher.device)
    tot_full = torch.zeros(nclass, device=batcher.device)
    counts = torch.zeros(nclass, device=batcher.device)
    for start in range(0, batcher.size, config.eval_batch_size):
        index = torch.arange(start, min(start + config.eval_batch_size, batcher.size),
                             device=batcher.device)
        tokens, classes = batcher.gather(index)
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            full, region = denovo.both_nll(model, tokens)
        tot_region.index_add_(0, classes, region.float())
        tot_full.index_add_(0, classes, full.float())
        counts.index_add_(0, classes, torch.ones_like(region, dtype=torch.float))
    model.train()

    region = (tot_region / counts.clamp(min=1)).tolist()
    full = (tot_full / counts.clamp(min=1)).tolist()
    m = {f"val_nll_{n}": region[i] for i, n in enumerate(data_mod.CLASS_NAMES)}
    m.update({f"val_fullnll_{n}": full[i] for i, n in enumerate(data_mod.CLASS_NAMES)})
    m["val_ppl_high"] = math.exp(min(m["val_nll_high"], 20.0))
    m["separation"] = m["val_nll_low"] - m["val_nll_high"]
    return m


def train(config: TrainConfig) -> Dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = (
        torch.bfloat16
        if config.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )

    os.makedirs(config.run_dir, exist_ok=True)
    logger = RunLogger(config.run_dir)
    with open(os.path.join(config.run_dir, "config.json"), "w") as fh:
        json.dump({**dataclasses.asdict(config), "path": "denovo-fullseq"}, fh, indent=2)

    tokenizer = IgLMTokenizer()
    logger.line(f"[denovo] device={device} autocast={autocast_dtype} seed={config.seed}")

    splits = data_mod.prepare(config.csv_path, config.cache_path, tokenizer)
    logger.line(f"data counts: {json.dumps(splits.counts())}")

    train_spans, train_classes = splits.train_spans, splits.train_classes
    if config.loss_mode == "high-only":
        keep = train_classes == data_mod.CLASS_TO_INDEX["high"]
        train_spans, train_classes = train_spans[keep], train_classes[keep]
        logger.line(f"loss_mode=high-only -> {int(keep.sum())} training rows")

    template = denovo.build_fullgen_template(splits, tokenizer)
    train_batcher = denovo.FullGenBatcher(template, train_spans, train_classes, device)
    val_batcher = denovo.FullGenBatcher(template, splits.val_spans, splits.val_classes, device)

    model = load_model(config.model, tokenizer=tokenizer, device=device)
    model.train()
    logger.line(f"model={config.model} params={parameter_count(model):,} "
                f"fullgen_len={denovo.FG_SEQ_LENGTH}")

    weights = torch.tensor(config.class_weights(), device=device, dtype=torch.float)
    steps_per_epoch = max(1, train_batcher.size // config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay, betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < config.warmup_steps:
            return config.learning_rate * step / max(1, config.warmup_steps)
        prog = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, prog))))

    baseline = evaluate(model, val_batcher, config, autocast_dtype)
    logger.line("baseline (pretrained IgLM, full-seq): "
                + " ".join(f"{k}={v:.4f}" for k, v in baseline.items()))
    logger.metric({"step": 0, "phase": "baseline", **baseline})

    best_metric = baseline["val_nll_high"]
    best_step = 0
    stale = 0
    stop_reason = "completed"
    started = time.time()
    deadline = started + config.max_hours * 3600
    gen = torch.Generator(device="cpu").manual_seed(config.seed)

    from tqdm.auto import tqdm
    progress = tqdm(total=total_steps, desc="finetune-denovo", dynamic_ncols=True, smoothing=0.05)
    running: List[float] = []
    step = 0
    try:
        for epoch in range(config.epochs):
            order = torch.randperm(train_batcher.size, generator=gen)
            for bs in range(0, steps_per_epoch * config.batch_size, config.batch_size):
                index = order[bs:bs + config.batch_size].to(device)
                tokens, classes = train_batcher.gather(index)
                lr = lr_at(step)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    nll = denovo.fullseq_nll(model, tokens)
                ew = weights[classes]
                loss = (nll * ew).sum() / ew.sum().clamp(min=1e-6)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                running.append(loss.item())
                progress.update(1)

                if step % config.log_every == 0:
                    mean_loss = float(np.mean(running[-config.log_every:]))
                    elapsed = time.time() - started
                    progress.set_postfix(loss=f"{mean_loss:.4f}", best=f"{best_metric:.4f}",
                                         lr=f"{lr:.2e}", refresh=False)
                    logger.metric({"step": step, "phase": "train", "epoch": epoch,
                                   "train_loss": mean_loss, "lr": lr, "grad_norm": float(grad_norm),
                                   "elapsed_s": elapsed,
                                   "examples_per_s": step * config.batch_size / elapsed})

                if step % config.eval_every == 0 or step == total_steps:
                    metrics = evaluate(model, val_batcher, config, autocast_dtype)
                    improved = metrics["val_nll_high"] < best_metric - 1e-5
                    logger.metric({"step": step, "phase": "val", "epoch": epoch,
                                   "elapsed_s": time.time() - started, "improved": improved, **metrics})
                    logger.line(
                        f"step {step}/{total_steps} ep{epoch} "
                        f"train={float(np.mean(running[-config.eval_every:])):.4f} "
                        f"val_high={metrics['val_nll_high']:.4f} val_low={metrics['val_nll_low']:.4f} "
                        f"sep={metrics['separation']:+.4f} fullnll_high={metrics['val_fullnll_high']:.4f} "
                        f"{'IMPROVED' if improved else f'no-improve x{stale + 1}'}")
                    meta = {"step": step, "epoch": epoch, "metrics": metrics, "baseline": baseline,
                            "config": dataclasses.asdict(config), "path": "denovo-fullseq",
                            "chain_token": "[HEAVY]", "species_token": "[HUMAN]",
                            "infill_range": list(data_mod.CDRH3_RANGE),
                            "framework_prefix": splits.framework_prefix,
                            "framework_suffix": splits.framework_suffix,
                            "vl_sequence": splits.vl_sequence}
                    if improved:
                        best_metric = metrics["val_nll_high"]
                        best_step = step
                        stale = 0
                        save_checkpoint(model, tokenizer, os.path.join(config.run_dir, "best"), meta)
                    else:
                        stale += 1
                        if stale >= config.patience:
                            stop_reason = "early_stopping"
                            raise StopIteration
                    save_checkpoint(model, tokenizer, os.path.join(config.run_dir, "last"), meta)

                if time.time() > deadline:
                    stop_reason = "time_budget"
                    raise StopIteration
    except StopIteration:
        pass
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    finally:
        progress.close()

    elapsed = time.time() - started
    summary = {"path": "denovo-fullseq", "stop_reason": stop_reason, "steps": step,
               "total_steps_planned": total_steps, "elapsed_s": elapsed, "elapsed_h": elapsed / 3600,
               "best_val_nll_high": best_metric, "best_step": best_step,
               "baseline_val_nll_high": baseline["val_nll_high"],
               "improvement": baseline["val_nll_high"] - best_metric,
               "best_checkpoint": os.path.join(config.run_dir, "best")}
    with open(os.path.join(config.run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.line("summary: " + json.dumps(summary))
    logger.close()
    return summary
