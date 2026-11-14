"""Stage-2 of the *diversified* de-novo path: class-weighted full-sequence fine-tune on
framework-diverse grafted chains.

Same objective, class weights (0.05 / 0.3 / 1.0), cosine-LR-with-warmup, early stopping and
checkpoint layout as ``her2_iglm.train_denovo``. Two differences:

* batches are **variable-length / padded** (``denovo_div.PaddedFullGenBatcher``) because the
  corpus now spans many frameworks of different lengths, whereas ``train_denovo`` could use
  one constant template;
* at every evaluation it **samples 500 chains** from the model-in-training and logs the mean
  (and median) VH identity to trastuzumab and the fraction passing the eligibility gates --
  so collapse-back-onto-trastuzumab is visible live, not only at the end.

Model selection / early stopping use the held-out CDRH3-region NLL of strong binders
(``val_nll_high``), identical to the other paths: it is the slice that carries the binder
signal and is directly comparable across runs. Early stopping is still load-bearing -- it
preserves the framework diversity the <70%-identity novelty filter depends on.

Run inside the CUDA container:

    env/run.sh her2_iglm/train_denovo_div.py --run-dir runs/finetune_denovo_div \
      --diverse-path data/processed/diverse_splits.npz \
      --batch-size 128 --learning-rate 5e-5 --warmup-steps 300 --epochs 30 \
      --max-hours 0.75 --eval-every 500 --log-every 25 --patience 20 \
      --loss-mode weighted --weight-low 0.05 --weight-medium 0.3 --weight-high 1.0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import statistics
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from her2_iglm import data as data_mod
from her2_iglm import denovo
from her2_iglm import denovo_div as dv
from her2_iglm.eligibility import eligibility_report
from her2_iglm.model import load_model, parameter_count
from her2_iglm.tokenizer import IgLMTokenizer
from her2_iglm.train import TrainConfig, RunLogger, save_checkpoint
from scripts.finetune import build_parser


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
        tokens, attn, length, span, classes = batcher.gather(index)
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            full, region = dv.padded_both_nll(model, tokens, attn, length, span)
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


@torch.no_grad()
def sample_diagnostic(model, tok, device, trast_vh, provided_matrix, n, temperature,
                      top_p, seed) -> Dict[str, float]:
    """Sample ``n`` chains; log mean/median VH identity to trastuzumab and eligible fraction."""
    raw = denovo.generate_full_vh(
        model, tok, n, device, temperature=temperature, top_p=top_p,
        batch_size=min(512, n), seed=seed, min_len=90, max_len=140,
    )
    model.train()
    distinct = list(dict.fromkeys(raw))
    if not distinct:
        return {"gen_distinct": 0, "gen_locatable_10mer": 0, "gen_frac_eligible": 0.0,
                "gen_mean_vh_identity": None, "gen_median_vh_identity": None}
    locs = [dv.locate_cdrh3_general(vh) for vh in distinct]
    ident = [dv.vh_identity_to_trastuzumab(
                 dv.host_clean_vh(vh, loc[1]) if loc is not None else vh, trast_vh)
             for vh, loc in zip(distinct, locs)]
    cores = [loc[2] for loc in locs
             if loc is not None and len(loc[2]) == data_mod.CDRH3_LENGTH]
    n_elig = 0
    if cores:
        rep = eligibility_report(cores, provided_matrix=provided_matrix, device=device)
        n_elig = int(rep["pass"].sum())
    return {
        "gen_distinct": len(distinct),
        "gen_locatable_10mer": len(cores),
        "gen_frac_eligible": round(n_elig / len(distinct), 4),
        "gen_mean_vh_identity": round(float(np.mean(ident)), 4),
        "gen_median_vh_identity": round(float(np.median(ident)), 4),
    }


def train(config: TrainConfig, *, diverse_path: str, provided_path: str,
          sample_n: int = 500, gen_temperature: float = 1.0, gen_top_p: float = 0.95) -> Dict:
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
        json.dump({**dataclasses.asdict(config), "path": "denovo-div",
                   "diverse_path": diverse_path, "sample_n": sample_n}, fh, indent=2)

    tokenizer = IgLMTokenizer()
    logger.line(f"[denovo-div] device={device} autocast={autocast_dtype} seed={config.seed}")

    splits = dv.load_diverse_splits(diverse_path)
    logger.line(f"diverse data: {json.dumps(splits.counts())} meta={json.dumps(splits.meta)}")
    trast_vh = splits.trast_vh
    provided_matrix = np.load(provided_path, allow_pickle=False)

    tr_tok, tr_len, tr_span, tr_cls = (splits.train_tokens, splits.train_length,
                                       splits.train_span, splits.train_classes)
    if config.loss_mode == "high-only":
        keep = tr_cls == data_mod.CLASS_TO_INDEX["high"]
        tr_tok, tr_len, tr_span, tr_cls = tr_tok[keep], tr_len[keep], tr_span[keep], tr_cls[keep]
        logger.line(f"loss_mode=high-only -> {int(keep.sum())} training rows")

    train_batcher = dv.PaddedFullGenBatcher(tr_tok, tr_len, tr_span, tr_cls,
                                            splits.pad_id, device)
    val_batcher = dv.PaddedFullGenBatcher(splits.val_tokens, splits.val_length,
                                          splits.val_span, splits.val_classes,
                                          splits.pad_id, device)

    # BASE IgLM is the starting point (diversity lives in the pretrained framework prior).
    model = load_model(config.model, tokenizer=tokenizer, device=device)
    model.train()
    logger.line(f"model={config.model} params={parameter_count(model):,} lmax={train_batcher.lmax}")

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
    base_gen = sample_diagnostic(model, tokenizer, device, trast_vh, provided_matrix,
                                 sample_n, gen_temperature, gen_top_p, config.seed)
    logger.line("baseline (base IgLM, full-seq): "
                + " ".join(f"{k}={v:.4f}" for k, v in baseline.items()))
    logger.line("baseline generation: " + json.dumps(base_gen))
    logger.metric({"step": 0, "phase": "baseline", **baseline, **base_gen})

    best_metric = baseline["val_nll_high"]
    best_step = 0
    stale = 0
    stop_reason = "completed"
    started = time.time()
    deadline = started + config.max_hours * 3600
    gen = torch.Generator(device="cpu").manual_seed(config.seed)

    from tqdm.auto import tqdm
    progress = tqdm(total=total_steps, desc="finetune-denovo-div", dynamic_ncols=True, smoothing=0.05)
    running: List[float] = []
    step = 0
    try:
        for epoch in range(config.epochs):
            order = torch.randperm(train_batcher.size, generator=gen)
            for bs in range(0, steps_per_epoch * config.batch_size, config.batch_size):
                index = order[bs:bs + config.batch_size].to(device)
                tokens, attn, length, span, classes = train_batcher.gather(index)
                lr = lr_at(step)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    nll, _ = dv.padded_both_nll(model, tokens, attn, length, span)
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
                    gen_m = sample_diagnostic(model, tokenizer, device, trast_vh, provided_matrix,
                                              sample_n, gen_temperature, gen_top_p,
                                              config.seed + step)
                    improved = metrics["val_nll_high"] < best_metric - 1e-5
                    logger.metric({"step": step, "phase": "val", "epoch": epoch,
                                   "elapsed_s": time.time() - started, "improved": improved,
                                   **metrics, **gen_m})
                    logger.line(
                        f"step {step}/{total_steps} ep{epoch} "
                        f"train={float(np.mean(running[-config.eval_every:])):.4f} "
                        f"val_high={metrics['val_nll_high']:.4f} val_low={metrics['val_nll_low']:.4f} "
                        f"sep={metrics['separation']:+.4f} "
                        f"gen_id={gen_m['gen_mean_vh_identity']} "
                        f"gen_elig={gen_m['gen_frac_eligible']} "
                        f"{'IMPROVED' if improved else f'no-improve x{stale + 1}'}")
                    meta = {"step": step, "epoch": epoch, "metrics": metrics, "generation": gen_m,
                            "baseline": baseline, "baseline_generation": base_gen,
                            "config": dataclasses.asdict(config), "path": "denovo-div",
                            "chain_token": "[HEAVY]", "species_token": "[HUMAN]",
                            "infill_range": list(data_mod.CDRH3_RANGE),
                            "diverse_meta": splits.meta, "trast_vh": trast_vh}
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
    summary = {"path": "denovo-div", "stop_reason": stop_reason, "steps": step,
               "total_steps_planned": total_steps, "elapsed_s": elapsed, "elapsed_h": elapsed / 3600,
               "best_val_nll_high": best_metric, "best_step": best_step,
               "baseline_val_nll_high": baseline["val_nll_high"],
               "improvement": baseline["val_nll_high"] - best_metric,
               "baseline_generation": base_gen,
               "best_checkpoint": os.path.join(config.run_dir, "best")}
    with open(os.path.join(config.run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.line("summary: " + json.dumps(summary))
    logger.close()
    return summary


def main(argv=None) -> int:
    parser = build_parser()
    for action in parser._actions:
        if action.dest == "run_dir":
            action.default = "runs/finetune_denovo_div"
    parser.add_argument("--diverse-path", default="data/processed/diverse_splits.npz")
    parser.add_argument("--provided-path", default="data/processed/provided_cdrh3_10mers.npy")
    parser.add_argument("--sample-n", type=int, default=500)
    parser.add_argument("--gen-temperature", type=float, default=1.0)
    parser.add_argument("--gen-top-p", type=float, default=0.95)
    args = parser.parse_args(argv)
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    config = TrainConfig(**{k: v for k, v in vars(args).items() if k in fields})
    summary = train(config, diverse_path=args.diverse_path, provided_path=args.provided_path,
                    sample_n=args.sample_n, gen_temperature=args.gen_temperature,
                    gen_top_p=args.gen_top_p)
    print(f"\n[denovo-div] best val_nll_high {summary['best_val_nll_high']:.4f} "
          f"(baseline {summary['baseline_val_nll_high']:.4f}, "
          f"improvement {summary['improvement']:+.4f}) at step {summary['best_step']} "
          f"-> {summary['best_checkpoint']}  stop={summary['stop_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
