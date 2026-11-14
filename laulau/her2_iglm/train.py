"""Class-weighted infilling fine-tune of IgLM on trastuzumab CDRH3 FACS variants.

Objective
---------
Per example we compute the mean cross-entropy over the 11 loss targets (the 10 CDRH3
residues plus the terminating [CLS]). Examples are weighted by their measured HER2
binding class, so the model's probability mass moves toward CDRH3 loops that were
measured as strong binders:

    loss = sum_i w(class_i) * nll_i / sum_i w(class_i)

Default weights are high=1.0, medium=0.3, low=0.05. Low binders are kept at a small
positive weight rather than dropped or pushed to negative likelihood: they still carry
information about which loops are physically plausible, and negative-likelihood terms
make this objective unstable at these batch sizes.

What to watch
-------------
``val_nll_high`` is the model selection metric -- held-out negative log-likelihood of
measured strong binders, i.e. literally "how close is the model to the strong binders".
``separation`` (``val_nll_low - val_nll_high``) is the diagnostic: it should start near
zero for the pretrained model and grow as the model learns to prefer binders over
non-binders. A rising ``val_nll_high`` with still-falling train loss means overfitting,
which is what the early-stopping patience is for.
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
import torch.nn.functional as F
from tqdm.auto import tqdm

from her2_iglm import data as data_mod
from her2_iglm.model import load_model, parameter_count
from her2_iglm.tokenizer import IgLMTokenizer


@dataclasses.dataclass
class TrainConfig:
    csv_path: str = "data/raw/her2-trastuzumab/tz_her2_facs_3classes_524346.csv.zip"
    cache_path: str = "data/processed/facs_splits.npz"
    run_dir: str = "runs/finetune"
    model: str = "IgLM"

    batch_size: int = 128
    eval_batch_size: int = 512
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 300
    max_grad_norm: float = 1.0
    epochs: int = 12
    max_hours: float = 2.0

    # Class weights, indexed low / medium / high.
    weight_low: float = 0.05
    weight_medium: float = 0.3
    weight_high: float = 1.0
    #: "weighted" uses all rows with the weights above; "high-only" trains on
    #: measured strong binders alone.
    loss_mode: str = "weighted"

    log_every: int = 25
    eval_every: int = 500
    patience: int = 12
    seed: int = 0
    bf16: bool = True
    compile_model: bool = False

    def class_weights(self) -> List[float]:
        return [self.weight_low, self.weight_medium, self.weight_high]


class Batcher:
    """Builds fixed-length token batches by writing spans into the template row.

    Every example shares one constant 125-token template, so there is no padding and
    no attention mask. We keep the whole dataset resident on the GPU as uint8 (524k x
    10 bytes, ~5 MB) and gather batches there, which removes the dataloader from the
    critical path entirely.
    """

    def __init__(
        self,
        template: np.ndarray,
        spans: np.ndarray,
        classes: np.ndarray,
        device: torch.device,
    ) -> None:
        self.device = device
        self.template = torch.from_numpy(template.astype(np.int64)).to(device)
        self.spans = torch.from_numpy(spans.astype(np.int64)).to(device)
        self.classes = torch.from_numpy(classes.astype(np.int64)).to(device)
        self.size = self.spans.shape[0]

    def gather(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.template.unsqueeze(0).repeat(index.numel(), 1)
        tokens[:, data_mod.SPAN_START : data_mod.SPAN_END] = self.spans[index]
        return tokens, self.classes[index]


def example_nll(
    model: torch.nn.Module, tokens: torch.Tensor
) -> torch.Tensor:
    """Mean cross-entropy over the infilled span, per example.

    Mirrors the slice ``iglm.IgLM.log_likelihood`` scores for an infill range: logits
    from the [SEP] position onward predict the span residues and the final [CLS].
    """
    logits = model(tokens).logits
    shift_logits = logits[:, data_mod.SEP_POS : -1, :]
    shift_labels = tokens[:, data_mod.SEP_POS + 1 :]

    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(shift_labels.shape)

    return nll.mean(dim=1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    batcher: Batcher,
    config: TrainConfig,
    autocast_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    """Held-out NLL, reported overall and per binding class."""
    model.eval()
    totals = torch.zeros(len(data_mod.CLASS_NAMES), device=batcher.device)
    counts = torch.zeros(len(data_mod.CLASS_NAMES), device=batcher.device)

    for start in range(0, batcher.size, config.eval_batch_size):
        index = torch.arange(
            start,
            min(start + config.eval_batch_size, batcher.size),
            device=batcher.device,
        )
        tokens, classes = batcher.gather(index)
        with torch.autocast(
            "cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None
        ):
            nll = example_nll(model, tokens)
        totals.index_add_(0, classes, nll.float())
        counts.index_add_(0, classes, torch.ones_like(nll, dtype=torch.float))

    model.train()

    per_class = (totals / counts.clamp(min=1)).tolist()
    metrics = {
        f"val_nll_{name}": per_class[idx]
        for idx, name in enumerate(data_mod.CLASS_NAMES)
    }

    weights = torch.tensor(config.class_weights(), device=batcher.device)
    weighted = float((totals * weights).sum() / (counts * weights).sum().clamp(min=1))
    metrics["val_nll_weighted"] = weighted
    metrics["val_ppl_high"] = math.exp(min(metrics["val_nll_high"], 20.0))
    metrics["separation"] = metrics["val_nll_low"] - metrics["val_nll_high"]
    return metrics


class RunLogger:
    """Writes a JSONL metric stream plus a human-readable log, and mirrors to stdout."""

    def __init__(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.log_path = os.path.join(run_dir, "train.log")
        self._metrics = open(self.metrics_path, "a", buffering=1)
        self._log = open(self.log_path, "a", buffering=1)

    def metric(self, record: Dict) -> None:
        self._metrics.write(json.dumps(record) + "\n")

    def line(self, text: str) -> None:
        stamped = f"[{time.strftime('%H:%M:%S')}] {text}"
        self._log.write(stamped + "\n")
        tqdm.write(stamped)

    def close(self) -> None:
        self._metrics.close()
        self._log.close()


def save_checkpoint(model, tokenizer: IgLMTokenizer, path: str, meta: Dict) -> None:
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    # Keep vocab.txt beside the weights so the checkpoint is self-contained.
    with open(os.path.join(path, "vocab.txt"), "w") as handle:
        handle.write("\n".join(tokenizer.tokens) + "\n")
    with open(os.path.join(path, "finetune_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)


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
    with open(os.path.join(config.run_dir, "config.json"), "w") as handle:
        json.dump(dataclasses.asdict(config), handle, indent=2)

    tokenizer = IgLMTokenizer()
    logger.line(f"device={device} autocast={autocast_dtype} seed={config.seed}")

    splits = data_mod.prepare(config.csv_path, config.cache_path, tokenizer)
    logger.line(f"data counts: {json.dumps(splits.counts())}")
    if splits.dropped_rows:
        logger.line(f"dropped {splits.dropped_rows} rows failing geometry checks")

    train_spans, train_classes = splits.train_spans, splits.train_classes
    if config.loss_mode == "high-only":
        keep = train_classes == data_mod.CLASS_TO_INDEX["high"]
        train_spans, train_classes = train_spans[keep], train_classes[keep]
        logger.line(f"loss_mode=high-only -> {keep.sum()} training rows")

    train_batcher = Batcher(splits.template, train_spans, train_classes, device)
    val_batcher = Batcher(splits.template, splits.val_spans, splits.val_classes, device)

    model = load_model(config.model, tokenizer=tokenizer, device=device)
    model.train()
    logger.line(f"model={config.model} params={parameter_count(model):,}")

    if config.compile_model:
        model = torch.compile(model)

    weights = torch.tensor(config.class_weights(), device=device, dtype=torch.float)

    steps_per_epoch = max(1, train_batcher.size // config.batch_size)
    total_steps = steps_per_epoch * config.epochs

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    def lr_at(step: int) -> float:
        if step < config.warmup_steps:
            return config.learning_rate * step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, total_steps - config.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    baseline = evaluate(model, val_batcher, config, autocast_dtype)
    logger.line(
        "baseline (pretrained IgLM): "
        + " ".join(f"{k}={v:.4f}" for k, v in baseline.items())
    )
    logger.metric({"step": 0, "phase": "baseline", **baseline})

    best_metric = baseline["val_nll_high"]
    best_step = 0
    evals_without_improvement = 0
    stop_reason = "completed"
    started = time.time()
    deadline = started + config.max_hours * 3600

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    progress = tqdm(
        total=total_steps, desc="finetune", dynamic_ncols=True, smoothing=0.05
    )
    running: List[float] = []
    step = 0

    try:
        for epoch in range(config.epochs):
            order = torch.randperm(train_batcher.size, generator=generator)
            for batch_start in range(0, steps_per_epoch * config.batch_size, config.batch_size):
                index = order[batch_start : batch_start + config.batch_size].to(device)
                tokens, classes = train_batcher.gather(index)

                lr = lr_at(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                with torch.autocast(
                    "cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None
                ):
                    nll = example_nll(model, tokens)

                example_weights = weights[classes]
                loss = (nll * example_weights).sum() / example_weights.sum().clamp(
                    min=1e-6
                )

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                running.append(loss.item())
                progress.update(1)

                if step % config.log_every == 0:
                    mean_loss = float(np.mean(running[-config.log_every :]))
                    elapsed = time.time() - started
                    progress.set_postfix(
                        loss=f"{mean_loss:.4f}",
                        best=f"{best_metric:.4f}",
                        lr=f"{lr:.2e}",
                        refresh=False,
                    )
                    logger.metric(
                        {
                            "step": step,
                            "phase": "train",
                            "epoch": epoch,
                            "train_loss": mean_loss,
                            "lr": lr,
                            "grad_norm": float(grad_norm),
                            "elapsed_s": elapsed,
                            "examples_per_s": step * config.batch_size / elapsed,
                        }
                    )

                if step % config.eval_every == 0 or step == total_steps:
                    metrics = evaluate(model, val_batcher, config, autocast_dtype)
                    elapsed = time.time() - started
                    improved = metrics["val_nll_high"] < best_metric - 1e-5

                    logger.metric(
                        {
                            "step": step,
                            "phase": "val",
                            "epoch": epoch,
                            "elapsed_s": elapsed,
                            "improved": improved,
                            **metrics,
                        }
                    )
                    logger.line(
                        f"step {step}/{total_steps} ep{epoch} "
                        f"train={float(np.mean(running[-config.eval_every:])):.4f} "
                        f"val_high={metrics['val_nll_high']:.4f} "
                        f"val_low={metrics['val_nll_low']:.4f} "
                        f"sep={metrics['separation']:+.4f} "
                        f"ppl_high={metrics['val_ppl_high']:.3f} "
                        f"{'IMPROVED' if improved else f'no-improve x{evals_without_improvement + 1}'}"
                    )

                    if improved:
                        best_metric = metrics["val_nll_high"]
                        best_step = step
                        evals_without_improvement = 0
                        save_checkpoint(
                            model,
                            tokenizer,
                            os.path.join(config.run_dir, "best"),
                            {
                                "step": step,
                                "epoch": epoch,
                                "metrics": metrics,
                                "baseline": baseline,
                                "config": dataclasses.asdict(config),
                                "infill_range": list(data_mod.CDRH3_RANGE),
                                "framework_prefix": splits.framework_prefix,
                                "framework_suffix": splits.framework_suffix,
                                "vl_sequence": splits.vl_sequence,
                            },
                        )
                    else:
                        evals_without_improvement += 1
                        if evals_without_improvement >= config.patience:
                            stop_reason = "early_stopping"
                            raise StopIteration

                    save_checkpoint(
                        model,
                        tokenizer,
                        os.path.join(config.run_dir, "last"),
                        {"step": step, "epoch": epoch, "metrics": metrics},
                    )

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
    summary = {
        "stop_reason": stop_reason,
        "steps": step,
        "total_steps_planned": total_steps,
        "elapsed_s": elapsed,
        "elapsed_h": elapsed / 3600,
        "best_val_nll_high": best_metric,
        "best_step": best_step,
        "baseline_val_nll_high": baseline["val_nll_high"],
        "improvement": baseline["val_nll_high"] - best_metric,
        "best_checkpoint": os.path.join(config.run_dir, "best"),
    }
    with open(os.path.join(config.run_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    logger.line("summary: " + json.dumps(summary))
    logger.close()
    return summary
