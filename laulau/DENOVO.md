# De-novo path — full-antibody generation with IgLM

_Stage 2 run `runs/finetune_denovo`, eval `runs/denovo_gen`, 2026-11-14. Sibling of the
optimisation path (`train.py`/`generate.py`/the RL loop), which rewrites only the CDRH3._

## The three stages and what changed

| Stage | Optimisation path (existing) | De-novo path (this) |
|---|---|---|
| 1 — generate | `IgLM.infill` the 10-aa CDRH3 on trastuzumab's fixed framework | **`IgLM.generate` a whole heavy chain** (`her2_iglm/denovo.py`, `scripts/generate_denovo.py`) |
| 2 — fine-tune | class-weighted loss on the **CDRH3 span** only | **class-weighted loss on the full 123-token chain** (`train_denovo.py`), early-stopped on CDRH3-region `val_nll_high` |
| 3 — RL / score | RAFT loop: sample → eligibility → FACS P(high) → re-tune, KL-anchored | **unchanged** — same `eligibility`, `classifier`, `cofold`, `raft`; fed located CDRH3s |

## How IgLM incorporates the HER2 context

IgLM is **antigen-blind**. Its only conditioning inputs are two control tokens, `[HEAVY]`
and `[HUMAN]`; there is no channel for an antigen. HER2 is never shown to the model as
sequence or structure, so HER2 context **cannot enter through conditioning — only through
the weights**, by fine-tuning on sequences whose measured phenotype is HER2 binding.

Concretely:
- The corpus is the 524k trastuzumab VH variants, each FACS-labelled by measured HER2
  binding (low / medium / high).
- The class-weighted loss (high 1.0 / med 0.3 / low 0.05) makes the gradient pull
  probability mass toward the sub-distribution of sequences **measured to bind HER2
  strongly**. The model never "knows" it is HER2; it learns which heavy chains are the
  likely/strong ones, and because that label *was* HER2 binding, its generative prior
  becomes **implicitly HER2-conditioned**. HER2 lives in *which sequences now get high
  likelihood*, not in any input token.
- Proof the context was absorbed: held-out NLL of strong binders falls while weak-binder
  NLL lags, so **separation = `val_nll_low − val_nll_high`** grows. Baseline **+0.18 →
  +0.45**; strong-binder `val_nll_high` **3.46 → 1.48** (perplexity 31.8 → 4.4). That gap
  *is* the HER2 signal the model took on.

The de-novo difference: because the fine-tune biases the **whole-sequence** distribution
(not an infilled span), sampling full chains yields HER2-oriented antibodies de novo —
but, see below, the single-framework corpus means that bias arrives attached to
trastuzumab's framework.

## Why early stopping (stage 2)

The selection metric is held-out strong-binder CDRH3-region NLL (`val_nll_high`). Training
longer keeps lowering the *train* loss and eventually the *low-binder* val NLL too (the
model begins fitting weak binders and memorising the narrow single-framework corpus), at
which point `val_nll_high` plateaus and separation shrinks — overfitting that both loses
the "prefer strong binders" signal and **collapses generative diversity**. Patience-20
early stopping keeps the checkpoint at maximal binder preference, before that drift.

It is load-bearing **twice** on the de-novo path: the framework is memorised almost
immediately (`val_fullnll_high` → 0.12), so without early stopping generation collapses
hard onto trastuzumab's exact framework — which fails the `<70%`-identity de-novo novelty
filter. Run: `stop_reason=early_stopping`, best step 47,000 of 119,130 planned, 25.8 min.

## Evaluation (20,000 samples, T=1.0, top-p=0.95, `runs/finetune_denovo/best`)

| Metric | Value | Reading |
|---|---:|---|
| raw valid + terminating | 20,000 / 20,000 | generation is healthy; every sample emits `[CLS]` and a legal length |
| distinct VH | 19,101 | high diversity at the sequence level |
| CDRH3 locatable | 19,101 (100%) | **fine-tune worked** — pretrained IgLM framed 0/8 as trastuzumab; now all carry the framework |
| mean VH identity to trastuzumab | **94.3%** | **framework collapsed onto trastuzumab** (110 aa identical) |
| mean CDRH3 mismatches vs trastuzumab | 6.84 / 10 | the CDRH3 loop *is* well diversified |
| CDRH3 novel vs 524k screen | 8,336 (44%) | ~half the loops are unseen in the screen |
| eligible (abguard proxy, team's gate) | **20** | yield 0.1%; the `>2 muts from every one of 524k provided` rule is the bottleneck |
| top-16 eligible mean P(high) | 0.40 | predicted-affinity of the surviving designs |

## Conclusion (the honest finding)

Contextualising IgLM to HER2 can only draw on HER2-*binding* sequences, and the only such
data this hackathon is the **single-framework** trastuzumab screen. So the `.generate`
de-novo path faithfully reproduces that framework (94% identity) while diversifying the
CDRH3. **These are therefore optimisation-track designs reached through full-sequence
generation, not de-novo designs** (`<70%` identity to any known antibody) — HER2 context
and framework novelty are in direct tension given the available data.

Implications:
- For the **optimisation** track, the full-generate path is a valid, slightly more
  flexible alternative to CDRH3 infilling (it can also drift the framework a few residues).
- For a **genuine de-novo** track, IgLM needs HER2 binders spanning *diverse* frameworks
  (not available here) — or use a structure-based de-novo generator (RFdiffusion / RFD4-
  Proteina `her2-novel`) and keep IgLM for optimisation. The hackathon's own data note
  echoes this: `her2-novel` passes ~86% novelty where the FACS-tuned weights pass ~6%.

## Reproduce

```bash
# stage 2 (early-stopped, ~26 min on one H100)
env/run.sh scripts/finetune_denovo.py --run-dir runs/finetune_denovo \
  --batch-size 128 --learning-rate 5e-5 --warmup-steps 300 --epochs 30 \
  --max-hours 2.0 --eval-every 500 --log-every 25 --patience 20 \
  --loss-mode weighted --weight-low 0.05 --weight-medium 0.3 --weight-high 1.0
# stage 1 + eval
env/run.sh scripts/generate_denovo.py --ckpt runs/finetune_denovo/best -n 20000 \
  --temperature 1.0 --top-p 0.95 --out runs/denovo_gen
# stage 3 is unchanged: point scripts/rl_worker.py at runs/finetune_denovo/best
```
