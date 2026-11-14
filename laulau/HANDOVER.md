# HANDOVER — HER2 / trastuzumab CDRH3 optimisation on IgLM

_Last updated: 2026-11-14. Project root: `/home/hackathon/laulau` (a fork of Graylab/IgLM)._

This document lets a fresh session resume without re-deriving anything. Read it top to
bottom, then run the **Resume checklist** at the end.

---

## 1. Mission

Oslo **AI × Antibodies Hackathon** (pitch day 14 Nov 2026). We improve the clinical
antibody **trastuzumab** against antigen **HER2**, on the **optimisation track**.

A submitted design is judged on three experimentally measured properties vs. the
trastuzumab baseline: **HER2 affinity (SPR)**, **epitope preservation** (competitive
binding), **thermostability**. S-tier beats all three. Most S-tier designs wins.

**Hard filters** (`abguard` CLI, run by organisers — a failing design is not submitted):
- format VHH / scFv / Fab; total length ≤ 250 aa; no unpaired cysteines
- `immunum` numbering confidence ≥ 0.1
- optimisation track: **CDRH3 identity to trastuzumab < 70%** AND strictly **> 2 point
  mutations from any provided antibody**. Designs are never reclassified between tracks.

Consequence that drives the whole approach: optimisation still demands a substantially
rewritten CDRH3 (trastuzumab CDRH3 is 10 aa here, so ≥4 positions must change). A
point-mutation scan is ineligible → **CDRH3 infilling / redesign is the right tool**,
which is exactly what IgLM's `infill()` does.

Full task text extracted to `/tmp/.../scratchpad/out4.txt` (scratchpad is ephemeral);
source PDF is `ai-antibodies-hackathon-apheris.pdf` in the repo root.

---

## 2. The 3-step plan (user's own words)

1. **Set up the environment** to fine-tune the largest available IgLM model. ✅ DONE
2. **Fine-tune on the ~524k variants** to make the model agnostic to trastuzumab, with
   a loss that orients toward the **strong binders**. ✅ DONE (2026-11-14 12:17). Best
   checkpoint `runs/finetune/best/`: **val_nll_high 3.11 → 1.344** (ppl 22.5 → 3.83,
   improvement 1.77), separation 0.26 → 0.42. Early-stopped at step 45.5k (best at
   35.5k / epoch 8); 22.5 min wall-clock. Final checkpoint verified to generate 8/8
   valid terminating CDRH3 loops.
3. **RL loop**: use available scorers to generate new variants and further fine-tune the
   model with reinforcement learning toward optimal variants. ⬜ NOT STARTED.

User instructions for step 2 that must be honoured:
- Fine-tune for ~1–2 hours, with live logging and progress bars.
- Write all env/training files into the laulau folder.
- Write out model weights and keep track of the best loss.
- The loss should orient itself on being close to the strong binders.

---

## 3. What is in place (all validated on this H100 VM)

**Environment** — no venv/pip on this VM, so everything runs inside a pre-pulled CUDA
image via `env/run.sh` (image `quay.io/apheris/oslo-hackathon:apheris-esmfold2-v1.4.0`,
torch 2.15.0+cu132, transformers 5.22.2, numpy/pandas/tqdm). The project is bind-mounted
at `/work`; `/tmp` and the scratchpad are **not** visible inside the container, so keep
all scripts under the project. Usage: `env/run.sh scripts/<script>.py [args]`.

**Model** — `IgLM` (the full 12.9M-param GPT-2, 4 layers) is the larger of the two
bundled checkpoints (`IgLM` vs `IgLM-S`); there is no bigger IgLM. `her2_iglm/model.py`
loads it for training (upstream `iglm.IgLM` only exposes an eval-mode wrapper and a
`BertTokenizerFast` that is **broken under transformers 5.x**).

**Tokenizer** — `her2_iglm/tokenizer.py` re-implements IgLM's 33-token vocab from
`vocab.txt` directly, because the upstream BertTokenizerFast path yields vocab_size 5
and maps every residue to [UNK] under this stack. Use this tokenizer for everything,
including generation.

**Data** — `her2_iglm/data.py` turns the FACS table into fixed-length (125-token)
infilling examples: every row shares one constant template with 10 variable CDRH3
positions (`vh[98:108]`), so only the 10-residue spans + classes are cached. Cache at
`data/processed/facs_splits.npz` (already built): **508,348 train / 7,999 val / 7,999
test**, stratified by binding class, 0 dropped rows. Loss is computed on the 10 span
residues + the terminating [CLS] only — the exact slice `IgLM.log_likelihood` scores.

**Training** — `her2_iglm/train.py` + `scripts/finetune.py`. Class-weighted infilling
loss `sum_i w(class_i)·nll_i / sum_i w(class_i)`, weights low=0.05 / med=0.3 / high=1.0,
so probability mass moves toward measured strong binders. Model-selection metric is
`val_nll_high` (held-out NLL of strong binders = "how close to the strong binders").
`separation = val_nll_low − val_nll_high` is the diagnostic (should grow from ~0). Best
checkpoint saved to `runs/<run>/best`, last to `runs/<run>/last`; JSONL metrics +
human log + `summary.json`. Cosine LR with warmup, bf16, early stopping on patience,
hard `--max-hours` wall-clock stop.

### Smoke test result (validation, now deleted)
2.4 min / 2200 steps drove `val_nll_high` 3.11 → 1.46 (ppl 22.5 → ~4.3), separation
0.26 → ~0.49. The saved checkpoint reloaded and generated **8/8 valid 10-mer CDRH3
spans** that terminate on [CLS] and diverge from trastuzumab's `WGGDGFYAMD`
(`scripts/_reload_test.py <ckpt>` reproduces this — the generation path step 3 reuses).

### The step-2 run (COMPLETE)
Finished 2026-11-14 12:17. `stop_reason=early_stopping`, 45,500 steps in 0.375 h. Best
`val_nll_high=1.3438` at step 35,500 (epoch 8) vs baseline 3.1137 → **improvement 1.77**
(ppl 3.83). Held-out NLL by class at best: low 1.761 / med 1.518 / high 1.344, so the
model clearly prefers the strong binders (separation +0.42, up from the pretrained
+0.26). It early-stopped — not a time-out — because `val_nll_high` plateaued while
`val_nll_low` kept falling (the model beginning to fit weak binders); the best
checkpoint predates that drift, which is the intended behaviour. The finished model was
re-checked with `scripts/_reload_test.py runs/finetune/best`: 8/8 valid terminating
CDRH3 loops, all diverged from `WGGDGFYAMD`. Full metric stream in
`runs/finetune/metrics.jsonl`, human log in `runs/finetune/train.log`, stdout in
`runs/finetune_stdout.log`. Launch command was:
```
env/run.sh scripts/finetune.py --run-dir runs/finetune --model IgLM \
  --batch-size 128 --learning-rate 5e-5 --warmup-steps 300 --epochs 30 \
  --max-hours 2.0 --eval-every 500 --log-every 25 --patience 20 \
  --loss-mode weighted --weight-low 0.05 --weight-medium 0.3 --weight-high 1.0
```
30 epochs = 119,130 planned steps so the cosine LR could anneal across ~the full
budget, with a 2 h hard stop and early stopping (patience 20 = 20 evals × 500 steps) as
the real stopper — which is what fired, at 22.5 min. **Best weights:** `runs/finetune/best/`
(model.safetensors + vocab.txt + finetune_meta.json carrying the framework
prefix/suffix, VL, infill range and metrics). `runs/finetune/last/` is the final step.

If step 2 should be *rerun longer* (e.g. to honour a strict 1–2 h wall-clock), it would
need a change that keeps `val_nll_high` improving — lower peak LR, stronger weight decay
or dropout, or more aggressive class weighting — not merely a larger `--max-hours`;
as configured it has genuinely converged on the model-selection metric.

Note: one benign `UserWarning` ("tensor with requires_grad=True to a scalar") appeared
at launch from `float(loss)` in logging; fixed in-tree to `loss.item()` after launch
(the running process had already imported the module, so it is unaffected).

### Summary artifact (training curves + stage-3 plan)
A Claude artifact visualises the stage-2 training curves (val NLL by binding class +
binder separation, from `runs/finetune/metrics.jsonl`) and lays out the proposed stage-3
RL loop and its scorer plan:

**https://claude.ai/artifact/4hrCZa8nfNCt5gEQxd59XJ**

It is the reference for the step-3 design: the RL cycle (Sample → Filter → Score →
Update, KL-anchored to the stage-2 model) and the tiered scorer table (per-judged-property:
in-house FACS binder classifier for affinity on every sample; interface ΔΔG on a folded
VH:VL:HER2 complex, epitope-contact overlap vs trastuzumab reference `PDB 1N8Z` on HER2
domain IV, and StaB-ddG/Flex ddG/FoldX + ESM-2/CamSol for thermostability on the
shortlist; abguard filters as the eligibility gate; IgLM likelihood as a shaping term).
Caveat carried on the page: raw ipTM separates true from decoy binders poorly (Smorodina
et al., provided data) — use interface ΔΔG / ipSAE. The artifact is private to the owner
(laurenz.wicke@web.de); to update it from a new session, pass the URL as `url` to the
Artifact tool (a plain publish would create a separate copy). Source HTML was built in the
scratchpad (ephemeral) — regenerate from the metrics if the file is gone.

---

## 4. Resume checklist
1. `tail -n 40 runs/finetune/train.log` and `cat runs/finetune/summary.json` — is step 2
   done? Check `best_val_nll_high` vs `baseline_val_nll_high` (expect ≈1.4 vs 3.11) and
   `stop_reason`.
2. If still running, leave it; if crashed, re-launch with the command above.
3. Sanity-check the finished model: `env/run.sh scripts/_reload_test.py runs/finetune/best`.
4. Then start **step 3 (RL)** per the plan in the summary artifact (link in §3): generate
   CDRH3 infills from `runs/finetune/best`, score them with the tiered scorers, and
   fine-tune further toward optimal variants. Remember the **hard filters** in §1 (CDRH3
   identity to trastuzumab < 70%, > 2 point mutations, length ≤ 250, no unpaired
   cysteines, immunum confidence ≥ 0.1).