# Stage 3 — RL loop runbook (HER2 / trastuzumab CDRH3 optimisation)

How to **continue, watch, and harvest** the stage-3 reinforcement-learning run. Anyone can
pick this up from a fresh shell. See `HANDOVER.md` for stages 1–2 (environment + the
supervised fine-tune that produced the starting policy `runs/finetune/best`).

---

## TL;DR

```bash
cd /home/hackathon/laulau

# continue the run (auto-resumes from the latest cycle's policy; 5 more cycles)
bash scripts/continue_run.sh

# watch it live (per-fold notifications + streamed training metrics)
tail -f runs/rl/live_feed.log

# once some cycles are done, build a ranked shortlist from ALL co-folded candidates
env/run.sh scripts/build_top.py --n 50     # -> runs/rl/top50.csv + runs/rl/top50.fasta
```

Tune scale with env vars: `ADD_CYCLES` (default 5), `N_PROPOSE` (80000), `TEMP` (1.6),
`N_COFOLD` (24), `N_ELITE` (2000), `RAFT_EPOCHS` (4), `RAFT_LR` (2e-5).

---

## What one cycle does (the filter cascade)

1. **Propose** — sample `N_PROPOSE` CDRH3 10-mers from the current IgLM policy; dedup and
   drop anything in the 524k screen or seen in an earlier cycle → distinct novel proposals.
2. **Eligibility gate** (`her2_iglm/eligibility.py`, exact Python — abguard/immunum are
   *not* installed here, the organisers run them): CDRH3 identity to trastuzumab < 70%
   (≥4/10 differ), CDRH3 Hamming ≥ 3 to **every** provided antibody (GPU min-Hamming over
   all 525k provided 10-mers), no Cys in the loop (no new unpaired cysteine), length ≤ 250.
3. **FACS classifier** (`her2_iglm/classifier.py`, trained on the 524k screen, val AUC≈0.99)
   → P(high binder). The dense RL reward.
4. **Developability** (`her2_iglm/developability.py`) — standard CDR sequence-liability
   motifs (N-glyc sequon, deamidation/isomerisation, oxidation, low-complexity).
5. **Co-fold the top `N_COFOLD`** with **local Boltz-2** + the **1N8Z** template
   (`her2_iglm/cofold.py`): holo (VL+VH+HER2 domain IV) and apo (VL+VH alone). Metrics:
   `iptm`, `min_ipae_her2`, `ab_ag_iptm`, `binder_plddt`, epitope contacts; apo `plddt`/`ptm`.
6. **Score top-10** on the three judged properties vs the in-batch trastuzumab baseline:
   affinity (iptm / interface-PAE), **epitope preservation** (CDRH3–HER2 contact overlap,
   Jaccard, on 1N8Z domain IV), **thermostability** (apo-fold pLDDT **proxy** — no ddG/Tm
   tool exists on this VM; treat as a proxy, not a measured Tm).
7. **RL update** — RAFT / expert iteration (`her2_iglm/raft.py`): fine-tune the policy on
   the cycle's eligible high-reward elites (co-fold-validated ones up-weighted ×3),
   KL-anchored by small LR + few epochs. The result is the next cycle's policy.

Every co-folded candidate (all cycles) is appended to `runs/rl/all_candidates.csv`.

---

## Live observation

`scripts/rl_loop.sh` streams everything to `runs/rl/live_feed.log`. During co-folding,
`scripts/rl_watch.py` prints one line per fold as Boltz writes it (~7 s apart), e.g.:

```
[C6 HOLO] baseline trastuzumab WGGDGFYAMD  iptm=0.960 ipae_her2=0.63 binder_plddt=98.4 epitope=7 contacts
[C6 HOLO] cand_003 NAAARIWPFE  iptm=0.945(Δ-0.015) ipae_her2=0.76(Δtighten-0.13) epitope_overlap=0.88 | affinity↑:no epitope✓:YES
[C6 APO ] cand_003 NAAARIWPFE  apo_plddt=97.3(Δ+1.2) apo_ptm=0.90 | thermostability↑:YES
```

RAFT training metrics (per-epoch loss, val NLL before/after) stream from the `score` stage.

---

## File layout (`runs/rl/`)

```
runs/rl/
  live_feed.log            # the streamed console feed (tail -f this)
  loop.log                 # timestamped stage log
  all_candidates.csv       # EVERY co-folded candidate, all cycles (for build_top.py)
  seen.json                # all proposed CDRH3s across cycles (dedup/exclusion)
  top50.csv / top50.fasta  # produced by build_top.py
  cycleN/
    propose_report.json    # counts: generated / eligible / dev-pass / mean P(high)
    pool.json              # ranked eligible pool (CDRH3 + P(high) + dev penalty)
    cofold_items.json      # trastuzumab + the N_COFOLD candidates folded this cycle
    holo_queries.json / apo_queries.json
    holo/  apo/            # Boltz-2 outputs (per-complex scores.json, model.cif, cbor)
    score_report.json      # top-10 with 3-property proxy flags + baseline + raft meta
    policy/                # the fine-tuned checkpoint this cycle produced (next policy)
```

## Specifications

- **Model / policy**: IgLM GPT-2 (12.9M params), infilling the 10-residue CDRH3 at VH
  positions 99–108 in the trastuzumab framework. Start policy `runs/finetune/best`; each
  cycle `runs/rl/cycleN/policy` feeds the next. Generation path: `her2_iglm/generate.py`.
- **Execution**: no venv/pip on this VM. Project Python runs in the ESMFold2 Foundry image
  via `env/run.sh`. Co-folding runs the `apheris-boltz-v1.3.0-boltz2` image directly with
  `docker run` (the driver handles this). Co-fold seed is fixed at 42.
- **Data**: `data/processed/facs_splits.npz` (splits), `data/processed/db_cdrh3_set.txt`
  (524k exclusion set), `data/processed/provided_cdrh3_10mers.npy` (525k novelty set) —
  rebuild with `env/run.sh scripts/build_provided.py`. Classifier `runs/classifier/`
  (retrain: `env/run.sh scripts/train_classifier.py`).
- **Template**: `her2_iglm/assets/her2_fv_1n8z_template.cif` + `her2_chains.json` (HER2
  domain IV, chains C 489–580 / D 591–607, from PDB 1N8Z).

## Resuming / safety

- Resumable: each stage is skipped if its output exists; a crashed cycle re-runs only its
  unfinished stages. Delete a `cycleN/` subfolder's stage output to force a redo.
- `continue_run.sh` refuses to start a second loop if one is already running.
- The VM is shut down at the end of the hackathon — copy `runs/rl/` off the machine to keep it.

## Caveats (be honest in the pitch)

- **Thermostability is an apo-fold confidence proxy**, not a measured Tm — there is no
  FoldX/StaB-ddG on this VM and the Nebius gateway is not wired into this session.
- **Affinity/epitope come from co-folding confidence** (iptm, interface-PAE, contact
  overlap), compared only within one predictor (Boltz-2) and against the in-batch
  trastuzumab baseline, as the HER2 example advises. Not an experimental SPR/competition.
- Final eligibility is decided by the organisers' `abguard`; our gate is a faithful,
  conservative re-implementation of the rules that bite a CDRH3-only redesign.
