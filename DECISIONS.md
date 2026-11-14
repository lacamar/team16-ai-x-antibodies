# DECISIONS

Thesis: fold-model confidence is not affinity. ipTM vs measured SPR affinity: Spearman 0.285 across a ~600,000x KD range. We rank on signals fit to measured data instead.

Owner column: human to initial.

## D1. Both tracks, lead with optimisation
- **Decision:** Submit both tracks; most of the 20 slots go to optimisation (CDRH3 variants of trastuzumab).
- **Rejected:** De novo only; optimisation only.
- **Why:** Scoring is affinity >= baseline, epitope preserved, thermostability. Optimisation starts from a known epitope binder, so two of three criteria are near-given. De novo is the high-variance bet and is kept as a minority slice.
- **Evidence:** Task rules (BUILD_SPEC); RFD4-Proteina HER2 weights `her2-2026-11-07` (opt) and `her2-novel-2026-11-09` (de novo, ~86% pass novelty, VM-RUN).
- **Owner:** ____

## D2. Rank on a measured-data surrogate, not ipTM
- **Decision:** Primary rank = `affinity_surrogate.py` (trained on provided SPR KD + FACS binding data).
- **Rejected:** Ranking on ipTM / min_ipAE / pLDDT.
- **Why:** ipTM is nearly blind to affinity (0.285). The surrogate recovers it (0.72).
- **Evidence:** ipTM vs SPR KD Spearman 0.285 (scorer calibration). Surrogate: 768 SPR KD + 525k FACS rows, 5-fold Spearman 0.72 on log-KD, high-binder AUC 0.90.
- **Owner:** ____

## D3. Epitope-preservation gate
- **Decision:** Designs are scored on how much of the trastuzumab-HER2 epitope they keep (19 epitope residues + 4 hotspots, from 1N8Z). Low-overlap designs are dropped.
- **Rejected:** Trusting high affinity alone.
- **Why:** Epitope preservation is judged by competitive binding. A tight binder at another site fails it.
- **Evidence:** `epitope_check.py`, reference complex PDB 1N8Z.
- **Owner:** ____

## D4. Novelty + developability hard filter before ranking
- **Decision:** Reject designs that duplicate known sequences or break basic developability rules before any scoring.
- **Rejected:** Scoring first, filtering after.
- **Why:** The track rules require novelty. Known binders would inflate scores with no new information. Unpaired Cys and length > 250 fail the hard filters.
- **Evidence:** `novelty_filter.py`, validated against all 525,063 provided sequences. Rule: CDRH3 differs >=2/10 from every provided sequence and >=3/10 from trastuzumab (`WGGDGFYAMD`); length <=250; no odd Cys count.
- **Owner:** ____

## D5. Trastuzumab baseline as control
- **Decision:** Run trastuzumab through every gate and show it in every table.
- **Rejected:** Reporting design scores alone.
- **Why:** Success is defined relative to baseline. A control shows the scorers behave and exposes any scorer that rates the baseline poorly.
- **Evidence:** BUILD_SPEC "Reproducibility + the baseline". [fill from final run: baseline surrogate score and rank]
- **Owner:** ____

## D6. Keep ipTM as a geometry gate only
- **Decision:** Boltz-2 co-fold ipTM/ipAE gates for sane interface geometry, never for ordering.
- **Rejected:** Dropping fold models entirely.
- **Why:** Weak for affinity, still useful to catch broken complexes and to check epitope contact.
- **Evidence:** Pipeline: RFD4-Proteina -> Boltz-2 co-fold -> score (VM-RUN, H100).
- **Owner:** ____

## Honest caveats
- Surrogate is a one-hot model on the 10-mer CDRH3. It cannot score changes outside CDRH3 and extrapolates poorly far from the training distribution.
- ipTM calibration rests on n=10 variants. The 0.285 is indicative, not tight; the 600,000x KD span is the point.
- ipTM is still used as a geometry gate (D6).
- Surrogate 5-fold CV measures interpolation within the provided data, not prospective SPR performance.
- Thermostability has no measured training data here. [state method used / fill from final run]
- Final numbers (counts per stage, top-20 scores) are not in this file: [fill from final run].
