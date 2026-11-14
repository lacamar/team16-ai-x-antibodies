# Pitch (5 slides)

## 1. Problem and who needs it
- Trastuzumab (HER2) is the baseline. Better binders mean lower dose, fewer resistant tumours, new formats (e.g. bispecific, ADC).
- Users: antibody engineers choosing what goes to the bench. Wet-lab slots are scarce (4 tested regardless, 20 ranked).
- Pain: computational tools rank designs by fold confidence. It looks rigorous. It is not affinity.
- We deliver a ranked shortlist with a written reason for every pick.

## 2. Thesis: rank on measured data, not ipTM
We measured it.

| Signal | Spearman vs real affinity |
|---|---|
| ipTM (fold confidence) | **0.285** across variants spanning a ~600,000x KD range |
| Our surrogate (5-fold, log-KD) | **0.72** (high-binder AUC 0.90) |

- Surrogate trained on the provided 768 SPR KD + 525k FACS rows.
- ipTM is nearly blind to affinity, so it is demoted to a geometry gate.

## 3. Pipeline funnel and three measured signals
RFD4-Proteina HER2 weights -> filter -> Boltz-2 co-fold -> score -> rank (H100).

1. **Novelty + developability filter** (`novelty_filter.py`): validated against all 525,063 provided sequences.
2. **Affinity surrogate** (`affinity_surrogate.py`): 0.72 / AUC 0.90.
3. **Epitope check** (`epitope_check.py`): fraction of the 1N8Z trastuzumab epitope (19 residues, 4 hotspots) preserved.

Plus ipTM as a geometry gate and the trastuzumab baseline run through every step as control.
Counts per stage: [fill from final run].

## 4. Results and how we pick the top 20
- Lead with optimisation (CDRH3 variants), reserve slots for de novo.
- Order: pass hard filter -> keep epitope -> surrogate score -> developability flags -> diversity across CDRH3 clusters.
- Each pick carries: surrogate score, epitope overlap, filter verdict, ipTM (geometry only), and the reason it beat the baseline.
- Top-20 table: [render from final run]. Baseline row: [fill from final run].
- Caveats up front: one-hot 10-mer CDRH3 surrogate; ipTM calibration n=10. See DECISIONS.md.

## 5. Path to value and reproducibility
- Path: SPR + competitive-binding results on the 4 tested designs feed back as new training rows, so the surrogate improves each round. Beyond CDRH3: extend to full-CDR and Fab-level models.
- Reproducible: one GitHub repo, pinned weights and seeds, cached model calls, every number traceable to a run id.
- One command: `just all` (planned target, [confirm exists before pitch]). Individual steps: `python3 novelty_filter.py`, `affinity_surrogate.py`, `epitope_check.py`.
- Judge-facing provenance: DECISIONS.md (decision / rejected / why / evidence).
