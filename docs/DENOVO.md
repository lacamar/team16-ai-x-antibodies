# De-novo VHH track (HER2)

Status: set up, **not yet run** (GPU busy with the optimisation run). Inputs and spec syntax were taken from the
RFD4 VHH example (caplacizumab framework); the real contig/condition validity is untested until a first N=1 run.

## Strategy
- Format: VHH (one chain, 3 CDRs). RFD4-Proteina `antibody-2026-10-30` weights, framework backbone fixed, CDR1/2/3 generated.
- Target: 1N8Z HER2 domain IV (chains C 489-580, D 591-607), hotspots C572, C573, D591, D593 = trastuzumab epitope, so binding is competitive.
- Novelty: <70% identity to ANY known antibody. A fixed caplacizumab framework alone is ~73% identical with short CDRs, so the
  default spec `her2_vhh_fwfree` fixes only Cys22, FR2 (hallmarks), Cys75 and FR4 WGQG; the rest of the framework sequence is designed.
  `her2_vhh` keeps the full framework (fallback, likely fails the identity gate).
- CDR lengths: 10-13 / 9-10 / 12-21 (from the example's placeholder labels 13 / 9-10 / 12-21).

## Generation command
```
docker run --rm --gpus all --shm-size 16g --user $(id -u):$(id -g) -e HOME=/tmp -v /etc/passwd:/etc/passwd:ro -v $PWD:/work -w /work \
  quay.io/apheris/oslo-hackathon:apheris-rfd4-proteina-v0.3.0-antibody-2026-10-30 module-run design \
  --input Data/13-her2-vhh/her2_vhh_fwfree --output runs/denovo/design \
  --num-replicates 100 --seed 1 --recycle-strategy high/front_loaded
```
Full chain: `bash run_denovo.sh push`, `bash run_denovo.sh vm` (on the VM; try `N=1` first), `bash run_denovo.sh rank`.

## Versus the optimisation track
| | optimisation | de novo |
|---|---|---|
| scaffold | trastuzumab Fv, CDRH3 only | VHH framework, 3 CDRs |
| weights | `her2-*` | `antibody-2026-10-30` |
| novelty | CDRH3 vs provided sets | <70% identity to any known antibody |
| ranker | `rank.py`, surrogate primary | `denovo_rank.py`, no surrogate |

## Ranking (`denovo_rank.py`)
The trastuzumab surrogate is trained on trastuzumab CDRH3 variants only and does not transfer to a new framework, so it is not used.
- Hard gates: CDR3 novelty (`novelty_filter.is_novel`), `hard_checks` (length, no unpaired Cys), global identity <0.70 to every
  reference (template framework, trastuzumab VH, optional `data/denovo/known_antibodies.fasta`), >=2/4 FR2 VHH hallmarks (37/44/45/47), ipTM >= 0.6.
- Score: ipTM - 0.05 * min_ipAE + 1.5 * epitope recall vs 1N8Z + 0.25 per hotspot hit (-1 if recall <0.3) + CDR developability
  (`thermostability.stability_score`) - liabilities (N-glyc, NG/DG/DP) - hallmark and identity shortfall.
- Dedup on CDR3 (Hamming >=3), top 20 (16 + 4 spare). Compare only within one folding model.

## Caveats
- Identity references are only the template and trastuzumab; load real SAbDab/INDI/OAS sequences into `known_antibodies.fasta`.
- Hallmark check is a motif heuristic; TNP is the real profiler.
- Epitope check needs HER2 numbering 489-607 (RFD4 `*_generated.cif`); Boltz-renumbered CIFs would mis-score.
- `module-run design --dry-run` only validates config, not the spec.
