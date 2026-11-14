# Trastuzumab → HER2, optimisation track

Team 16 · AI × Antibodies, Oslo · 2026-11-14

20 ranked scFv designs that redesign 4–5 of the 10 CDRH3 residues of trastuzumab
on an otherwise byte-identical framework.

- **Deliverables:** [`results/submission_top20.csv`](results/submission_top20.csv),
  [`results/submission_top20.fasta`](results/submission_top20.fasta),
  [`results/summary.html`](results/summary.html) (open in a browser)
- **Full decision log, every run id:** [`NOTES.md`](NOTES.md)
- **Rejection ledger (kept as rows with reasons, not dropped):**
  [`results/rejected_designs.csv`](results/rejected_designs.csv)

## The argument in short

Trastuzumab already binds HER2 at **K<sub>D</sub> 0.095 nM**, and the track rules forbid
staying near it: >2 point mutations from *every* one of the 525,063 provided antibodies.
Those two facts fight each other. RFD4 satisfies novelty by jumping 6–8 mutations away
from a clinically tuned loop, and **none** of its 73 novelty-passing designs was predicted
to bind as well (best ≈15 nM, ~150× worse — Finding F1).

So we stopped sampling. The provided library covers only 5.25×10⁵ of 10¹³ possible 10-mers,
so compliant sequences exist at just **4 mutations** from trastuzumab. The ridge affinity
model on one-hot CDRH3 is additive, so that shell (2,231,712 sequences) can be enumerated
**exactly** rather than sampled. We then required **independent methods to agree** before
promoting anything — because they frequently don't (F2, F7).

Lead design `TZ-OPT-01` = CDRH3 **`WGSQGFYEME`** (G101S, D102Q, A106E, D108E):
above the trastuzumab control on 4 of 5 independent axis-1 scorers, retains 4/4 epitope
hotspot contacts at 0.82 Jaccard overlap, and **deletes trastuzumab's own Asp102–Gly103
isomerisation liability** while being markedly less aggregation-prone.

## Reproduce

```bash
uv venv .venv && source .venv/bin/activate
uv pip install numpy pandas scikit-learn scipy gemmi

python verify_gates.py                              # re-derive the gates from raw data
python epitope.py runs/fold_shell results/epitope_shell.csv
python rank.py                                      # scoring + ranking
python make_submission.py                           # top-20 csv + fasta
python make_summary.py                              # results/summary.html
```

`verify_gates.py` needs the provided HER2 datasets under
`/home/hackathon/team16-ai-x-antibodies/data/her2/` (524,346-row FACS zip + 768-row SPR csv);
they are not vendored here, being the organisers' data. Everything else runs from committed files.

### Regenerating the structures (needs the H100 + Apheris images)

Exact commands, seeds and timings are in `NOTES.md`. In outline:

```bash
# generation — RFD4-Proteina, HER2-novel weights, no sequence_tsr_k
docker run --rm --gpus all --shm-size 16g --user $(id -u):$(id -g) \
  -e HOME=/tmp -v /etc/passwd:/etc/passwd:ro -v $PWD:/work -w /work \
  quay.io/apheris/oslo-hackathon:apheris-rfd4-proteina-v0.3.0-her2-novel-2026-11-09 \
  module-run design --input Data/12-her2/her2_cdrh3.yaml --output runs/prod100/design \
    --num-replicates 100 --seed 1 --recycle-strategy high/front_loaded --compile true

# folding — Boltz-2, templated on trastuzumab–HER2, baseline folded in the SAME batch
module-run predict --input runs/shell_queries.json --output runs/fold_shell \
  --template-paths runs/prod100/refold_prep/holo/template.cif \
  --seeds 1 --num-diffusion-samples 1
```

Lane B (Nebius, no GPU, concurrent with the above):

| run | op id | purpose |
|---|---|---|
| StaB-ddG | `op-79006e6f713984a7` | binding ddG, 42 variants |
| ESM-2 650M | `op-07b3495e1f4d104a` | VH pseudo-likelihood |
| AntiFold | `op-441067223204e140` | inverse-folding perplexity, 41 Fv |

## What each file does

| file | role |
|---|---|
| `verify_gates.py` | re-derives all submission gates from the raw provided data, independently of the code that proposed the candidates |
| `shell_search.py` | exact enumeration of the 4–5 mutation novelty-compliant shell |
| `train_binding.py`, `train_kd.py` | the FACS classifier and SPR K<sub>D</sub> regressor |
| `crosscheck.py` | cross-model agreement tests (the source of F2) |
| `epitope.py` | axis 2: contacted-HER2-residue overlap vs the in-run control |
| `developability.py` | axis 3: deterministic aggregation / liability / charge panel |
| `harvest_folds.py` | pulls Boltz-2 confidence metrics out of the fold outputs |
| `rank.py` | gates, the three axes, and the unanimity-based ranking |
| `make_submission.py`, `make_summary.py` | deliverables, rendered from the tables |

## Honest limitations

- **`abguard` does not exist on this VM** — not on PATH, not in pip, not on PyPI, not inside
  any Apheris image, absent from a recursive search of the hackathon bucket. `verify_gates.py`
  reimplements its four documented checks. The one check we **cannot** reproduce is immunum
  numbering confidence ≥0.1 (needs ANARCI/ANARCII; neither installs here, and AntiFold's
  bundled ANARCII could not number our Boltz-2 output chains). Unverified, and stated as such.
- **No thermostability predictor exists on either lane.** Aggrescan3D, SoluProt, NetSolP and
  ProteinMPNN-soluble are all absent; StaB-ddG is a *binding* ddG, not a folding one. Axis 3 is
  argued by construction (identical framework, AntiFold-confirmed flat framework perplexity)
  plus a deterministic loop panel — not predicted. AntiFold CDRH3 perplexity is slightly *worse*
  than trastuzumab's own loop for 37/40 designs; that is reported, not buried.
- **Axis 1 is convergent but not unanimous.** 9/40 designs beat the control on 4 of 5 scorers,
  none on 5/5, and those scorers partly anti-correlate with each other in this region (F7).
- Comparisons are only ever made inside one folding model. Credentials (`*.private.json`) are
  gitignored and never committed.
