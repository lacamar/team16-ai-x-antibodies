#!/usr/bin/env bash
# End-to-end HER2 runbook: generate -> filter -> fold -> score -> epitope -> surrogate -> rank -> top-20.
# Stages 1-4 run on the H100 VM (docs/VM-RUN.md); 5-7 run on the laptop in the scribe venv.
# Usage: bash run_pipeline.sh {vm|local|all}
set -euo pipefail

PY=${PY:-/home/lm/.local/src/ai-antibodies/scribe/.venv/bin/python}
OUT=${OUT:-runs/her2}
IMG=${IMG:-her2-2026-11-07}        # her2-novel-2026-11-09 for the de-novo track
Q=quay.io/apheris/oslo-hackathon
RUN="docker run --rm --gpus all --shm-size 16g --user $(id -u):$(id -g) \
     -e HOME=/tmp -v /etc/passwd:/etc/passwd:ro -v $PWD:/work -w /work"
RFD4=$Q:apheris-rfd4-proteina-v0.3.0-$IMG

vm() {  # ssh -i ~/.ssh/team-16 hackathon@195.242.31.250 ; cd ~/lachlan/rfd4-docs
  # 1. GENERATE CDRH3 variants on the trastuzumab framework (target 1N8Z domain IV).
  #    Regime: --sequence-tsr-k 2 balanced, 1 exploratory; de novo = novel image, no flag.
  $RUN $RFD4 module-run design --input Data/12-her2/her2_cdrh3.yaml --output $OUT/design \
    --num-replicates 100 --seed 1 --recycle-strategy high/front_loaded --sequence-tsr-k 2

  # 2. NOVELTY + HARD FILTER (abguard replacement) on generated CDRH3s — known binders fail.
  #    novelty_filter.py's is_novel/hard_checks are also re-applied inside rank.py (hard gates).

  # 3. REFOLD-PREPARE: wider than 20 so the re-ranker has room to drop designs.
  $RUN $RFD4 module-run refold-prepare --input $OUT/design --output $OUT/prepared --top-n 60

  # 4. BOLTZ-2 co-fold design + HER2 (OpenFold3 alternative: images -of3ob / -of3p2).
  $RUN $Q:apheris-boltz-v1.3.0-boltz2 python -m apheris_boltz.cli predict \
    --input $OUT/prepared/holo/queries.json --output $OUT/boltz2 \
    --template-paths $OUT/prepared/holo/template.cif \
    --write-detailed-scores true --num-diffusion-samples 1 --seeds 42 --query-failures skip

  # 5. REFOLD-SCORE -> $OUT/scored-boltz2/refold_metrics.csv (ipTM, min_ipae, binder_plddt).
  #    Compare only within one folding model. ipTM is a geometry gate only (Spearman 0.285 vs affinity).
  $RUN $RFD4 module-run refold-score --input $OUT/design --output $OUT/scored-boltz2 \
    --predictions $OUT/boltz2 --model boltz2
  # Then copy $OUT (scored CSV + boltz2 predicted CIFs) to the laptop; VM dies Saturday night.
}

local_() {
  # Build candidates.csv (cdrh3, vh_sequence, iptm, min_ipae, design_cif) by joining
  # refold_metrics.csv with the design sequences and the predicted CIFs (join script is per-run).
  CAND=$OUT/candidates.csv

  # 6a. EPITOPE: recall vs trastuzumab's 1N8Z epitope + hotspots C572/C573/D591/D593 (per design).
  #     $PY epitope_check.py path/to/design.cif       (rank.py calls preservation() itself)
  # 6b. AFFINITY SURROGATE (primary ranker, Spearman 0.72): AffinitySurrogate().fit().score(cdrh3)
  #     (rank.py calls it itself)

  # 7. RANK: hard gates (novelty, hard_checks, ipTM>=0.6) -> surrogate + epitope bonus -> dedup -> top-20.
  #    Weights are constants at the top of rank.py. Output has selected/final_score/reason per candidate.
  $PY rank.py "$CAND" "$OUT/ranked.csv"

  # 8. SUBMIT: the selected=True rows of $OUT/ranked.csv (max 20: 16 + 4 spare, per track).
  "$PY" -c "import csv,sys; [print(r[\"cdrh3\"],r[\"final_score\"]) for r in csv.DictReader(open(sys.argv[1])) if r[\"selected\"]==\"True\"]" "$OUT/ranked.csv"
}

case "${1:-all}" in vm) vm ;; local) local_ ;; all) vm; local_ ;; *) echo "usage: $0 {vm|local|all}"; exit 2 ;; esac
