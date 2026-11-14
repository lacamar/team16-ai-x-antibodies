#!/usr/bin/env bash
# De-novo HER2 VHH runbook: generate -> refold-prepare -> Boltz-2 co-fold -> refold-score -> rank.
# vm stages run on the H100 VM from ~/lachlan/rfd4-docs; rank runs on the laptop. NOT yet executed (GPU shared).
# Usage: bash run_denovo.sh {push|vm|rank}     SPEC=her2_vhh_fwfree (default) | her2_vhh   N=100 OUT=runs/denovo
set -euo pipefail

SPEC=${SPEC:-her2_vhh_fwfree}
N=${N:-100}
OUT=${OUT:-runs/denovo}
DATA=Data/13-her2-vhh
Q=quay.io/apheris/oslo-hackathon
RFD4=$Q:apheris-rfd4-proteina-v0.3.0-antibody-2026-10-30   # antibody weights, not the her2-* (trastuzumab-CDRH3) ones
PY=${PY:-/home/lm/.local/src/ai-antibodies/scribe/.venv/bin/python}
VM=hackathon@195.242.31.250

push() {  # laptop -> VM: spec + target + framework (one spec per dir; RFD4 reads every spec in --input)
  ssh -i ~/.ssh/team-16 $VM "mkdir -p ~/lachlan/rfd4-docs/$DATA/$SPEC"
  scp -i ~/.ssh/team-16 data/denovo/$SPEC.json data/denovo/her2_target_CD.cif data/denovo/vhh_framework_caplacizumab.cif \
    $VM:lachlan/rfd4-docs/$DATA/$SPEC/
}

vm() {  # ssh -i ~/.ssh/team-16 hackathon@195.242.31.250 ; cd ~/lachlan/rfd4-docs ; check nvidia-smi is free first
  RUN="docker run --rm --gpus all --shm-size 16g --user $(id -u):$(id -g) -e HOME=/tmp -v /etc/passwd:/etc/passwd:ro -v $PWD:/work -w /work"
  # 1. GENERATE: 3 CDRs on a VHH framework against HER2 domain IV (hotspots = trastuzumab epitope). ~24 s/design on A100.
  #    Smoke test first: N=1.  Pass rate upstream ~8-11% -> plan 100-200.
  $RUN $RFD4 module-run design --input $DATA/$SPEC --output $OUT/design \
    --num-replicates $N --seed 1 --recycle-strategy high/front_loaded
  # 2. REFOLD-PREPARE: wide, the ranker gates hard (identity, hallmarks, Cys).
  $RUN $RFD4 module-run refold-prepare --input $OUT/design --output $OUT/prepared --top-n 60
  # 3. CO-FOLD with Boltz-2 (VHH + HER2 C/D).
  $RUN $Q:apheris-boltz-v1.3.0-boltz2 python -m apheris_boltz.cli predict \
    --input $OUT/prepared/holo/queries.json --output $OUT/boltz2 \
    --template-paths $OUT/prepared/holo/template.cif \
    --write-detailed-scores true --num-diffusion-samples 1 --seeds 42 --query-failures skip
  # 4. SCORE -> $OUT/scored-boltz2/refold_metrics.csv. Then copy $OUT to the laptop.
  $RUN $RFD4 module-run refold-score --input $OUT/design --output $OUT/scored-boltz2 \
    --predictions $OUT/boltz2 --model boltz2
}

rank() {  # laptop, after copying $OUT. NOT the trastuzumab surrogate (see denovo_rank.py)
  $PY denovo_rank.py $OUT/scored-boltz2/refold_metrics.csv $OUT/ranked.csv
  "$PY" -c "import csv,sys; [print(r['id'],r['final_score'],r['vhh_sequence']) for r in csv.DictReader(open(sys.argv[1])) if r['selected']=='True']" $OUT/ranked.csv
}

case "${1:-}" in push) push ;; vm) vm ;; rank) rank ;; *) sed -n '2,4p' "$0"; exit 2 ;; esac
