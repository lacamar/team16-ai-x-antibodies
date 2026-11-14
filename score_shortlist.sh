#!/usr/bin/env bash
# IgLM top-K -> fold on VM (Boltz-2) -> epitope + rank. Usage: bash score_shortlist.sh topk.csv [outdir]
# topk.csv: cdrh3,vh_sequence[,reward]. Output: $OUT/ranked.csv (final_score, selected, reason). ~75 s/design: keep K <= ~30.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/lm/.local/src/ai-antibodies/scribe/.venv/bin/python}
TOPK=$1; OUT=${2:-runs/shortlist}
VM=hackathon@195.242.31.250; SSH="ssh -i $HOME/.ssh/team-16"; W=lachlan/rfd4-docs   # relative to VM $HOME
TPL=runs/opt/prepared/holo/template.cif                                         # existing HER2 template on the VM
RUN="docker run --rm --gpus all --shm-size 16g --user \$(id -u):\$(id -g) -e HOME=/tmp -v /etc/passwd:/etc/passwd:ro -v \$PWD:/work -w /work"
R=runs/iglm_$(basename "$OUT")_$(date +%s)

$PY shortlist.py prep "$TOPK" "$OUT"
$SSH $VM "mkdir -p $W/$R"
scp -q -i ~/.ssh/team-16 "$OUT/queries.json" $VM:$W/$R/queries.json
$SSH $VM "cd $W && $RUN quay.io/apheris/oslo-hackathon:apheris-boltz-v1.3.0-boltz2 python -m apheris_boltz.cli predict \
  --input $R/queries.json --output $R/boltz2 --template-paths $TPL \
  --write-detailed-scores true --num-diffusion-samples 1 --seeds 42 --query-failures skip" >"$OUT/fold.log" 2>&1
scp -q -r -i ~/.ssh/team-16 $VM:$W/$R/boltz2 "$OUT/boltz2"
$PY shortlist.py metrics "$OUT/boltz2" "$OUT/refold_metrics.csv" "$OUT/queries.json"
$PY make_candidates.py "$OUT/refold_metrics.csv" "$OUT/candidates.csv"
$PY rank.py "$OUT/candidates.csv" "$OUT/ranked.csv"
