# VM run cheat-sheet — Team 16

The H100 VM runs the **Apheris Foundry** models as local Docker images (no Foundry Hub, no login). The Nebius models run from your laptop via `forge.py`.

## Connect
```bash
ssh -i ~/.ssh/team-16 hackathon@195.242.31.250     # shared user, one GPU — work in ~/<yourname>/
```
Working dir with the HER2 materials + data: `~/lachlan/rfd4-docs` (docs, `Data/12-her2/`, `Scripts/`, `Settings/`) and `~/lachlan/data/her2-trastuzumab/`. VM dies Saturday night — copy results off.

## The Docker run wrapper (every Foundry model)
```bash
Q=quay.io/apheris/oslo-hackathon
RUN="docker run --rm --gpus all --shm-size 16g --user $(id -u):$(id -g) \
     -e HOME=/tmp -v /etc/passwd:/etc/passwd:ro -v $PWD:/work -w /work"
```
Entry commands (if you bypass the wrappers): OpenFold3 `python -m apheris_openfold3.cli`, Boltz-2 `python -m apheris_boltz.cli`.

## Whole HER2 pipeline (easiest)
```bash
cd ~/lachlan/rfd4-docs
N=4 TOPN=2 IMG=her2-2026-11-07 bash Scripts/run-her2.sh all   # design→prepare→fold(boltz2)→score, ~4 min
# IMG=her2-novel-2026-11-09 for the de-novo track (default). OUT=runs/her2.
```

## Or step by step
```bash
# 1. GENERATE (CDRH3 variants on the trastuzumab framework, target = 1N8Z HER2 domain IV)
$RUN $Q:apheris-rfd4-proteina-v0.3.0-her2-2026-11-07 module-run design \
  --input Data/12-her2/her2_cdrh3.yaml --output runs/her2/design \
  --num-replicates 100 --seed 1 --recycle-strategy high/front_loaded --sequence-tsr-k 2
# 2. REFOLD-PREPARE top N  → runs/her2/prepared/holo/{queries.json,template.cif}
$RUN $Q:apheris-rfd4-proteina-v0.3.0-her2-2026-11-07 module-run refold-prepare \
  --input runs/her2/design --output runs/her2/prepared --top-n 20
# 3. FOLD the complex (Boltz-2 shown; OpenFold3 = image -of3ob / -of3p2)
$RUN $Q:apheris-boltz-v1.3.0-boltz2 python -m apheris_boltz.cli predict \
  --input runs/her2/prepared/holo/queries.json --output runs/her2/boltz2 \
  --template-paths runs/her2/prepared/holo/template.cif \
  --write-detailed-scores true --num-diffusion-samples 1 --seeds 42 --query-failures skip
# 4. SCORE  → runs/her2/scored-boltz2/refold_metrics.csv
$RUN $Q:apheris-rfd4-proteina-v0.3.0-her2-2026-11-07 module-run refold-score \
  --input runs/her2/design --output runs/her2/scored-boltz2 \
  --predictions runs/her2/boltz2 --model boltz2
```

## RFD4 HER2 weights (pick the regime)
| regime | image tag | flag | note |
|---|---|---|---|
| balanced (optimisation) | `her2-2026-11-07` | `--sequence-tsr-k 2` | many distinct loops, ~6% pass novelty |
| exploratory (optimisation) | `her2-2026-11-07` | `--sequence-tsr-k 1` | 100% distinct, ~74% pass novelty |
| de novo | `her2-novel-2026-11-09` | (none) | ~90% distinct, ~86% pass, lowest interface error |

`refold-score --model` also accepts: `openfold3-of3ob`, `openfold3-of3p2`, `protenix-v2`, `protenix-v1`, `mol1-base`, `boltz1`, `esmfold2-release`. Drop `--template-paths` for non-template models.

## Rank + filter
- **Rank** on `min_ipae` (lower better), then `iptm`, `binder_plddt`. Compare only *within one folding model*. The HER2 example leaves `pass` empty (relative coords) — rank, don't gate on it.
- **Novelty + hard filter** (the `abguard` replacement) is `novelty_filter.py` in the repo root: CDRH3 must differ ≥2/10 from every provided sequence and ≥3/10 from trastuzumab `WGGDGFYAMD`; plus length ≤250 and no odd Cys count. `python3 novelty_filter.py` runs its self-check against `data/her2/`. **Filter before ranking** — known binders fail.

## Nebius (laptop)
```bash
export NEBIUS_CREDS=~/.local/src/team16-ai-x-antibodies/.keys/credentials.private.json  # already in ~/.bashrc
F=~/.claude/skills/nebius-scientific-ai/forge.py
python3 "$F" models                 # 34 models
python3 "$F" run boltz2 --args-file boltz.json --label fold-1   # inline models
python3 "$F" submit <model> --entry '...' --label x             # file models (antifold, stab-ddg, …)
python3 "$F" runs | wait <op> | fetch <op> --dest results/x
```
Useful Nebius models: `stab-ddg` (binding ΔΔG), `antifold` (CDR redesign), `nanobodybuilder2`, `proteinmpnn`, `msa-search-pdb70`.

## Submission
Ranked list of up to **20** designs (16 + 4 spare); 4 tested regardless. Two tracks (optimisation / de novo) = two separate prizes. Judged: SPR affinity ≥ baseline, epitope preservation (competitive), thermostability → S/A/B tiers.
