# Team 16 — reproducible pipeline entry points.
# Deps come from requirements.txt via uv (no hand-managed venv).
py := "uv run --with-requirements requirements.txt python"

default:
    @just --list

# Run every gate's self-check (novelty, affinity surrogate, epitope, stability, ranker).
test:
    {{py}} novelty_filter.py
    {{py}} affinity_surrogate.py
    {{py}} epitope_check.py
    {{py}} thermostability.py
    {{py}} rank.py

# End-to-end runbook: {vm|local|all}. See run_pipeline.sh / docs/VM-RUN.md.
pipeline target="all":
    bash run_pipeline.sh {{target}}

# Boltz-2 refold-score output -> rank.py input.
candidates metrics out="candidates.csv":
    {{py}} make_candidates.py {{metrics}} {{out}}

# Rank candidates -> final top-20 with reasons.
rank candidates="candidates.csv" out="ranked.csv":
    {{py}} rank.py {{candidates}} {{out}}

# Full local scoring: candidates.csv -> ranked top-20.
rank-all metrics:
    just candidates {{metrics}} candidates.csv
    just rank candidates.csv ranked.csv
