# BUILD SPEC — Team 16 HER2 binder pipeline

**Who reads this:** a Claude Code session building the repo on the day.
**What to do:** build a modular, reproducible pipeline that designs and ranks antibody variants that bind **HER2** better than **trastuzumab**, then stop and let the team tune the gates.

Mirrors our grant pipeline: a **declared graph of discrete gates**, each a self-contained, reproducible step with a fixed input→output contract. The things flowing through are **candidate antibody designs**.

---

## The task (fixed — confirm against the official 09:30 brief)

Improve **trastuzumab** (the anti-HER2 antibody). The target is given; there is no target-selection step. Two tracks (a candidate declares one):

- **Optimisation:** CDRH3 < 70% identity to trastuzumab, AND > 2 point mutations from any provided antibody.
- **De novo:** < 70% identity to any known antibody, AND ≥ 2 CDRH3 mutations from trastuzumab.

**Judged properties (each improved vs the trastuzumab baseline → tiers S/A/B by how many improve):**
1. **Affinity** — SPR KD ≥ baseline (tighter).
2. **Epitope preservation** — still binds the trastuzumab epitope (competitive binding).
3. **Thermostability.**

**Hard filters (`abguard` CLI, on the Apheris VM — a pass/fail gate):** must be VHH / scFv / Fab; no unpaired cysteine; numbering confidence ≥ 0.1; total length ≤ 250 aa.

**Submission:** a ranked list of **up to 20 designs** (16 + 4 spare). 4 are tested regardless of rank. Reference complex: **PDB 1N8Z** (trastuzumab Fab–HER2) for the epitope, baseline, and co-fold template.

**Pitch scoring:** Judgment 40% · Execution 20% (reproducible GitHub repo) · User fit 20% · Path to value 20%.

---

## Ground rules (contest hygiene)

- **Private repo.** Judges get access when asked.
- **Nothing before 09:00, 14 Nov 2026.** Every commit/author date ≥ 09:00. Author fresh; never commit a pre-existing file. Keys stay in gitignored `.keys/` — never committed.

## Core principle

Candidates flow through **gates**. Each gate reads candidates, adds scores + a pass/fail verdict, writes candidates out. Every gate is a **reproducible programmatic step** — deterministic, pinned model weights + seeds, cached I/O. **No LLM inside a gate.** The LLM proposes strategies, writes reasoning and the pitch — it never scores a candidate.

Three rules: **Reproducible** (`just all` reruns from a tagged commit, from cache); **Modular** (add an evaluation = one gate file + one line in the graph); **Legible** (`just graph` prints the whole pipeline + each gate's reads/writes).

---

## Two compute environments

| | Nebius Scientific AI | Apheris Foundry VM (H100) |
|---|---|---|
| Where | your laptop, `forge.py` (set up, verified) | per-team VM over SSH (IP + key given at the event) |
| Run via | `python3 forge.py run/submit <model>` | `foundry-toolkit` (Docker images) + `abguard` |
| Creds | `NEBIUS_CREDS` → `.keys/credentials.private.json` | VM-local read-only `s3://apheris-oslo-hackathon` |
| Strengths | fast iteration, 34 models, `stab-ddg` (ΔΔG), MSA, nanobodybuilder2, antifold | **HER2-tuned RFD4-Proteina**, OpenFold3/Protenix/Boltz-2 with iptm/min_ipae, `abguard` |
| Ephemeral? | hosted | **dies Sat night — copy results off** |

**Prize-relevant:** RFD4-Proteina HER2 weights on the VM — `her2-2026-11-07` (trastuzumab CDRH3 redesign; `--sequence-tsr-k 2` balanced, `1` exploratory) and `her2-novel-2026-11-09` (mostly-new CDRH3, ~86% pass novelty). A step-by-step HER2 script + `Tutorial.html` live in the bucket under `rfd4-proteina/docs/`. (Verify any "must use Apheris OpenFold" prize rule with the organizers.)

---

## The funnel (single — the target is fixed)

| Gate | Job | Tooling (pin weights + seed) |
|---|---|---|
| `g10_generate` | make candidate sequences for the chosen track | **opt:** RFD4-Proteina `her2-2026-11-07`, or `antifold` CDR redesign on trastuzumab · **de novo:** RFD4-Proteina `her2-novel-2026-11-09`, or rfdiffusion→proteinmpnn |
| `g20_abguard` | **hard filters** — VHH/scFv/Fab, no unpaired Cys, numbering ≥0.1, len ≤250 | `abguard` (VM). Cheap → runs early, before folding |
| `g25_novelty` | track identity constraints; **drop known binders** before ranking | deterministic sequence/CDRH3 identity vs trastuzumab + known-antibody set |
| `g30_fold` | co-fold binder + HER2 → complex + interface metrics | Apheris `predict` (OpenFold3/Boltz-2/Protenix → **iptm, min_ipae, binder_plddt**), or Nebius `boltz2`/`openfold3` |
| `g40_epitope` | binds the **trastuzumab epitope**? (preservation, judged) | overlap of contacts with 1N8Z paratope/epitope; interface quality |
| `g50_developability` | **thermostability** (judged) + manufacturability | `stab-ddg` (ΔΔG vs baseline), Aggrescan3D, SoluProt, NetSolP, ProteinMPNN soluble, liability-motif regex |
| `g60_select` | hard gates + composite rank + diversity → **≤20** (16+4) | reason per pick + per rejection |
| `g90_report` | render tables + provenance from the data | no hand-typed numbers |

Validation loop RFD4 recommends (inside g30/g40): **refold-prepare → fold with a Predict model → refold-score**, rank on `min_ipae`, `iptm`, `binder_plddt`. `boltz2` output has **no ipTM** — get interface metrics from Apheris `predict`, or count 1N8Z-epitope contacts yourself.

## The graph (`pipeline.py`)

```python
PIPELINE = [
    ("g10_generate",       []),
    ("g20_abguard",        ["g10_generate"]),      # hard filter first — cheap
    ("g25_novelty",        ["g20_abguard"]),
    ("g30_fold",           ["g25_novelty"]),
    ("g40_epitope",        ["g30_fold"]),
    ("g50_developability", ["g25_novelty"]),       # sequence/structure, parallel to fold
    ("g60_select",         ["g40_epitope", "g50_developability"]),
    ("g90_report",         ["g60_select"]),
]
```
`just graph` prints it and checks every field a gate READS is WRITTEN upstream.

## Gate contract

Each gate is a module in `gates/` exposing `GATE`, `READS`, `WRITES`, and `run(run, cfg)` — a deterministic function of (candidates, config). Wrap every `forge.py`/VM/model call in `core.disk_cache` (keyed on input hash) so a rerun is offline and identical. A rejecting gate writes a verdict row `{candidate_id, gate, passed:false, reason}` and drops the candidate — the verdict survives (negative results are evidence). Provenance on every number: `scorer`, `model_id`, `weights`, `seed`, `run_id`, `git_sha`, `config_hash`.

## Reproducibility + the baseline

- **Score the trastuzumab baseline through the same gates** — it is the control every design must beat (affinity ≥ baseline, epitope preserved, thermostability ↑). Put it in the tables.
- Pin RFD4 weight tags, model ids, and seeds explicitly — never "latest". Keep `nebius-runs/` (run ids) and `results/` so every number traces to a run.
- Confidence (iptm/plddt) is **not** affinity; use it to rank, add the baseline + a decoy as controls.

---

## Day-0 steps

1. Fresh private repo (`gh repo create … --private`), add collaborators from `CONTRIBUTORS.md`.
2. `export NEBIUS_CREDS=…/.keys/credentials.private.json`; `forge.py models` to confirm. Get the **Apheris VM SSH** and `foundry-toolkit module list` + `abguard` working; `aws s3 sync s3://apheris-oslo-hackathon/rfd4-proteina ./data` for the HER2 tutorial/script.
3. Pick the track (config `track`). Pull trastuzumab sequence + **1N8Z**.
4. Wire gates one at a time, each proven on ~20 dummy candidates (`just all`) before the next. Walking skeleton first; real tools second.
5. Commit after each working gate; tag the final run.

## Definition of done (skeleton, before real tools)
`just all` runs g10→g90 on dummy candidates, offline, producing a ranked ≤20 table with provenance + the baseline row, and `just graph` validates. Then swap dummy gates for the real HER2 tools.
