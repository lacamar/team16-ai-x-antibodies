# HANDOFF — read this first, then `NOTES.md`

Written 2026-11-14 11:52 UTC. Team 16, AI × Antibodies Oslo, **optimisation track**.
Repo: `~/ubaydolla/her2` (git, local only, HEAD `5760f9e`). `NOTES.md` is the full running log
with every finding (F1–F12) and decision (D1–D6); this file is the 2-minute version.

```bash
cd ~/ubaydolla/her2
export NEBIUS_CREDS=~/ubaydolla/nebius-kit/credentials.private.json
PY=.venv/bin/python          # system python has no numpy/sklearn — always use .venv
FORGE=~/ubaydolla/nebius-kit/agent/nebius-scientific-ai/forge.py
```

---

## 1. The task in one paragraph

Redesign trastuzumab's CDRH3 (10 residues, `WGGDGFYAMD`, chain B res 99–108 of
`Data/12-her2/her2_fv_1n8z.cif`, PDB 1N8Z Fv + HER2 domain IV). Judged on three axes:
**(1) binding affinity**, **(2) epitope preservation** vs trastuzumab, **(3) thermostability**.
Hard novelty filter: **>2 point mutations from every one of the 525,063 provided CDRH3s**, and
CDRH3 identity to trastuzumab **under 70%** (so ≥4 of 10 changed, strict reading — see Q1 in NOTES,
now moot for everything we ship). The bar for axis 1 is trastuzumab's own measured
**K_D = 0.095 nM mean / 0.135 nM median** (8 SPR replicates, 0.029–0.178 nM).

---

## 2. What is already delivered (committed, in `results/`)

A complete, ranked, fully gated submission of **20 scFv designs** exists and is defensible:

| file | what |
|---|---|
| `results/submission_top20.csv` | ranked list, 29 columns, every scorer + every gate |
| `results/submission_top20.fasta` | the 20 scFvs as submitted (242 aa, `(G4S)3` linker) |
| `results/summary.html` | one-page story, rendered from the CSVs |
| `results/all_designs_scored.csv` | all 41 incl. the in-run trastuzumab control |
| `results/rejected_designs.csv` | rejection ledger with reasons |
| `results/gate_verification.csv` | independent gate re-derivation, per design |
| `results/{epitope_shell,stabddg_scores,stabddg_joint,antifold_scores}.csv` | per-axis raw tables |
| `README.md` | the argument, exact reproduce commands, honest limitations |

Where the three axes stand:
- **Axis 2 (epitope): strong.** Measured, not assumed. 4/4 conditioned hotspot contacts retained,
  `epitope_jaccard` 0.82–0.95 vs an **in-run** trastuzumab fold (`epitope.py`).
- **Axis 3 (thermostability): defensible, not predicted.** No Tm predictor exists on either lane and
  we did not invent one. The argument is *construction*: identical framework, identical VL, identical
  disulfides; only 4–5 of 10 loop positions differ. Plus **F6**: every top design replaces D102 and
  thereby **deletes trastuzumab's own Asp102–Gly103 isomerisation liability** — a real improvement,
  obtained free, with no predictor.
- **Axis 1 (binding): the honest weak point.** 9 of 40 designs beat trastuzumab on 4 of 5
  independent scorers; none on 5/5, and we do not claim 5/5.

---

## 3. The three things that actually matter

### (a) The central scientific result: independent scorers stop agreeing in the design region
Established twice (**F2**, **F8**) and acted on (**D5**). Both of our additive generators — the
SPR-fitted ridge and AntiFold — produce argmax sequences that every *other* model rejects. In the
4-mutation shell, `kd_rf`/ipTM agree with each other, StaB-ddG **anti**-correlates with both
(ρ = −0.405 / −0.338, **F7**). Therefore:

> **Rule in force: a candidate is promoted only if independent models agree. Single-model argmax is
> never evidence.** Ranking is by `axis1_n_beat_baseline` — how many of five independent scorers put
> the design above trastuzumab — then by the **worst** of the three normalised axes. A percentile
> mean is a tie-break only, because averaging hides the disagreement. (`rank.py`)

### (b) The newest and best lead: seed on the measured champion, not on trastuzumab (F11/F12)
The user asked for the successful sequences in the data. **6 of 182** SPR-quantified CDRH3s beat
trastuzumab's median; the best is

```
WGSQGFYAYD    K_D = 0.0234 nM   (5.8x tighter than trastuzumab)   G101S, D102Q, M107Y
```

It is *in* the provided library, so it fails novelty and **cannot be submitted** — but it is a far
better seed than trastuzumab. `champion_shell.py` holds its three winning substitutions fixed,
enumerates all 1–3 further mutations at the other seven positions (247,779 sequences, exact), and
yields **56 novelty-compliant candidates** whose best `kd_rf` is **+0.427** — against +0.322 for the
shipped list, +0.127 for trastuzumab, −0.189 for the AntiFold pool and −1.173 for RFD4.
**This is the strongest axis-1 pool we have and it is NOT yet folded, ddG'd, or shipped.**

### (c) RISK 1, still open and now known to be hard
The shipped 20 are **not 20 independent bets**: all 20 mutate the same three positions (101/102/106),
**19 of 20** carry the identical **A106E**, only **4 distinct position-sets**. If the ridge's A106E
preference is an artifact of its 180-row fit, all 20 fail together.
Today's attempt to fix this by switching generators was **refuted** (**D6**): the AntiFold-intersection
pool loses on the SPR RF (0 of 731) *and* on physics-based StaB-ddG (2/39 stabilising vs **18/40** for
the shipped list). And the champion-seeded pool in (b) is **more** correlated still.
**Axis-1 quality and portfolio decorrelation are in direct conflict, now measured three ways.**
Do not present the 20 as 20 shots on goal.

---

## 4. Do this next, in order

1. **Fold + ddG the 56 champion-seeded candidates** (`runs/champion_candidates.csv`; re-run
   `champion_shell.py` with `MAX_PER_POSSET` raised, as the cap currently collapses the output to 3).
   - StaB-ddG, lane B, ~11 min, no GPU: copy `submit_stabddg_joint.py`, point it at the champion CSV,
     **keep an in-run reference variant** for sign calibration.
   - Boltz-2 holo cofold for ipTM + epitope: **see the GPU note in §5.** Include
     `BASELINE_trastuzumab` as an in-run control — comparisons must stay inside one folding model.
2. **Re-rank with `rank.py`** over the union of the shipped shell pool + the champion pool, and
   regenerate every deliverable (`make_submission.py`, `make_summary.py`). Re-run
   `verify_gates.py` from scratch on the final list — a generator's own filter is not evidence.
3. **Decide the portfolio explicitly** and say so in the pitch: N slots to the champion-seeded
   leaders, M to decorrelated hedges (the RFD4 pool is mechanistically independent of the ridge and
   its ipTM/epitope numbers for all 100 already exist, so hedges cost nothing to add).
4. **Get the repo off this VM — see §6. The VM is wiped tonight.**

---

## 5. Lanes, and the one real scheduling constraint

- **Lane A — 1× H100.** **NOT ours alone.** A teammate's container
  (`rfd4-proteina-her2-2026-11-07`, mount `/home/hackathon/lachlan/rfd4-docs`) has held it at
  95–100% util since before 11:00 and restarted fresh at ~11:32. Only ~4 GB of 80 GB VRAM is in use,
  so our fold *would* fit alongside — but it would slow their job.
  **The user was asked at 11:38 how to proceed and has not answered; do not share the GPU
  unilaterally.** Fallback: fold on lane B; `epitope.py` counts contacts from the mmCIF and does not
  need ipTM.
- **Lane B — Nebius Scientific AI**, no contention, runs concurrently with lane A. Verbs:
  `forge.py models|schema|run|submit|status|wait|fetch|resume|runs`. **The download verb is
  `fetch --dest DIR`, not `download`.** StaB-ddG entry semantic type is **`stab-ddg-workload-tar/v1`**.
  Fetched results are **hash-named blobs**: restore real names by matching each file's sha256 against
  the result manifest's `files` map (as done in `nebius-runs/antifold-skipnum-out/named/`).
- **Boltz-2 ipTM:** the *Apheris* Boltz-2 **does** return `interface_predicted_tm_score` and
  per-chain-pair ipTM. The "no ipTM" note in Nebius `RECIPES.md` applies only to Nebius's adapter.

Reference timings (all observed, not estimated): RFD4 design 100 replicates **11.4 min**;
Boltz-2 fold **~8 structures/min**; `refold-prepare`/`refold-score` are **CPU-only**; StaB-ddG
42 variants **~11–12 min**; AntiFold 41 backbones **~2 min**; champion shell search **~25 s**.

---

## 6. Open blockers

1. **No GitHub remote, `gh` not installed, VM wiped tonight.** The repo is committed locally only
   (`5760f9e`). This blocks the reproducible-repo deliverable (**20% of the pitch score**) and needs a
   human with credentials. Asked at 11:25 and again at 11:38, **still unanswered.** Options offered:
   user creates the remote; `git bundle` + `results/` tarball copied off the VM; or push a bundle to
   `s3://fs2-oslo-hackathon-participant-15-cd27f09d96/`. **Raise this first in a new session.**
2. **`abguard` does not exist** — not on PATH, not in pip, not on PyPI, not in any Apheris image, and
   a recursive search of `s3://apheris-oslo-hackathon/` found nothing. `verify_gates.py` reimplements
   its four documented checks. **The one check we cannot reproduce is immunum numbering confidence
   ≥0.1** (needs ANARCI/ANARCII; neither installs here, and AntiFold's bundled ANARCII could not
   number our Boltz-2 chains — it only ran at all with `skip_anarcii_numbering: true`, which also
   makes its `assumed_region` column wrong, so take CDRH3 positions from `pdb_posins`).
   Every design is a 227-aa scFv on an unmodified clinical framework so the risk is low, but it is
   **unverified and must be stated as such.**
3. **No thermostability predictor on either lane.** Axis 3 is a construction argument, not a
   prediction (§2). Do not let a Tm number appear anywhere.
4. **An artifact the user shared** (`claude.ai/artifact/SkR3mHVoR2bw88RnTn35Aq`, "some sequences that
   have been identified") **could not be read** — it is org-internal and the reader refused it.
   **Ask the user to paste those sequences in**; they have not yet been run through `verify_gates.py`
   or any scorer.

---

## 7. House rules (from `~/team16-ai-x-antibodies/CLAUDE.md` + `BUILD_SPEC.md:90`)

- Every number carries `scorer, model_id, weights, seed, run_id, git_sha, config_hash`.
- Tables are **rendered from data, never hand-typed**. Rejections are kept as **rows with a reason**,
  never silently dropped.
- Score the **trastuzumab baseline through the same gates**, in the same run.
- Compare scores **only within one folding model**.
- `resample_hotspots` stays **false** (it would drift the epitope, and axis 2 is epitope preservation).
- Trust RFD4's `result.json`, **not** `error.json` — the latter is written stale.
- **Open question for the organisers** (D2): `BUILD_SPEC.md:61` maps her2-novel → "de novo", but all
  four HER2 settings redesign CDRH3 *in place on the trastuzumab framework* (`her2_cdrh3.yaml` has no
  `contig`), so every output is an **optimisation**-track molecule. BUILD_SPEC looks wrong.
