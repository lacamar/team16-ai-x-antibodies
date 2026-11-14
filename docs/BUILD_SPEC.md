# BUILD SPEC — spin up the Team 16 repo + pipeline

**Who reads this:** a Claude Code session, on the day, right after the 09:30 brief.
**What to do:** build the repo and a modular, reproducible candidate-evaluation pipeline, then stop and let the team fill in the real target and gates.

## Ground rules (contest hygiene)

- **Private repo.** `--private`. Judges get access when asked, not a public link.
- **Nothing before the day.** The repo is born on the day: no repo, no commit, no push exists before **09:00, 14 Nov 2026**. Every commit and author date must be ≥ 09:00. Author the code **fresh in the new repo today** — do not import or commit any file created earlier (git doesn't store file mtimes, so a fresh commit today is clean regardless of a file's age on disk). Prep material on this machine is reference only.

The design mirrors our grant pipeline (`creatorbrunchclub/tools/grants`): a **declared graph of discrete nodes**, each a self-contained step with a fixed input→output contract, runnable and inspectable on its own. Here the nodes are **gates** and the things flowing through them are **candidate nanobodies**.

---

## Core principle

A candidate flows through a pipeline of **gates**. Each gate reads candidates in, adds scores and a pass/fail verdict, writes candidates out. Every gate is a **reproducible programmatic step** — deterministic code, pinned model versions, fixed seeds, cached I/O. **No LLM inside a gate** (LLM output isn't reproducible). The LLM's job is *around* the pipeline — proposing approaches, writing reasoning, driving the tools — never scoring a candidate.

Three rules, non-negotiable:
1. **Reproducible:** `just all` reproduces the whole run from a tagged commit, offline, from cache, with no LLM in the loop.
2. **Modular:** adding an evaluation = one new gate file + one line in the graph. Nothing else changes.
3. **Legible:** `just graph` prints the pipeline so anyone can see, in one screen, what runs in what order and what each gate reads and writes.

---

## Two funnels: validate the problem, then the solution

A nanobody has to bind *something worth binding*. So the pipeline is **two funnels in series**, each the same generate → gate → rank → pick shape.

**Funnel 1 — Target validation (the "why").** Candidates are *(disease, target)* pairs. Gates keep only a prevalent disease with a well-established, antibody-tractable target that has a usable structure. Reproducible and programmatic — it runs off structured databases, not opinion:

| Gate | Question | Reproducible source (public API, cacheable) |
|---|---|---|
| burden | Is the disease prevalent — not n-of-1? | GBD / Our World in Data / WHO (prevalence, DALYs) |
| association | Is the target–disease link well established? | Open Targets association score + evidence classes |
| tractability | Reachable by an antibody (extracellular), with precedent? | Open Targets tractability; existing approved mAbs |
| structure | Is there a usable antigen structure + a defined epitope? | PDB / UniProt / AlphaFold DB |

Output: **one target dossier** — disease, UniProt target, antigen structure, chosen epitope, and the scores/evidence behind the pick. That dossier is the input to Funnel 2, *and* it is most of the User-fit (20%) and Path-to-value (20%) story plus the reasoning that earns Judgment (40%). Most teams skip this — it's where the points are.

**Funnel 2 — Binder validation (the "what").** Candidates are nanobodies, conditioned on the dossier's epitope/structure. The two filters the team named:
- **Plausibly binds** — co-fold the binder against the *validated* antigen; score interface quality and whether it hits the intended epitope (contacts, clashes, ipSAE, ipTM). Confidence ≠ affinity; use controls.
- **Plausibly manufacturable** — developability: expression, solubility/aggregation, chemical liabilities, stability, germline-ness. All programmatic.

Output: ≤10 binders, each with a reason, against a target that was itself justified.

The LLM drafts the dossier narrative and the pitch *from* the fetched evidence; it never sets a gate verdict. (Confirm each data source is reachable on the day; cache every response so a rerun is offline and identical.)

---

## Day-0 steps (do these in order)

1. Work in a **fresh empty directory** (e.g. `~/.local/src/team16-antibodies`), **not** the prep workspace — so nothing pre-written is ever swept into a commit. Copy only this spec and `CONTRIBUTORS.md` in as reference. Then `git init`; create a **private GitHub repo** (`gh repo create team16-antibodies --private --source=. --remote=origin`). Add the handles in `CONTRIBUTORS.md` as collaborators: `gh repo edit --add-collaborator <handle>` (or via the web UI).
2. Scaffold the layout below, **authored fresh in the new repo today**. Reference patterns (the gate contract, provenance columns, `disk_cache`) are described in this spec — write them anew; do not commit any pre-existing file.
3. Fill `config.toml` with the **real target** from the brief (antigen, PDB id, epitope, format: VHH/scFv/Fab).
4. Wire gates **one at a time**, each proven end-to-end on ~20 dummy candidates before the next. Walking skeleton first, real tools second.
5. Commit after every working gate. Tag the final run commit.

---

## Repo layout

```
config.toml              # the target + all knobs (hashed into every result = provenance)
pipeline.py              # THE GRAPH: ordered list of gates + their deps. The one place the pipeline is defined.
core.py                  # run ids, git sha, config hash, JSONL io, disk_cache  (reuse from seed)
gates/
  # Funnel 1 — target validation (the "why")
  t10_candidates.py      # assemble candidate (disease, target) pairs
  t20_burden.py          # disease prevalence / DALYs gate               (GBD / OWID / WHO)
  t30_association.py     # target-disease association gate                (Open Targets)
  t40_tractability.py    # antibody-tractable + extracellular + precedent (Open Targets)
  t50_structure.py       # usable antigen structure + epitope             (PDB / UniProt / AlphaFold)
  t90_target_select.py   # rank survivors, pick ONE -> target dossier
  # Funnel 2 — binder validation (the "what"), conditioned on the dossier
  b10_generate.py        # make nanobody candidates (de novo or redesign)
  b20_fold.py            # co-fold binder+antigen -> confidence (pLDDT, ipTM, PAE)
  b30_interface.py       # binding plausibility: interface quality + right epitope (contacts, clashes, ipSAE)
  b40_developability.py  # manufacturability: expression, solubility, liabilities, stability, germline-ness
  b50_select.py          # hard gates + composite rank + diversity dedup -> <=10
  b90_report.py          # render tables + provenance from the data
runs/<run_id>/           # all artifacts for one run (gitignored except the final tagged one)
  targets.jsonl          # Funnel 1 candidates, a column per gate
  target_dossier.json    # the one chosen target + evidence -> input to Funnel 2
  candidates.jsonl       # Funnel 2 nanobodies, a column per gate
  gates.jsonl            # one row per (candidate, gate): passed, reason, scorer_version, model_id, seed
  shortlist.jsonl
.cache/                  # disk_cache of every external/model call, keyed by input hash
DECISIONS.md             # why, per choice (Judgment = 40%)
docs/report.md           # rendered tables (never hand-typed)
```

---

## The gate contract

Every gate is a module in `gates/` exposing exactly:

```python
GATE = "b30_interface"          # id, matches filename
READS = ["structure"]           # fields it needs present on a candidate
WRITES = ["interface_score", "n_contacts", "clashes"]   # fields/scores it adds

def run(cands: list[dict], cfg: dict) -> list[dict]:
    """Pure function of (cands, cfg). Deterministic. Wrap any external/model
    call in core.disk_cache so a second run is offline and identical.
    Append scores to each candidate and emit a gates.jsonl verdict row."""
    ...
    return cands
```

- **Deterministic or it doesn't belong here.** If a step uses a model, pin `model_id` + `seed` and cache by input hash — same input, same output, forever.
- **A gate that rejects** writes a `verdict` row `{candidate_id, gate, passed:false, reason:"developability: NG motif in CDR3"}` and drops the candidate from the output (but the verdict row survives — negative results are evidence).
- **Provenance travels with every number:** `scorer`, `scorer_version`, `model_id`, `seed`, `run_id`, `git_sha`, `config_hash`. Set once in `core`, attached by every gate.

## The graph (`pipeline.py`)

The whole pipeline is one declared list — the single source of truth, like the grant router's node graph:

```python
PIPELINE = [
    # Funnel 1 — target validation; a go/no-go for the whole project
    ("t10_candidates",    []),
    ("t20_burden",        ["t10_candidates"]),
    ("t30_association",   ["t20_burden"]),
    ("t40_tractability",  ["t30_association"]),
    ("t50_structure",     ["t40_tractability"]),
    ("t90_target_select", ["t50_structure"]),             # -> target_dossier.json
    # Funnel 2 — binder validation; consumes the dossier
    ("b10_generate",      ["t90_target_select"]),
    ("b20_fold",          ["b10_generate"]),
    ("b30_interface",     ["b20_fold"]),
    ("b40_developability",["b10_generate"]),               # needs only the sequence
    ("b50_select",        ["b30_interface", "b40_developability"]),
    ("b90_report",        ["b50_select"]),
]
```

The two funnels are one graph, so `just graph` shows the whole thing and the target dossier is the visible seam between them. Funnel 1 is also the project's **go/no-go**: if no disease–target pair clears its gates, stop and pick a different area rather than designing binders for a weak target.

- `just all` runs them in dependency order, piping `runs/<id>/candidates.jsonl` through each.
- `just gate b30_interface` runs one gate against the current run — the unit of work and of debugging.
- `just graph` prints the DAG with each gate's READS/WRITES — the "see how it works" view. (Mirror `grants/.../graphmap.py`: it just imports each gate, checks every READS is produced by some upstream WRITES, and errors if the graph is broken.)

---

## What goes where: deterministic vs LLM

| Reproducible gate (in `gates/`) | LLM / human work (NOT a gate) |
|---|---|
| structure prediction (pinned model+seed) | deciding which target/epitope to go for |
| interface geometry, clashes, buried area | proposing generation strategies |
| developability motif/charge/liability scan | writing `DECISIONS.md` and the pitch |
| composite ranking, diversity dedup, hard gates | reading literature via AMASS |
| confidence scores (pLDDT/ipTM/PAE/ipSAE) | judging a borderline shortlist pick (then record the reason) |

Confidence scores are **not** affinity — they rank, they don't prove binding. Include positive/negative/baseline controls in the scoring so the numbers mean something.

## Prize + scoring hooks to bake in
- Put **Apheris' OpenFold** in the fold/score path (required for the cash prize).
- `b50_select` must emit a **reason per pick and per rejection**, and enforce **diversity** so the 10 aren't near-duplicates.
- `b90_report` renders the shortlist + scorer-control table straight from the data into `docs/report.md` and the README — no hand-typed numbers.

---

## Definition of done for the skeleton (before real tools)
`just all` runs g10→g90 on 20 dummy candidates, offline, and produces a shortlist table with provenance and a `just graph` that validates. Commit it. *Then* swap dummy gates for real tools one at a time.
