# Scoring API

One call that scores a HER2 CDRH3 design on every judged axis and tells you whether it is
even eligible.

```bash
./.venv/bin/python scoring.py                      # self-test on 5 known sequences
./.venv/bin/python server.py 8765                  # HTTP, loads in ~3 s
curl 'localhost:8765/score?cdrh3=WASSSEYYFDY'
```

Interactive docs at **http://localhost:8765/docs** (Swagger UI). From your laptop:
`ssh -L 8765:localhost:8765 hackathon@<vm>`, then open the same URL locally.

## Why this exists

Scoring lives in six modules in `team16-ai-x-antibodies`, each with its own CDRH3 length
convention and its own novelty threshold. That is where both known bugs came from: thresholds
one too permissive, and a demo that silently truncated an 11-mer to a 10-mer. This wraps them
in a single entry point with one convention and one set of rules, and it never truncates.

It does not reimplement the scorers. It imports them, so `team16` stays the single source of
truth for affinity, stability and epitope. Novelty and track classification are implemented
here because the team version is known to be wrong.

## Routes

| Route | Does |
|---|---|
| `GET /docs` | Swagger UI — interactive, with Try it out |
| `GET /openapi.json` | OpenAPI 3.1 spec (`openapi.json`) |
| `GET /health` | liveness, scorer version, provided-sequence count |
| `GET /rules` | the thresholds in force and where they come from |
| `GET /score?cdrh3=…&vh=…` | score one design |
| `POST /score` | `{"cdrh3": …, "vh": …, "design_cif": …}` |
| `POST /score/batch` | `{"items": [{…}, …]}`, up to 500 |

Or import it directly:

```python
from scoring import score_sequence, score_batch
score_sequence("WASSSEYYFDY", vh=full_vh, design_cif="path/to/complex.cif")
```

## What comes back

```jsonc
{
  "cdrh3": "WASSSEYYFDY",
  "length": 11,
  "convention": "11-residue (Kabat, trailing Y)",
  "eligible": true,
  "verdict": "eligible",
  "reasons": [],
  "track":    { "track": "optimisation", "identity_to_trastuzumab_vh": 0.9417 },
  "novelty":  { "mutations_from_trastuzumab": 7, "identity_to_trastuzumab": 0.3636,
                "legacy_team16_passes": true, "note": null },
  "axes": {
    "affinity":        { "pred_kd_nM": 69.66, "p_high": 0.062, "out_of_distribution": true },
    "thermostability": { "score": 1.0, "baseline_trastuzumab": 1.0 },
    "epitope":         { "available": false }
  },
  "predicted_tier": { "axes_beating_baseline": 0, "axes_decided": 1, "tier": null },
  "provenance": { "scorer_version": "1.0.0", "provided_sequences": 525063 }
}
```

## Design decisions worth knowing

**Eligibility uses the task description's thresholds, not team16's.** Identity to trastuzumab
strictly under 70% (4+ mutations on a 10-mer) and strictly more than 2 mutations from every
one of the 525,063 provided sequences. `novelty.legacy_team16_passes` reports what the current
`novelty_filter.py` would have said, so a disagreement is visible rather than silent.

**The track is computed, not chosen.** `abguard` assigns it from whole-chain identity to the
nearest known antibody. We compare against trastuzumab's VH only, so the real identity can
only be higher — if we say `optimisation`, it is optimisation. A CDRH3 graft on the
trastuzumab framework lands at ~94% identity and can never reach the de novo track.

**Tiering is a conjunction, so axes are reported separately.** S-tier needs all three to beat
the baseline; there is no weighted sum here on purpose. A tie counts as a failure, which is
why trastuzumab-equal stability scores `false`.

**Out-of-distribution is surfaced, not hidden.** The affinity surrogate was trained on 9- and
10-mers. For an 11-mer it still returns a number, but `out_of_distribution: true` and the axis
is excluded from the tier count rather than being trusted.

**Missing dependencies degrade, they do not crash.** Novelty, track and eligibility are
standard library only. Affinity needs numpy + scikit-learn, epitope needs gemmi and a
co-folded CIF; each reports `available: false` with the reason.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy scikit-learn scipy gemmi
```

Paths are overridable: `HER2_DATA` (default `~/data/her2-trastuzumab`) and `TEAM16_DIR`
(default `~/team16-ai-x-antibodies`).

## Performance

Measured on this VM, 10-residue CDRH3s, surrogate already fitted:

| How | Throughput | Per sequence |
|---|---:|---:|
| `POST /score/batch`, 200 items | 157 seq/s | 6.4 ms |
| `POST /score`, one at a time | 142 seq/s | 7.0 ms |
| Direct import, novelty only | 483 seq/s | 2.1 ms |
| **A fresh process per sequence** | **0.7 seq/s** | **1,390 ms** |

The API exists to avoid that last row: loading 525,063 sequences costs ~1.3 s and fitting the
surrogate ~10 s, both once per process. HTTP adds roughly 4 ms over a direct import, so:

- **Import `scoring` directly** inside a tight loop in your own long-running process — and for
  the IgLM reward loop use team16's `reward.py`, which skips folding entirely.
- **Call the API** whenever you cross a process boundary: curl, the dashboard, a teammate in a
  different environment, or the final shortlist gate where one agreed implementation matters
  more than 4 ms.

## Known limits

- `hard_checks` from team16 covers length and cysteine parity only. Format and
  numbering-confidence checks need ANARCI/abnumber, which is not installed. A design can pass
  here and still fail `abguard`.
- The first `/score` call fits the surrogate (~10 s). After that it is cached in-process.
- Epitope needs a co-folded CIF with chains A/B = antibody, C/D = HER2.
- Server binds `127.0.0.1` only. For another machine, tunnel over SSH rather than exposing it.
