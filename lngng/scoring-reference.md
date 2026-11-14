# Scoring reference — every function that ranks a design

Two layers decide outcomes: **what the organisers measure** (which we cannot change, only
predict) and **what we score internally** (which decides what we submit). This documents both,
with formulas taken from the code as implemented on 14 November 2026.

Sources: task PDF (`lngng/ai-antibodies-hackathon-apheris.pdf`), and
`~/team16-ai-x-antibodies/{novelty_filter,affinity_surrogate,thermostability,epitope_check,reward,rank}.py`.

---

# Layer 1 — the organisers' scoring

## 1.1 Hard filters (`abguard`) — pass/fail, runs before anything else

Not installed on our VM. All thresholds below come from the task description.

**Shared checks, every design:**

| Check | Threshold |
|---|---|
| Format | VHH, scFv or Fab |
| Numbering confidence (`immunum`) | ≥ 0.1 |
| Total length | ≤ 250 aa (combined VH+VL for paired formats) |
| Unpaired cysteines | none |

**Track assignment — computed, not chosen:**

```
identity(design, nearest known antibody) < 70%  →  de novo track
otherwise                                       →  optimisation track
```

Type-aware: a VHH is compared against known VHHs and heavy chains; scFv/Fab are split into
paired chains and compared as pairs.

**Track-specific checks:**

| Track | Must satisfy |
|---|---|
| De novo | ≥ 2 CDRH3 mutations from trastuzumab |
| Optimisation | CDRH3 identity to trastuzumab **< 70%** AND **> 2** point mutations from *any* provided antibody |

A design that fails its track's checks is **rejected, never reclassified**. There is no
fallback from de novo into optimisation.

For a 10-residue CDRH3 this means **≥ 4 mutations from trastuzumab** (3 leaves exactly 70.0%,
which is not *under* 70%) and **≥ 3 differences from every one of the 525,063 provided
sequences**.

## 1.2 Tiering — per design, on measured values

Adaptyv expresses each design cell-free and measures three properties against trastuzumab,
which is measured in the same run as the baseline.

| Axis | Measurement |
|---|---|
| Affinity | SPR |
| Epitope preservation | Competitive binding against trastuzumab |
| Thermostability | Direct measurement |

```
S-tier = beat baseline on 3 of 3
A-tier = 2 of 3
B-tier = 1 of 3
Unsuccessful = 0 of 3
```

**A tie counts as a failure.** The task's figure marks only a strict improvement as a filled
vertex; ties and regressions are hollow. "Comparable" affinity is not enough on that axis.

## 1.3 Team ranking

Most S-tier designs wins; ties broken by most A-tier, then most B-tier. **Ten B-tier designs
lose to two S-tier designs.** The objective is a conjunction over three axes, not a sum — this
is the single most important fact for how we rank internally.

## 1.4 Pitch scoring and allocation

| Weight | Criterion |
|---|---|
| 40% | Judgment — reasoning behind the molecules chosen |
| 20% | Execution — reproducible pipeline in a GitHub repo |
| 20% | User fit — what the design is for, who needs it |
| 20% | Path to value — credible route to real use |

Allocation: 4 designs tested for every team, +12 for 1st, +8 for 2nd, +4 for 3rd, +2 per judge
nomination. We submit a ranked list of up to 20. Prizes: €3000 split equally between the two
tracks (€800/€500/€200 each), $500 best pitch, plus an AMASS platform award.

---

# Layer 2 — our scoring functions

## 2.1 `novelty_filter.py` — hard gate

```python
hamming(a, b) = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
```

`is_novel(cdrh3, by_len)` rejects when:

| Condition | Current code | **Should be** |
|---|---|---|
| Too close to trastuzumab | `hamming(tz) < 3` | **`< 4`** |
| Identical to a provided CDRH3 | in provided set | same |
| Too close to a provided CDRH3 | any 1-mutation neighbour in set | **any 1- or 2-mutation neighbour** |

**Both thresholds are one too permissive** — the rule is strictly more than 2 mutations from
any provided sequence, and strictly under 70% identity to trastuzumab. The neighbour trick
(checking the candidate's mutational neighbourhood instead of scanning 525k rows) stays valid
at radius 2; it costs 10 × 19 = 190 lookups at radius 1 and 16,245 at radius 2, still trivial.

`hard_checks(full_seq)`: length ≤ 250, even cysteine count. **Missing** the format and
numbering-confidence checks — those need ANARCI/abnumber, which is noted in the docstring and
not yet installed.

## 2.2 `affinity_surrogate.py` — the primary ranker

**Encoding.** Fixed L=12 window, 21 channels per slot (20 amino acids + a gap channel). The
first ⌈n/2⌉ residues fill slots from the N-terminus, the rest from the C-terminus, gap in the
middle — CDRH3 is anchored at both ends, so this handles 9–12-mers without truncating. For
10-mers it reduces to plain position-wise one-hot.

**Two heads, both trained on the provided measurements:**

| Head | Model | Training data | Target |
|---|---|---|---|
| `reg` | `RidgeCV(alphas=logspace(-2,3,12))` | ~190 SPR rows with a quantified KD, excluding the 8 `ARDRIVDDYYYYGMDV` rows | log₁₀(KD in nM) |
| `clf` | `LogisticRegression(C=1.0, max_iter=400)` | all 524,346 FACS rows | 1 if `high`, else 0 |

**Output:**

```python
pkd       = -pred_log_kd                       # higher = tighter
surrogate = pkd + 2.0 * (p_high - 0.5)         # the ranking key
ood       = len(cdrh3) not in (9, 10)          # longer loops: unreliable
```

Reported performance: Spearman 0.72 against measured affinity, FACS high-binder AUC 0.90.
Note the demo measures the regressor alone by 5-fold CV and the classifier on a 120k subsample
— the 0.72 headline should be re-derived on the full fit before it goes in the pitch, per the
repo's provenance rule.

## 2.3 `thermostability.py` — heuristic proxy, not a Tm

Explicitly documented as *"NOT a measured Tm… soft penalty / tie-break only."*

```
penalty = 0.15 × count of [FILVWY]{4,}        (hydrophobic runs → aggregation)
        + 0.10 × count of [KR]{3,} or [DE]{3,} (charge patches)
        + 0.30 × count of C                    (unpaired thiol)
        + 0.05 × (count of P - 1, floored at 0)
        + 0.10 if glycine fraction > 0.4
        + 0.10 × max(0, |net charge| - 2)      where net = (K+R) - (D+E)
        + 0.00 × lost_anchors                  (disabled: overfit to trastuzumab)

stability_score = clamp(1 - penalty, 0, 1)
```

Upgrade path already named in the docstring: ThermoMPNN, ESM-IF ddG, or Nebius `stab-ddg` on
the full VH.

## 2.4 `epitope_check.py` — structural, needs a co-folded CIF

Chain layout: A = trastuzumab VL, B = VH, C+D = HER2 domain IV. A contact is any antibody
heavy atom within **4.5 Å** of a HER2 heavy atom.

```
epitope_recall = |design_contacts ∩ ref_contacts| / |ref_contacts|
jaccard        = |∩| / |∪|
hotspots_hit   = design_contacts ∩ {C572, C573, D591, D593}
```

Reference is 1N8Z (trastuzumab–HER2). Recall is the share of trastuzumab's epitope the design
keeps — the closest proxy we have for the competitive-binding assay.

## 2.5 `reward.py` — fast reward for the IgLM loop

Sequence-level only, no folding (~ms per candidate instead of ~75 s).

```
reward = surrogate - 0.75 × (1 - stability_score)
       = -10.0                                  if not novel, or hard_checks fail
```

The novelty gate is applied *inside* the reward, so the loop does not spend its budget on
ineligible sequences. That is correct and worth keeping.

## 2.6 `rank.py` — final re-ranker

**Gates (reject before scoring):**

```
novelty_filter.is_novel(cdrh3)       must pass
novelty_filter.hard_checks(vh)       must pass
iptm >= 0.6                          IPTM_FLOOR, geometry gate only
min_ipae <= MAX_IPAE                 optional, currently None
```

**Score:**

```
final = 1.00 × surrogate
      + 1.50 × epitope_recall                 (only if a design_cif is given)
      + 0.25 × |hotspots_hit|                 (0 to 4)
      - 1.00 if epitope_recall < 0.30         (off-epitope penalty)
      - 0.75 × (1 - stability_score)
      + 0.20 × (iptm - 0.60)                  (tie-break)
```

**Selection:** sort by `final` descending, drop any candidate within Hamming < 2 of an
already-picked design (`MIN_DIST = 2`), stop at 20. Every rejected row keeps its reason, per
the repo's provenance rule.

## 2.7 Folding metrics (Boltz-2) — measured, not assumed

Correlations against our own scorers on 100 R1 designs (`ubaydolla/NOTES.md`, F4):

| Metric | vs SPR-model affinity | vs FACS classifier |
|---|---:|---:|
| `iptm_B_D` | **+0.343** (p=0.00047) | +0.107 (ns) |
| `iptm_B_C` | **+0.330** (p=0.0008) | +0.092 (ns) |
| ipTM global | +0.297 (p=0.0027) | +0.100 (ns) |
| pTM | +0.273 (p=0.0059) | +0.084 (ns) |
| ranking score | +0.211 (p=0.035) | +0.102 (ns) |
| **pLDDT** | **−0.013** (p=0.90) | +0.073 (ns) |

ipTM carries real signal on our designs — about the same magnitude as the empirical model's
own correlation with real SPR (0.35). **pLDDT is noise; do not rank on it**, despite the task
brief suggesting `binder_plddt`.

The separate Smorodina result (ipTM ≈ blind) is about telling cognate from *shuffled* pairs, a
much coarser task than ranking variants of one real complex. It does not transfer here.

---

# Known issues in the current scoring stack

1. **Two off-by-one novelty thresholds** (§2.1). Permissive direction, so current pass counts
   are overcounts. Fix before any submission.
2. **The aggregation shape is wrong.** `rank.py` sums the axes, but tiering is a conjunction:
   S-tier needs all three to beat baseline. A design with a huge surrogate score and a hollow
   thermostability vertex is B-tier. Rank on *predicted probability of beating baseline on
   each axis independently*, and prefer candidates that clear all three.
3. **Thermostability has no real predictor** (§2.3) — a third of the tier currently scored by
   a hand-weighted liability heuristic. Cheapest available tier upgrade: StaB-ddG over the
   shortlist.
4. **Length handling is inconsistent across modules.** The surrogate flags anything outside
   9–10 residues as OOD, but 11-mer designs are in circulation; `rank.py`'s demo silently
   truncates `WASSSEYYFDY` to `WASSSEYYFD`. A truncated candidate is a different molecule.
   Decide on one CDRH3 convention (10-residue or 11-residue Kabat) and apply it everywhere.
5. **`MIN_DIST = 2` contradicts the novelty rule.** Two picks 2 apart are mutually fine, but
   the rule demands ≥3 from every provided sequence; once one is submitted, consistency argues
   for ≥3 between picks too.
6. **ipTM may be underweighted.** `W_IPTM = 0.2` on `(iptm - 0.6)` contributes ~0.05 at most,
   while F4 measured ipTM at ρ≈0.33, comparable to the surrogate's own 0.35 against SPR.
7. **`hard_checks` is incomplete** — no format or numbering-confidence check, so a design can
   pass locally and fail `abguard`.
8. **The IgLM prior pulls toward ineligible space.** It is fine-tuned on exactly the sequences
   the novelty filter forbids. `reward.py` gates correctly, so the loop is safe, but expect
   generation yield to fall as the filter tightens, and budget for it.
