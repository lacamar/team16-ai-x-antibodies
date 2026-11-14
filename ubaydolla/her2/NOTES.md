# Team 16 — HER2 / trastuzumab optimisation — running log

Hackathon: AI × Antibodies, Oslo, 2026-11-14. Track: **optimisation**.
Working root: `~/ubaydolla/her2`. VM is wiped tonight — everything worth keeping goes to GitHub.

---

## Inventory (10:00–10:12)

### Hardware / lanes
- **Lane A**: 1× NVIDIA H100 80GB, idle at start. 16 vCPU, 196 GB RAM, 363 GB free on `/`.
  **One GPU only — all GPU stages must be serialised.** This is the scheduling constraint for the whole day.
- **Lane B**: Nebius Scientific AI via `forge.py`. 34 models with a tested input format.
  `export NEBIUS_CREDS=~/ubaydolla/nebius-kit/credentials.private.json`

### Corrections to the starting brief
| Brief said | Reality |
|---|---|
| `ls /data` | Does not exist. Real HER2 material is `~/lachlan/rfd4-docs/` (already a sync of the Apheris bucket) and `~/team16-ai-x-antibodies/data/her2/`. |
| `aws s3 ls s3://apheris-oslo-hackathon/` | **Bucket does not resolve** from this VM with our private-storage keys (`NoSuchBucket`), nor via public AWS. Our own bucket `fs2-oslo-hackathon-participant-15-cd27f09d96` has only `examples/` + `onboarding-checks/`. Not blocking: the `rfd4-proteina/docs/` tree is already on disk, Tutorial.html included. **Still unreached: the `zenodo-*` affinity datasets** referenced by `17-her2-data.md`. |
| `abguard --help` | **`abguard` does not exist** — not on PATH, not in pip, not on PyPI (404), not inside any Apheris image. Apheris docs mention it once (`18-vm.md:184`) with no definition or install. See "Blocker 1". |
| Aggrescan3D / SoluProt / NetSolP / ProteinMPNN-soluble | **None are present** on either lane. See "Blocker 2". |
| `foundry-toolkit weight/skill/workflow list` | All return `[]` — we are entitled to **modules only**. The "her2-novel-2026-11-09 weights" are the image tag, not a separate artifact. |

### Foundry modules (all images already pulled — no download wait)
`apheris-rfd4-proteina` v0.3.0 × {antibody-2026-10-30, her2-2026-11-07, her2-novel-2026-11-09},
`apheris-boltz` {boltz1, boltz2}, `apheris-openfold3` {of3ob, of3p2}, `apheris-protenix` {v1-default, v2},
`apheris-esmfold2`, `apheris-mol1`, `apheris-msa`, `apheris-data`.

Each RFD4 image holds **exactly one** flow checkpoint (verified: `ls /weights/flow_checkpoints`), so
`--model` is unnecessary — omitting it selects the image's own weights. Tasks per image: `design`,
`refold-prepare`, `refold-score`, `metrics`, `finetune`.
Resources: `design` = 1 GPU / 8 CPU / 64 GiB. **`refold-prepare` and `refold-score` are CPU-only (1 CPU, 2 GiB)** —
they never contend for the GPU.

### The four HER2 "models" = 2 weights × `sequence_tsr_k` (`12-her2.md:63-84`)
| Model | weights | `--sequence-tsr-k` | distinct | pass challenge filter | interface err |
|---|---|---|---|---|---|
| conservative | her2-2026-11-07 | unset (4) | 7% | **0%** | 9.4 Å |
| balanced | her2-2026-11-07 | 2 | 64% | **6%** | 9.1 Å |
| exploratory | her2-2026-11-07 | 1 | 100% | **74%** | 9.1 Å |
| **novel** | her2-novel-2026-11-09 | **unset** | 90% | **86%** | **6.8 Å** |

**Decision D1 — use `her2-novel-2026-11-09` with NO `--sequence-tsr-k`.**
Why: best novelty pass rate and lowest interface error. `12-her2.md:120` is explicit — "For HER2
conservative or HER2 novel, remove `sequenceTsrK=2`". An earlier reading of mine that the ready-made
script was buggy for omitting the flag was **wrong**; omitting it is correct for this weight.
Caveat on record: `12-her2.md:132` — her2-novel's numbers come from **one seed, 100 designs**.
Rejected: conservative/balanced (0% and 6% novelty pass = wasted GPU). Exploratory held as backup.

**Decision D2 — the weights do not determine the track; the molecule does.**
`~/team16-ai-x-antibodies/docs/BUILD_SPEC.md:61` maps her2-novel → "de novo" and her2-2026-11-07 →
"optimisation". That mapping is not in the Apheris docs and looks like our own error: all four settings
redesign CDRH3 *in place on the trastuzumab framework* (`her2_cdrh3.yaml` has no `contig`), so every
output is an optimisation-track molecule. **To confirm with organisers.** BUILD_SPEC should be fixed.

### HER2 input geometry (verified against the CIF, not assumed)
`~/lachlan/rfd4-docs/Data/12-her2/her2_fv_1n8z.cif` — PDB 1N8Z cropped to Fv + HER2 domain IV. mmCIF only; PDB is rejected.
- **A** = trastuzumab VL, res 1–107
- **B** = trastuzumab VH, res 1–120. CDRH3 = **B99–108**, `WGGDGFYAMD`, exactly 10 designable residues
  (the CIF annotates B97–109 as `cdr3`, but S97/R98/Y109 are held by `C_SEQ`)
- **C** = HER2 domain IV res 489–580, **D** = HER2 res 591–607
- HER2 is split C/D because res 581–590 are unresolved and input validation rejects a numbering gap in one chain.
- **Epitope, operationally** = the 4 `C_HOT` atoms: HER2 **Pro572 CD**, **Phe573 CD1** (chain C),
  **Ile591 CG2**, **Lys593 NZ** (chain D). `hotspot_obedience` scores against these.
- `resample_hotspots` stays **false** — it would drift the epitope, and axis 2 is epitope preservation.

### Prior run already on disk (`~/lachlan/rfd4-docs/runs/live/`, 09:52–09:56 today)
Full loop at N=4/TOPN=2 in ~4 min. Two things learned:
- `design/error.json` ("completed without writing designs") is **stale** — written 09:52:33, while
  `generation.log` ends "✓ Generation complete!" at 09:54:53 and `result.json` lists 4 designs.
  **Trust `result.json`, not `error.json`.**
- `refold_metrics.csv`: `pass` **empty**, `pass_note` = "fewer than 3 target atoms fixed in absolute
  coordinates (coordinate group 0), so target-aligned RMSDs are undefined". Scores that *did* compute,
  best design: `iptm` 0.965, `min_ipae` 0.310, `binder_plddt` 98.5, `binder_ptm` 0.993.

---

## Runs

### R1 — production design, 100 replicates (fired 10:14)
```
cd ~/ubaydolla/her2
docker run --rm --gpus all --shm-size 16g --user $(id -u):$(id -g) \
  -e HOME=/tmp -v /etc/passwd:/etc/passwd:ro -v $PWD:/work -w /work \
  quay.io/apheris/oslo-hackathon:apheris-rfd4-proteina-v0.3.0-her2-novel-2026-11-09 \
  module-run design \
    --input Data/12-her2/her2_cdrh3.yaml --output runs/prod100/design \
    --num-replicates 100 --seed 1 --recycle-strategy high/front_loaded --compile true
```
Log: `~/ubaydolla/her2/logs/prod100-design.log`. Spec = the unmodified shipped `her2_cdrh3.yaml`.
`--compile true` added (CDRH3 length is fixed at 10, so one padded length; pays off above ~8 designs).
Expect ~25–50 min. No `--sequence-tsr-k` (D1). Seed 1 recorded for provenance.

---

## Open blockers

**Blocker 1 — `abguard` is missing, and it is the gate that decides whether a design counts at all.**
Four documented checks: VHH/scFv/Fab class · no unpaired Cys · numbering confidence ≥0.1 · length ≤250 aa.
`~/team16-ai-x-antibodies/novelty_filter.py` already covers the novelty rule
(≥3 of 10 CDRH3 positions changed vs trastuzumab = ≤70% identity; ≥2 changes vs all 525,063 provided
CDRH3s) plus length ≤250 and odd-Cys. **Not covered: the format-class check and numbering confidence —
both need ANARCI/abnumber, which is not installed.** Either get `abguard` from the organisers or
finish the reimplementation. Do not submit anything ungated.

**Blocker 2 — thermostability has no predictor on either lane, and it is the axis that separates A from S.**
`stab-ddg` (lane B, ~12 min) is **binding** ΔΔG across two partner chains, not folding/thermal stability.
ThermoMPNN / ESM-IF are named in our GUIDE_DISTILLED with no runner. Nothing predicts Tm.
Available substitutes: ESM-2 / ESM-C sequence likelihood, germline proximity, deterministic
liability-motif regex. All are weak proxies and must be labelled as such.

**Blocker 3 — nothing in RFD4 optimises affinity.** `12-her2.md:125`: "The screen measured binding, not
affinity. **Nothing trained them to improve affinity, so use the SPR data for that.**" RFD4 gives novel
loops with good interface geometry; it does not rank by KD. Axis 1 needs a data-driven scorer.

---

## The data we are underusing (local, no bucket needed)
- `~/team16-ai-x-antibodies/data/her2/trastuzumab_her2_spr_all_768.csv` — 768 rows,
  `cdrh3,vh_sequence,vl_sequence,kd_nM,binding_result`. **190 real SPR KD values**, nM, linear, lower=better.
  Trastuzumab appears 8× at KD 0.03–0.18 nM — **that is the bar for axis 1**. 12 variants beat 0.152 nM.
  CDRH3 is the only variable region: VH = `EVQ…YYCSR` + cdrh3 + `YWGQGTLVTVSS`.
  Drop the 8 `ARDRIVDDYYYYGMDV` rows — different antibody.
- `~/team16-ai-x-antibodies/data/her2/tz_her2_facs_3classes_524346.csv.zip` — **524,346 rows**,
  all 10-mer CDRH3 on the trastuzumab framework, `binding_class` high/medium/low (172k/190k/162k),
  `edit_distance` 1–10 from trastuzumab. Trastuzumab itself absent.
Together: 525,063 unique provided CDRH3s = the novelty reference set **and** a large supervised
training set. These are the only experimentally grounded signal we have for axis 1.

---

## Provenance rule (from `~/team16-ai-x-antibodies/CLAUDE.md` + BUILD_SPEC:90)
Every number carries: `scorer, model_id, weights, seed, run_id, git_sha, config_hash`.
Tables rendered from data, never hand-typed. Rejections kept as rows with a reason.
Score the trastuzumab baseline through the same gates. Compare scores **only within one folding model**.

---

## Session 2 (10:17–) — inventory corrections and strategy revision

### Corrections to the Session-1 inventory
- **The S3 bucket DOES resolve.** `aws s3 ls s3://apheris-oslo-hackathon/` works from this VM now
  (prefixes `data/`, `foundry-assistant/`, `rfd4-proteina/`). Session 1's `NoSuchBucket` was transient
  or a wrong-profile call. The "unreached zenodo datasets" blocker is **closed**.
- **`abguard` confirmed absent** — searched the whole bucket recursively for
  `abguard|guard|wheel|.whl|install`: nothing. Not on PATH, not in the images. Blocker 1 stands;
  it must be reimplemented or obtained from the organisers.
- R1 is much faster than estimated: 77/100 samples at 10:19, ~3.5/min, ETA **~10:26** (not 25–50 min).
  `--compile true` cost ~1 min of warm-up at 10:10–10:11 and paid off.

### New material found (`s3://apheris-oslo-hackathon/data/README.md`, updated 08:15 today)
- **`zenodo-10831512-trastuzumab-ngs/data.zip` (2.5 GB) — downloading now** (`logs/zenodo-dl.log`).
  Holds `data/affinity_data/her2/her2_aff_large/processed/remove_overlap/clonotype_split/0.7_0.15_0.15/`:
  the 524,346 FACS variants **pre-split 70/15/15 by clonotype**, so similar CDRH3s stay in one split
  and a model is tested on unseen sequence families. `label` = 1 for high binders. No leakage work for us.
  Also `shanehsazzadeh_et_al/processed/test.csv`: 198 CDRH3s with **continuous log-scale affinity**
  (6.2–9.0, higher = tighter), and `mason_et_al/` binder/non-binder splits. Also ships the authors'
  trained classifier weights (`.h5`, `.pkl`).
- **`sigma2-ab-ag-champloo/`** — Smorodina et al. Their headline finding, quoted from the README:
  *"a folding model's interface confidence (ipTM) often fails to tell a real pair from a shuffled one."*
  This is the evidence base for Risk 1 below.
- SPR table caveat now explicit: `no_reportable_kd` (460 rows) **is not a true negative**. Do not train
  on those as negatives.

### Lane B has antibody-specific tools RFD4 does not (`agent/nebius-scientific-ai/RECIPES.md`)
- **AntiFold** (recipe 2c, ~100 s): antibody inverse folding — keeps the framework, proposes new CDR
  loops, returns per-residue perplexity and per-sequence scores. Antibody-aware, no GPU, runs
  concurrently with Lane A. **This is a second, independent generator** and diversifies away from a
  single-model bet on RFD4.
- **ABodyBuilder3** (recipe 2b variant, headers `H`+`L`): antibody-specific structure model for paired
  Fv — better than a general folder for our format.
- Boltz2 co-fold (recipe 2) returns **no ipTM**; interface contacts must be counted from the mmCIF.

### Environment
- System python has no pip/numpy/sklearn/torch. Built `~/ubaydolla/her2/.venv` with `uv`
  (numpy, pandas, scikit-learn 1.9.1). ANARCI and abnumber are **not installable from PyPI** here.

### Decision D3 — rank on measured data, not on folding confidence
Axis 1 is binding, measured by SPR. The brief's step 3 (rank on `min_ipae`/`iptm`/`binder_plddt`) relies
on exactly the signal the organisers' own Smorodina dataset shows to be unreliable for discriminating
binders. **We will rank primarily on a scorer trained on the 524k clonotype-split FACS labels and the
190 SPR KDs / 198 log-affinities, and use the folding metrics as a structural sanity gate, not as the
ranking key.** Folding confidence still gates geometry and epitope (axis 2); it does not order the list.

### Open question Q1 — the CDRH3 identity threshold may need 4 mutations, not 3
The rule is CDRH3 identity **under 70%**. On 10 designable residues, 3 changes leaves 7/10 = **70.0%
identity, which is not under 70%** and would fail. Session 1's novelty note ("≥3 of 10 changed = ≤70%")
may therefore be off by one. It matters only if `abguard` computes identity over the 10 designable
residues; if it uses the CIF's 13-residue `cdr3` annotation (B97–109), 4/13 = 69.2% and 3 changes pass.
**Must be resolved before any submission** — it potentially invalidates every 3-mutation design.
Mitigation meanwhile: prefer candidates with ≥4 CDRH3 changes, which pass under either reading.

---

## Session 2 runs and results (10:27–10:35)

### R1 complete — RFD4 her2-novel, 100 designs
Started 10:10:38, ended 10:22:03 (**11.4 min**, not the 25–50 min estimated). `result.json` clean,
no `error.json` this time. 100/100 CIFs. Extracted CDRH3 with `extract_cdrh3.py`
(parses `_entity_poly` one-letter codes, slices VH between `YYCSR` and `YWGQGTLVTVSS`).
All 100 are length 10; **96 unique**. Mutation count vs trastuzumab `WGGDGFYAMD`:
2 muts × 2, 3 × 6, 4 × 4, 5 × 4, 6 × 17, 7 × 38, 8 × 26, 9 × 3. Mean ≈ 6.8.

### R2 — refold-prepare (CPU, seconds)
`runs/prod100/refold_prep/` → `holo/queries.json` (100 queries, 4 chains, `msa_path: null`),
`holo/template.cif`, `apo/`, and **`selection.csv`**, which already carries per-design
`hotspot_obedience`, `interface_hbonds`, `clash_fraction`, `pre_oracle_pass` — **no folding needed
for a first structural gate**. `hotspot_obedience` is a direct axis-2 (epitope) signal.

### R3 — Boltz-2 templated holo cofold (GPU), fired 10:29:39
```
module-run predict --input runs/prod100/refold_prep/holo/queries.json \
  --output runs/prod100/fold_holo --template-paths runs/prod100/refold_prep/holo/template.cif \
  --seeds 1 --num-diffusion-samples 1
```
Queries are in `selection.csv` rank order, so a partial run still yields the best candidates.
Rate **~7 structures/min** (31 by 10:34) → all 100 by ~10:44.
**Apheris Boltz-2 DOES return ipTM**: `interface_predicted_tm_score`, plus
`chain_predicted_tm_score` and `chain_chain_interface_predicted_tm_score` (so B↔C / B↔D binder-target
interface ipTM is available), `average_predicted_local_distance_difference_test`,
`global_predicted_distance_error`, `predicted_tm_score`, `model_ranking_score`.
The "no ipTM" note in the Nebius `RECIPES.md` applies to **Nebius's boltz2 adapter, not this one**.

### Decision D4 — abguard reimplementation: the off-by-one does not bind
Implemented both readings in `score_designs.py`. Of 100 designs, **73 pass** novelty
(Hamming ≥3 to every one of the 525,063 provided CDRH3s, computed exactly by radius-1 and radius-2
neighbour lookup, not approximated). Failures: 2 **identical** to a provided CDRH3, 6 at distance 1,
19 at distance 2. **`PASS_strict` == `PASS_lenient` == 73**: every design that clears the
provided-library gate already has ≥4 mutations from trastuzumab, so **Q1 (the 70%-identity
off-by-one) changes nothing** and is not a blocker. Worth confirming with organisers, not worth waiting on.
Note `novelty_filter.py` implements "≥2 differences from every provided", but the task text says
*more than 2* point mutations → **≥3**. We use ≥3 (strict). That file should be corrected.

### Axis-1 scorers — trained, externally validated (`models/`)
Trained on the **clonotype-split** FACS data (367,044 train / 78,652 val / 78,652 test; similar CDRH3s
confined to one split, so test families are unseen). One-hot 10×20.
| model | val AUC | test AUC | test AP |
|---|---|---|---|
| logistic regression | 0.964 | 0.944 | 0.903 |
| **HistGradientBoosting** | **0.995** | **0.991** | **0.983** |

**External validation on data the model never saw** — this is the load-bearing evidence:
- Shanehsazzadeh 198 designs: spearman(score, log-affinity) **ρ=0.297, p=2.1e-05**
- SPR 187 quantified KDs: spearman(score, −log10 KD) **ρ=0.353, p=7.2e-07**

A classifier trained only on **binary FACS labels** predicts real **SPR affinity** at p<1e-6.
Modest ρ, but grounded in measurement — unlike ipTM, which the organisers' own Smorodina data shows
is near-blind to whether a pair binds. This is the justification for **D3**.

SPR KD regressor (`models/kd_reg.pkl`), trained on 180 unique quantified 10-mers, y = −log10 KD:
ridge 5-fold CV **ρ=0.723**, MAE 0.563 log units; RF ρ=0.671. Trastuzumab reference: 8 rows,
KD 0.029–0.178 nM, mean y=1.021 (**0.095 nM — the bar for axis 1**).

### Finding F1 — no RFD4 design is predicted to beat trastuzumab, and this is structural
Of the 73 novelty-passing RFD4 designs: **0 beat trastuzumab** on the FACS classifier
(best 0.978 vs tz 0.9965) and **0** on the SPR regressor (best kd_rf −1.173 vs tz +0.127;
best predicted KD ≈15 nM vs measured 0.095 nM, ~150× worse).
Cause: novelty demands ≥3 mutations from all 525k measured sequences, and RFD4 satisfies it by
jumping **6–8** mutations from a clinically optimised loop. **Axis 1 is unwinnable from this pool,
so S-tier is unreachable on the original plan.** This is the single most consequential result so far.

### Finding F2 — the constrained shell search, and why its headline number is WRONG
The provided library covers only 5.25e5 of 10^13 possible 10-mers, so novelty-compliant sequences
exist at just **4** mutations from trastuzumab. Ridge-on-one-hot is **additive**, so
score(s) = intercept + Σ_p W[p, s_p] and the shell can be searched **exactly**, no sampling
(`shell_search.py`, 2,231,712 unique candidates over k=4 and k=5, CPU, ran alongside the GPU fold).
Per-position deltas (log10 KD units, + = tighter): pos4 D→Q **+0.583**, pos8 A→E **+0.398**,
pos3 G→V **+0.384**, pos10 D→I +0.179.
Best additive score **1.310 vs trastuzumab −0.287** — apparently 1.6 log units tighter.

**That headline does not survive cross-checking, and must not be reported as a result.**
Against two models that did not share the ridge's fit, over the 400 novelty-passing shell candidates:
- spearman(ridge, FACS HistGB) = **−0.269** — *anti*-correlated
- spearman(ridge, SPR RF) = **+0.031** — zero, **though ridge and RF were trained on the same 180 rows**
- candidates beating trastuzumab on **all three** models: **0**

Two models that agree at ρ=0.72 in-distribution diverge completely in the region the argmax search
walked into. This is scorer-adversarial extrapolation: optimising the model, not the molecule.
**Rule adopted: a candidate is only promoted if independent models agree. Single-model argmax is never
evidence.** The honest statement is that we have **no computational evidence that any
novelty-compliant design beats 0.095 nM**.

### Finding F3 — the shell search still beats RFD4 by ~1.5 log units
| pool | best SPR `kd_rf` (tz = +0.127) | best FACS (tz = 0.9965) |
|---|---|---|
| RFD4, 73 novelty-passing | **−1.173** | 0.978 |
| shell search, 400 novelty-passing | **+0.322** | 0.891 |

**13 of 400 shell candidates beat trastuzumab on the nonlinear SPR model**; 0 RFD4 designs do.
Requiring `facs > 0.9965` is an unfair gate — trastuzumab is the archetypal high binder in a
saturating classifier, so nothing can exceed it; FACS is used as a **floor** (>0.8), not a ceiling.
Under that gate the consensus leader is **`WGSQGFYEME`, 4 mutations** (ridge 1.089, kd_rf 0.322 > tz,
FACS 0.837): tighter than trastuzumab on both SPR-trained models, confidently a high binder on
the 367k-row FACS model. Consensus ranking = mean percentile of the three models.
These, not the RFD4 pool, are our axis-1 candidates.

### R4 — shell-candidate fold queries built, ready to fire when R3 frees the GPU
`runs/shell_queries.json`: top 40 consensus shell candidates on the trastuzumab framework
(VH = `EVQ…YYCSR` + CDRH3 + `YWGQGTLVTVSS`, 120 aa; VL, HER2 C/D chains and residue_ids identical
to R3), **plus `BASELINE_trastuzumab` as an in-run control** so axis-1/2 comparisons stay inside one
folding model, per the provenance rule. Same `template.cif`.

### R3 harvest + Finding F4 — D3 was partly wrong, corrected here
All 100 folded by 10:42 (fired 10:29:39, **12.5 min**, ~8/min). `runs/prod100/fold_metrics.csv`.
Pool stats: pLDDT 91.7–96.9 (mean 94.4), pTM 0.745–0.881, ipTM 0.630–0.826,
iptm_B_C 0.288–0.675, iptm_B_D 0.251–0.630.

Tested the folding metrics against our empirically-grounded scorers, n=100:
| folding metric | vs SPR-model affinity (kd_rf) | vs FACS classifier |
|---|---|---|
| iptm_B_D | **+0.343 (p=0.00047)** | +0.107 (ns) |
| iptm_B_C | **+0.330 (p=0.0008)** | +0.092 (ns) |
| ipTM (global) | +0.297 (p=0.0027) | +0.100 (ns) |
| pTM | +0.273 (p=0.0059) | +0.084 (ns) |
| ranking score | +0.211 (p=0.035) | +0.102 (ns) |
| **pLDDT** | **−0.013 (p=0.90)** | +0.073 (ns) |

**Correction to D3.** I claimed ipTM was "near-blind" to binding, citing Smorodina. On *our* designs it
is not: ρ=0.33–0.34 at p<0.001, **the same magnitude as our empirical model's own correlation with real
SPR (0.35)**. Smorodina's result is about telling cognate from *shuffled* pairs — a different, much
coarser task than ranking 100 variants of one real complex on a fixed epitope. Over-generalising it was
my error. ipTM carries genuine, partly independent signal here.
**Revised D3: rank on a consensus of interface ipTM + the two empirical models. Drop pLDDT entirely
(ρ=−0.013, noise).** The brief's step 3 was right about iptm/min_ipae and wrong about binder_plddt.
Note ipTM tracks the *affinity* regressor but not the *FACS binary* classifier — consistent with SPR
affinity and interface quality both being continuous physical quantities, while FACS high/low is
avidity-driven. The two scorer families are measuring different things; keep both.

Generation-time vs fold-time interface agreement is weak: spearman(hotspot_obedience, iptm_binder_target)
= +0.193 (p=0.054), spearman(interface_hbonds, iptm_binder_target) = +0.130 (p=0.20). RFD4's own
structural gate and Boltz-2 largely disagree, so neither alone should gate.

### Finding F5 — the best binders in the pool are exactly what the rules forbid
Top 2 of 100 by predicted affinity: **`WGSSGFYAMD`** (kd_rf +0.356) and **`WGYSGFYAMD`** (+0.324) —
both *above* trastuzumab's +0.127 — and they rank **3rd and 5th of 100 on ipTM** too, so the structural
and empirical methods independently agree they are the pool's best. **Both FAIL novelty**: only
**2** CDRH3 mutations (pos3 G→S, pos4 D→S / pos3 G→Y, pos4 D→S), and the rule needs >2.
This is the challenge's central tension, demonstrated rather than asserted: the designs most likely to
bind HER2 well are precisely the ones the optimisation-track filter excludes. Any honest pitch has to
say that the filter and axis 1 pull against each other, and that our shell search is an attempt to find
the *least-damaging* novelty-compliant perturbation rather than to pretend the tension is absent.
Both are logged as **rejected rows with reasons**, per the provenance rule — not silently dropped.

### R5 — shell fold fired 10:43:10
41 queries (40 consensus shell candidates + `BASELINE_trastuzumab`), same template, seeds 1,
1 diffusion sample. ETA ~6 min at the observed 8/min.

---

## Session 3 (11:00–) — shell folds harvested, lane B second opinions, final ranking

### Inventory corrections (again)
- **The H100 is NOT ours alone.** At 11:00 the GPU was at 95% util running
  `rfd4-proteina-her2-2026-11-07` with `num_replicates=80` into `/work/runs/opt/design`.
  `docker inspect` shows the mount is `/home/hackathon/lachlan/rfd4-docs` — a **teammate's**
  container, not ours. Lane A is **contended**, not dedicated. All remaining GPU work must assume
  a queue behind someone else. This is why Session 3 put everything on lane B.
- `forge.py` has no `download`; the verb is **`fetch --dest DIR`**.
- stab-ddg's entry semantic type is **`stab-ddg-workload-tar/v1`** (not `protein-binding-ddg-tar/v1`;
  that was rejected with "workload semantic type is invalid").

### R5 harvest — the first real axis-1 result, with an in-run control
`runs/shell_fold_metrics.csv`, 41/41 folded (fired 10:43:10, done 10:48:45, 5.6 min).
The in-run `BASELINE_trastuzumab` fold is the comparator, so everything stays inside one folding model.

| | baseline tz | shell best | n beating baseline |
|---|---|---|---|
| `iptm_binder_target` | 0.576 | **0.728** (shell_20) | **12/40** |
| `iptm` (global) | 0.769 | 0.852 | 14/40 |
| `ranking` | 0.908 | 0.914 | 1/40 |
| `plddt` | 94.31 | 94.89 | 2/40 |

12 of 40 novelty-compliant 4-mutation designs beat the trastuzumab control on binder–target
interface ipTM. Combined with F3 (13/400 beat it on the SPR model), two methods independently
say the constrained shell contains designs better than trastuzumab. pLDDT again carries nothing
(2/40, and F4 measured ρ=−0.013) — the brief's `binder_plddt` ranking key stays dropped.

### Axis 2 now measured, not assumed — `epitope.py` → `results/epitope_shell.csv`
Axis 2 is judged by competitive binding against trastuzumab, so the question is not "does it bind"
but "does it touch the same HER2 residues". Per design, from its own folded complex:
`epitope_jaccard` = Jaccard overlap of the contacted-HER2-residue set (<5 Å heavy atom) with the
**in-run trastuzumab fold**, plus retention of the 4 conditioned `C_HOT` atoms.
Result: the baseline contacts 21 HER2 residues; the top designs retain **4/4 hotspot contacts** and
`epitope_jaccard` **0.82–0.95**. Epitope preservation is in good shape and is our strongest axis —
expected, since we only ever changed 4 of 10 loop positions and `resample_hotspots` stayed false.

### Axis 3 — what we can and cannot claim (closes Blocker 2 honestly)
No Tm predictor exists on either lane. We did not invent one. Instead, `developability.py`:

The whole axis-3 argument is **construction, not prediction**. Every shipped design differs from
trastuzumab at only 4–5 of 10 CDRH3 positions; the VH framework, the entire VL, the VH/VL packing
and both disulfides are byte-identical to the clinical molecule. Fv Tm is dominated by framework and
interface packing, so these designs are **near-isothermal with trastuzumab by construction**. What we
changed is a surface loop, so what we score is the loop's own contribution: aggregation propensity,
chemical-degradation liabilities, charge/solubility. All deterministic functions of the 10-mer —
no fitting, so none of F2's scorer-adversarial failure mode applies.

**Finding F6 — our designs improve on a real trastuzumab liability, and this was free.**
Trastuzumab's own CDRH3 `WGGDGFYAMD` carries an **Asp102–Gly103 isomerisation motif** (a documented
antibody degradation liability) plus Met107 and Trp99 oxidation sites. The single most affinity-
improving position in the shell search is **D102**, so *every* top design replaces it (D102→Q/E/Y)
and thereby **deletes the DG isomerisation motif** (`d_liabilities_vs_tz = −1`). Axis 3 improves on a
named chemical liability as a side effect of the axis-1 optimum. This is the strongest S-tier argument
we have, and it needed no predictor.
Internal trade-off, quantified: position 108 splits the field. **D108I** raises loop hydrophobicity
(Δagg proxy only −0.08) while **D108E** lowers it sharply (Δagg **−1.06**, i.e. markedly less
aggregation-prone than trastuzumab). So the axis-1 leader and the axis-3 leader are different
molecules, and the ranking has to choose. See D5.

### Lane B second opinions
| run | op id | result |
|---|---|---|
| ESM-2 650M PLL, 61 VH seqs | `op-07b3495e1f4d104a` | succeeded |
| **StaB-ddG binding ddG, 42 variants** | `op-79006e6f713984a7` | succeeded, 12 min |
| AntiFold, 41 Fv backbones | `op-c6966422ac4e4105` | **FAILED** — see below |
| AntiFold retry, `skip_anarcii_numbering` | `op-441067223204e140` | running |

- **ESM-2 PLL**: every shell variant sits slightly *below* trastuzumab (mean −0.988 → −1.003/−1.028).
  Expected when mutating away from a clinically optimised sequence; used as a weak naturalness
  penalty, never as a ranking key.
- **AntiFold failure, and what it tells us**: `ValueError: No PDBs left to run: ANARCII could not use
  the chains of any of the 41 given`. Resubmitted once with `skip_anarcii_numbering: true`.
  Worth recording for Blocker 1: **AntiFold bundles ANARCII**, so it is a route to the immunum
  numbering-confidence check abguard wants — but it could not number our Boltz-2 output chains, so
  it is not a route we can rely on today.
- **StaB-ddG** (`results/stabddg_scores.csv`), binding ddG vs the trastuzumab complex, with per-
  variant Monte-Carlo SE. Negative = stabilises the complex = tighter. Of 40 shell designs:
  **18 stabilising beyond 2 SE, 17 indistinguishable from trastuzumab, 5 destabilising.**
  The headline: a physics-based ddG method that shares no fit with any of our ML scorers says the
  4-mutation novelty jump **does not cost binding**. That is the single most reassuring number today,
  because F1 showed the RFD4 pool lost ~150× affinity to buy novelty.
  Calibration note: the two rule-excluded 2-mutation designs from F5 were deliberately included as
  reference points (`REF_excl_*`) and came out −0.13 / −0.43, confirming the sign convention against
  molecules our SPR model independently calls the pool's best binders.

### Finding F7 — a fourth method disagrees with the other three, and this reshaped the ranking
On the 40-design shortlist:
| pair | spearman | p |
|---|---|---|
| StaB-ddG vs SPR RF (`kd_rf`) | **−0.405** | 0.009 |
| StaB-ddG vs Boltz-2 `iptm` | **−0.338** | 0.033 |
| StaB-ddG vs `facs_histgb` | +0.103 | 0.53 |
| Boltz-2 ipTM vs `kd_rf` (F4) | +0.343 | <0.001 |

ipTM and the SPR model agree with each other; StaB-ddG **anti-correlates with both**. Two pairs agree,
two anti-correlate. This is F2's lesson recurring on the actual shipping list: in the 4-mutation shell,
independent scorers stop agreeing, and **any single ranking is therefore untrustworthy** — including
a mean of percentiles, which would silently average the disagreement away.

**Decision D5 — rank on unanimity of sign, not on a score.**
The primary key is `axis1_n_beat_baseline`: how many of five independent axis-1 scorers
(`kd_rf`, `facs_histgb`, `ridge_score`, `iptm_binder_target`, `stabddg`) place the design above
trastuzumab. Secondary key is the **worst** of the three normalised axes, because S-tier requires
improving all three and a design that is excellent on one axis and poor on another is explicitly not
what the judging rewards. The percentile mean is a tie-break only. `rank.py` implements this.
Result: **9 designs beat trastuzumab on 4 of 5 scorers; none on 5/5.** We do not claim 5/5.

### Gates re-verified from scratch — `verify_gates.py` → `results/gate_verification.csv`
Deliberately not reusing the shell-search filter: a filter applied by the generator is not evidence
that the shipped molecule passes. Recomputed exact min Hamming distance to all **525,063** provided
CDRH3s by enumerating radius 0/1/2 (exact, not sampled).
**40/40 shipped designs pass**: `min_dist_provided` = 3 for all 40 (so >2 point mutations from every
provided antibody), 4–5 mutations from trastuzumab (**≤60% CDRH3 identity — passes under either
reading of the 70% rule, so Q1 is now moot for the shipped list**), scFv 227 aa ≤ 250, no unpaired Cys.
The baseline correctly fails both novelty gates, which is the control working.
**Declared gap, not hidden**: immunum numbering confidence ≥0.1 is the one abguard check we cannot
reproduce — it needs ANARCI/ANARCII, neither installs here, and AntiFold's bundled copy could not
number our chains. Every design is a 227-aa scFv on an unmodified clinical framework, so the risk of
failing a numbering check is low, but it is unverified and must be stated as such.

### Where this leaves the three axes
- **Axis 2 (epitope): strong.** Measured, 4/4 hotspots, 0.82–0.95 overlap vs an in-run control.
- **Axis 3 (thermostability): defensible, not predicted.** Identical framework + a deleted
  isomerisation motif + lower aggregation proxy. No Tm number, and we say so.
- **Axis 1 (binding): genuinely uncertain and the honest weak point.** 9/40 designs beat trastuzumab
  on 4 of 5 independent scorers, and StaB-ddG says novelty costs no binding — but F7 shows those
  scorers disagree with each other in exactly this region, and trastuzumab at 0.095 nM is an
  extraordinarily high bar. We have **convergent but not unanimous** computational evidence.

---

## Deliverables written (11:15)
| file | what |
|---|---|
| `results/submission_top20.csv` | the ranked list, 29 columns, every scorer + every gate |
| `results/submission_top20.fasta` | the 20 scFv sequences as submitted (242 aa, `(G4S)3` linker) |
| `results/summary.html` | one-page story, rendered from the CSVs — open in a browser |
| `results/all_designs_scored.csv` | all 41 incl. the control, every column |
| `results/rejected_designs.csv` | rejection ledger with reasons |
| `results/gate_verification.csv` | independent gate re-derivation, per design |
| `results/{epitope_shell,stabddg_scores,antifold_scores}.csv` | per-axis raw tables |
| `README.md` | the argument, exact reproduce commands, honest limitations |

Repo consolidated: `NOTES.md` and `results/` now live **inside** the git repo at
`~/ubaydolla/her2/`, with symlinks left at `~/ubaydolla/NOTES.md` and `~/ubaydolla/results`
so the old paths still open. Committed `ca82b03`. Raw lane-B artifacts (21 MB of hash-named
blobs) are gitignored; the op-id run records and the derived CSVs are committed, so any
number can be re-fetched with `forge.py fetch <op-id> --dest <dir>`.

**No GitHub remote exists yet and `gh` is not installed on this VM.** The repo is committed
locally only. The VM is wiped tonight, so this is the one outstanding blocker on the
"reproducible repo" deliverable (20% of the pitch score) and it needs a human with
credentials. Asked the user at 11:25.

---

## RISK 1 (the single biggest risk) — the 20 designs are not 20 independent bets

Measured on the shipped list:
- **All 20** mutate the *same three* CDRH3 positions: **101, 102, 106**.
- **19 of 20** carry the *identical* substitution **A106E**.
- Only **4 distinct position-sets** across the whole list.

Every design descends from one generator — the additive-ridge shell enumeration — whose
per-position weights were fitted on 180 quantified SPR K<sub>D</sub>s. If the ridge's
preference for A106E (or for touching 101/102) is an artifact of that small training set,
**all 20 designs fail together**. The list looks like 20 shots on goal; it is closer to one
or four, replicated. This is exactly the failure mode F2 warned about, surviving into the
deliverable in a form the ranking cannot see, because ranking within a correlated pool
cannot detect that the pool is correlated.

Adaptyv measures a handful of molecules. Spending 20 slots on ~4 genuinely distinct
hypotheses is the most expensive mistake available to us today.

### Mitigation (next block of work)
1. **A second, independent generator.** AntiFold (lane B, ~110 s, no GPU) proposes CDR loops
   from the *backbone* rather than from our SPR fit — a genuinely different inductive bias.
   Its per-residue logits are already in hand from `op-441067223204e140`; sample CDRH3s from
   them and gate them through `verify_gates.py` unchanged.
2. **Force positional diversity into the submitted 20.** Add a constraint so the list spans
   distinct position-sets (e.g. at most 3 designs per position-set), accepting a lower
   consensus score in exchange for decorrelated risk. A 20-design list where the top 4 are
   the current leaders and the rest probe different positions dominates the current list
   under any scoring that rewards at least one S-tier hit.
3. **Re-include the RFD4 pool as the diversity tail.** F1 says those designs are unlikely to
   beat trastuzumab, but they are mechanistically independent of the ridge, and the measured
   ipTM/epitope numbers for all 100 already exist. A few as a hedge costs nothing.

---

## Timeline for the remaining hours (set 11:25, ~3.5 h to go)
| time | work | lane | blocking? |
|---|---|---|---|
| 11:25–11:35 | ask user re GitHub remote; sample CDRH3s from AntiFold logits | B / CPU | needs user for push |
| 11:35–12:05 | gate AntiFold + RFD4 candidates; build diversified fold queries | CPU | no |
| 12:05–12:35 | Boltz-2 fold the diversified set, same template + in-run control | A (GPU) | **teammate holds GPU** |
| 12:05–12:35 | StaB-ddG on the diversified set (concurrent) | B | no |
| 12:35–13:10 | harvest, epitope analysis, re-rank with the diversity constraint | CPU | no |
| 13:10–13:40 | regenerate all deliverables; final gate re-verification | CPU | no |
| 13:40–14:20 | pitch narrative; push repo | — | needs user |
| 14:20–15:00 | buffer (RFD4 reruns, ESM-C second opinion if time) | — | — |

GPU contention is the schedule's weak point: the 12:05 fold is the only step that needs the
H100 and a teammate's job has held it since before 11:00. Fallback if it is still busy:
fold on lane B instead (`boltz2` or `openfold3`, no GPU of ours), accepting that lane-B
Boltz-2 returns no ipTM so epitope contacts must be counted from the mmCIF — `epitope.py`
already does exactly that and does not depend on ipTM.

---

## Session 4 (11:29–) — RISK 1 mitigation: a genuinely independent generator

### AntiFold retry succeeded — and its region labels are wrong, harmlessly
`op-441067223204e140` (`skip_anarcii_numbering: true`, `custom_chain_mode: true`) **succeeded**.
Fetched to `nebius-runs/antifold-skipnum-out/`; the blobs are hash-named, so the result manifest's
per-file sha256 was used to restore real filenames into `.../named/` (83 files, all matched).

Because ANARCII numbering was skipped, AntiFold's `assumed_region` column is **wrong** — it labels
chain B CDRH3 as `FWL3`/`CDRL3`, and its own sampled sequence (`seq1`) consequently left CDRH3
untouched and mutated framework positions instead. **We never use that column.** Positions come from
`pdb_posins`, which is correct: chain B residues 99–108 read `WGGDGFYAMD`, asserted in code.
So AntiFold's *sampling* output is unusable here but its **per-residue log-prob matrix is exactly
what we need**, and it is read directly.

### Finding F8 — the two generators want almost disjoint positions
`antifold_generate.py`. AntiFold one-shot scoring is additive over positions, like the ridge, so the
k=4/k=5 shells (2,231,712 unique sequences) were enumerated **exactly** with AntiFold's matrix in
place of the ridge's — same machinery, different weights, no sampling.

| | ridge (180 SPR K<sub>D</sub>s) | AntiFold (Fv backbone, no antigen, never saw our data) |
|---|---|---|
| positions it wants | **101, 102, 106** | **99, 102, 103, 104, 107** |
| top substitutions | D102→Q, A106E, G101V | G103→Y (+6.0 nats), F104→G, D102→I |

Only position **102** is shared — both methods independently want trastuzumab's Asp102 changed, which
is the same residue whose removal deletes the DG isomerisation motif (F6). That is real
cross-method agreement and it strengthens F6.

Otherwise the overlap is nil: of AntiFold's top 20 novelty-compliant candidates, **every one** scores
~2 log units *worse* than trastuzumab on the SPR RF (`kd_rf` ≈ −2.0 vs tz +0.127) and below it on
FACS. `n_beat_tz_of4` = 1 for all of them — the only scorer that likes them is AntiFold itself.

**So F2's rule cuts both ways.** AntiFold-argmax is scorer-adversarial extrapolation in exactly the
way ridge-argmax was. Neither generator's ranking can be the shipping criterion. This is a cleaner
statement of F2 than F2 itself: it is not that *our* model was overfitted, it is that **any single
generator's argmax is untrustworthy**, and we now have two independent demonstrations.

### Finding F9 — the intersection is non-empty but costs the entire affinity edge
`joint_search.py`. Both models are additive, so the set where **both** beat trastuzumab can be
enumerated exactly rather than sampled. Alternatives per position = union of each model's top 6
(8–11 per position, so neither model's blind spot is excluded); shells k=4 and k=5;
**22,004,248 sequences enumerated.**

| stage | n |
|---|---|
| beat tz on **both** additive models (ridge AND AntiFold) | **731** |
| …and pass the FACS floor >0.8 | 228 |
| …**and** beat tz on the nonlinear SPR RF | **0** |
| novelty-compliant (`verify_gates.check`, unchanged) | 113 |
| after diversity cap (≤3 per position-set) | **39**, spanning **15 distinct position-sets** |

**Zero of 731.** Requiring AntiFold's agreement removes the SPR-model advantage completely — the best
joint candidate is `kd_rf` −0.19 against trastuzumab's +0.127.

But note the magnitude: **−0.3 log units, not the −1.5 the RFD4 pool cost (F1).** So decorrelated
designs are cheap here, not free. The shipped list's 4 position-sets become 15, and the cost is about
a third of a log unit of *predicted* affinity from a model we already know (F7) disagrees with
physics-based ddG in this very region. That is a trade worth making for part of the list, not all of it.

Representative joint candidates (4 mutations, all novelty-compliant):
`WGDIYFYSMD` (101+102+103+106), `EGDEYFYAMD` (99+101+102+103), `WGGNYFHEMD` (102+103+105+106),
`WYGEYWYAMD` (100+102+103+104) — four genuinely different hypotheses, none sharing A106E.

### R6 — StaB-ddG on the diversified pool, fired 11:36 (`op-2a778c6911ff6624`)
42 variants = the 39 diversity-capped joint candidates + **3 AntiFold-argmax sequences as declared
negative controls** (`CTRL_afargmax_*`). Rationale: StaB-ddG is physics-based, shares no fit with any
ML scorer, and F7 measured it *anti*-correlating with `kd_rf` (ρ=−0.405) — the exact axis on which the
joint pool looks bad. It is therefore the one method that can independently support or refute the
diversified tail, and it runs on lane B with no GPU contention. `submit_stabddg_joint.py`,
manifest in `nebius-runs/stabddg_joint_manifest.csv`.

### Lane A is still contended
At 11:36 the H100 is at 95% util under a **new** teammate container
(`rfd4-proteina-her2-2026-11-07`, up 4 min). Only 4 GB of 80 GB VRAM is in use, so our 41-structure
Boltz-2 fold would fit alongside it memory-wise, but it would slow their job down. **Not doing that
unilaterally — asked the user at 11:38.** Fallback remains lane B folding (`epitope.py` does not
depend on ipTM and counts contacts from the mmCIF).

### R6 harvest — Finding F10: StaB-ddG REFUTES the AntiFold diversification
`op-2a778c6911ff6624` succeeded in ~11 min. `results/stabddg_joint.csv`,
39 candidates + 3 controls. Negative ddG = stabilises the complex = tighter.

| pool | stabilising (>2 SE) | indistinguishable | destabilising (>2 SE) |
|---|---|---|---|
| original shell list (Session 3, n=40) | **18** | 17 | 5 |
| AntiFold-intersection pool (n=39) | **2** | 12 | **25** |

Median ddG +0.41 kcal/mol, range −0.50 to +1.46. The three AntiFold-argmax negative
controls came out +0.26 / +0.38 / +0.41 — destabilising, as the other three scorers predicted.

So the joint pool is rejected by a **physics-based method that shares no fit with any ML scorer**,
on top of being rejected by the nonlinear SPR RF (0 of 731, F9). Two independent lines of evidence.

**Decision D6 — abandon AntiFold-intersection as the diversification route.** Mitigation path 1
in RISK 1 is not merely unhelpful, it is **refuted**: the sequences AntiFold endorses are worse on
the SPR model, worse on physics-based binding ddG, and worse on FACS. Decorrelating the portfolio by
switching generators costs real predicted *and* physical binding quality. Recorded as a negative
result, with the data, rather than quietly dropped — and it means the shipped list's correlation
(RISK 1) is a risk we currently **cannot** buy our way out of by this route.

### Finding F11 — the right seed was in the measured data all along
The user asked for the successful sequences in the data. Of the 182 unique CDRH3s with a quantified
SPR K<sub>D</sub>, **6 are tighter than trastuzumab's median (0.135 nM, 8 replicates)**:

| CDRH3 | KD (nM) | mutations vs trastuzumab |
|---|---|---|
| **`WGSQGFYAYD`** | **0.0234** | **G101S, D102Q, M107Y** |
| `WGSDGWYAMD` | 0.0608 | G101S, F104W |
| `WGGQGFYAMD` | 0.0900 | D102Q |
| `WGGSGPYLFD` | 0.0950 | D102S, F104P, A106L, M107F |
| `WGGEGFYAMD` | 0.1170 | D102E |
| `WGGDGFYLMD` | 0.1280 | A106L |

Recurring substitutions: **G101S ×2, D102Q ×2, A106L ×2**. The ridge independently ranked D102Q its
single best substitution (+0.583) and position 101 in its top three, so model and measurement agree
on 2 of the champion's 3 positions. *Consistency check, not independent validation* — these rows are
in the ridge's 180-row training set. The **measurement** is what is independent.

All 6 fail novelty (`min_dist_provided` = 0: they are *in* the provided library). They cannot be
submitted. Their value is as a **seed**.

### Finding F12 — champion-seeded shell beats everything we have shipped
`champion_shell.py`. Every search so far expanded around trastuzumab (0.135 nM measured). This one
expands around `WGSQGFYAYD` (**0.0234 nM measured**, 5.8× tighter), holding its three winning
substitutions **fixed** and enumerating all 1–3 further mutations across the other seven positions
(247,779 sequences, exact).

| stage | n |
|---|---|
| ≥4 mutations from trastuzumab (strict identity rule) | 247,779 |
| beat trastuzumab on the nonlinear SPR RF | 7,999 |
| …and clear the FACS floor >0.8 | 240 |
| **…and novelty-compliant (`verify_gates.check`, unchanged)** | **56** |
| …and beat the measured champion itself on the SPR RF | 0 |

| pool | best `kd_rf` (trastuzumab = +0.127) |
|---|---|
| RFD4, 73 novelty-passing (F1) | −1.173 |
| AntiFold intersection, 113 (F9) | −0.189 |
| original shell list, 400 (F3) | +0.322 |
| **champion-seeded, 56** | **+0.427** |

Best three: `WGSQFFYHYE`, `WGSQPFYHYE`, `WGSQRFYHYE` (6 muts vs tz, `min_dist_provided` = 3,
extra substitutions G103F/P/R + A106H + D108E on top of the champion's three). Note the identical
`kd_rf` 0.42653 across all three — the RF is **tied/saturating** on these, so the ranking among them
is not real and must not be presented as if it were.

**Caveat to carry forward:** the diversity cap collapsed to 3 designs over 1 position-set, because
the top-ranked 56 concentrate in very few position-sets. So this pool is **even more correlated**
than the shipped list — RISK 1 gets *worse*, not better, on this route. The honest framing is that
axis 1 and portfolio decorrelation are in direct conflict, now measured three separate ways.
