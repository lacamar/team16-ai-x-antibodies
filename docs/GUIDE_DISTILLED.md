# Field guide → pipeline (distilled)

Only the parts of the participant guide that drive **reproducible** pipeline steps. Each item is tagged with the gate it feeds. Models/metrics/tools appear once. Pin every model id + seed; cache every call.

Source: `oslo-ai-antibodies-hackathon.vercel.app/participant_guide`.

---

## 1. Models by pipeline stage (pick one per gate, pin the version)

| Gate | Job | Candidates (choose + pin one) |
|---|---|---|
| `b10_generate` | de novo binder (structure) | RFD4-Proteina, RFdiffusion3, RFantibody, IgDiff, PXDesign, BoltzGen |
| `b10_generate` | sequence for a backbone (inverse fold) | ProteinMPNN, ESM-IF1, AntiFold |
| `b20_fold` | fold / co-fold → structure + confidence | AlphaFold2/3, **OpenFold (Apheris — prize)**, Boltz-2, Chai-1, ESMFold-2, NanoBodyBuilder2, ABodyBuilder3 |
| `b30_interface` | predict KD (regression) | AF-IG2, OpenDDE, related affinity models — **ranking only, not a KD value** |
| (encoding, if training a model) | sequence → embedding | ESMC, SaProt, ESM-2, nanoBERT, IgBERT (one-hot often fine) |
| `b40_developability` | VHH liability flags | TNP (nanobodies); TAP (conventional) |

A backbone-only generator (diffusion) **must** be followed by an inverse-fold step before you have a real sequence. Refold every generated design with a folder for an independent confidence score.

## 2. Score metrics — for gate thresholds

| Metric | Scope | Range | Better | Means |
|---|---|---|---|---|
| pLDDT | per residue | 0–100 | higher | local structure confidence |
| PAE | residue pair | Å | lower | error in relative positioning |
| ipTM | complex | 0–1 | higher | confidence in chain placement |
| ipSAE | interface | 0–1 | higher | interface-focused (impl. varies — pin one) |

Use these to **rank and filter**, never as affinity. A high interface score ≠ measured binding: also inspect contacts, clashes, and whether the intended epitope is hit.

## 3. Developability as programmatic checks (`b40`)

**Liability motifs = string patterns → fully deterministic, the strongest reproducible gate.** Regex over the CDRs (and flag, don't auto-reject — relevance is pH/formulation-dependent):

| Liability | Pattern | Note |
|---|---|---|
| N-glycosylation sequon | `N[^P][ST]` | |
| Asn deamidation | `N[GSTH]` | `NG` highest risk |
| Asp isomerization | `D[GSTDH]` | `DG` highest risk |
| Fragmentation | `DP` | |
| Oxidation | `M`, `W` in a CDR | context-dependent |
| Unpaired cysteine | odd count of `C` / free Cys | |
| Hydrophobic patch | run of `[FILVWY]` | aggregation risk |
| Charge patch | run of `[KR]` | |

**Model-based developability** (pin versions): stability → ThermoMPNN, ESM-IF; solubility/aggregation → Aggrescan3D, SoluProt, NetSolP, ProteinMPNN (soluble weights); immunogenicity → germline proximity (closer = safer). All are predictions — flags, not truths.

## 4. Numbering — the canonical step before any sequence analysis (`b10`/`b40`)

Run **ANARCI/ANARCII** to assign IMGT (or Kabat/Chothia) numbering; a VHH is numbered like a heavy-chain variable domain. This is what makes "CDR3" / "position 104" mean the same thing across designs. **IMGT positions are NOT PDB residue numbers** — map explicitly; never assume they line up.

## 5. Databases — references, controls, training (`t*` and `b*`)

| Data | Use | For |
|---|---|---|
| SAbDab / SAbDab-nano | curated antibody/VHH structures | templates, controls |
| Observed Antibody Space (OAS) | repertoire sequences | germline/naturalness baselines |
| SKEMPI 2.0 | ΔΔG on mutation | train/validate affinity models |
| AB-Bind | antibody mutational binding | train/validate |
| Adaptyv competition data | prospective wet-lab on designs | expression/binding priors (the lab testing yours) |
| Ginkgo Datapoints | developability (incl. VHH) | developability baselines |

## 6. Pitfalls → encode as gate guards

- **Confidence ≠ affinity.** pLDDT/ipTM/PAE say the model is sure about *geometry*, not that it *binds*. Rank, don't trust.
- **A predicted complex ≠ binding.** Co-fold/docking correlates weakly and is unreliable for novel epitopes. Treat as a filter, not proof.
- **KD is only real if measured** (SPR/BLI). No universally good predictor. Quote predicted KD as rank, never as a number you believe.
- **Affinity ranking is weak** (classical ~Pearson 0.3–0.5). Prefer rank-order over absolute values; add positive/negative/baseline controls to every scoring gate.
- **Numbering ≠ PDB numbering.** Always map via ANARCI.
- **No brute force.** CDR3 space is ~20^15. Start wide, filter hard, keep provenance of what you dropped.
- **Pin + cache everything.** Model ids, seeds, DB snapshots — so a rerun is identical and offline.

## 7. Worked generation recipe (reproducible) — `b10`

Practice target **PDB 1ZVY** (VHH = chain A, lysozyme = chain B). JapanFold → Drug Design → RFD4-Proteina, paste mmCIF, set the contig:

- New binder to the antigen: `B1-129, /0, 110-130` (target chain B, `/0` chain break, a 110–130-residue new binder — not necessarily a VHH).
- Redesign a VHH loop: `B1-129, /0, A2-96, 20, A117-125` (keep antigen + VHH segments A2–96 and A117–125, regenerate a 20-residue loop between them).

Verify residue numbering against the downloaded structure before copying contigs; unresolved residues (e.g. A1 missing in 1ZVY) shift the indices.
