"""Cheap sequence-liability developability screen for CDRH3 loops.

The hackathon write-up recommends scoring candidates for expressibility/developability
before ranking (Aggrescan3D / SoluProt / NetSolP / soluble ProteinMPNN are suggested but
are not installed on this VM). As an on-VM, no-dependency stand-in we score the standard
antibody CDR *sequence liabilities* -- well-established motifs that hurt developability:

  * N-linked glycosylation sequon  N-X-[S/T] (X != P)        -- hard liability
  * Asn deamidation hotspots       N-[G/S/T] (esp. NG)       -- soft
  * Asp isomerisation hotspots     D-[G/S/T] (esp. DG)       -- soft
  * Free cysteine                  C                         -- hard (also gated in eligibility)
  * Met / Trp oxidation            M, W                      -- soft, count
  * Hydrophobicity / aromatic load fraction of {F,W,Y,I,L,V} -- soft, aggregation proxy
  * Low-complexity runs            >=4 identical residues     -- soft

These are liabilities of the *CDRH3 span only* (the framework is trastuzumab's, already a
validated therapeutic). ``liability_report`` returns per-sequence counts and a hard-pass
mask (no glycosylation sequon, no Cys) plus a soft penalty for ranking/tie-breaking.
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

HYDROPHOBIC_AROMATIC = set("FWYILV")

_NGLYC = re.compile(r"N[^P][ST]")
_DEAMID = re.compile(r"N[GST]")
_ISO = re.compile(r"D[GST]")


def _runs_ge(seq: str, k: int) -> int:
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return int(best >= k)


def liability_report(seqs: List[str]) -> Dict[str, np.ndarray]:
    n_glyc, deamid, iso, cys, oxid, aromfrac, lowcomplex = ([] for _ in range(7))
    for s in seqs:
        n_glyc.append(len(_NGLYC.findall(s)))
        deamid.append(len(_DEAMID.findall(s)))
        iso.append(len(_ISO.findall(s)))
        cys.append(s.count("C"))
        oxid.append(s.count("M") + s.count("W"))
        aromfrac.append(sum(c in HYDROPHOBIC_AROMATIC for c in s) / len(s))
        lowcomplex.append(_runs_ge(s, 4))
    n_glyc = np.array(n_glyc); deamid = np.array(deamid); iso = np.array(iso)
    cys = np.array(cys); oxid = np.array(oxid)
    aromfrac = np.array(aromfrac); lowcomplex = np.array(lowcomplex)

    hard_pass = (n_glyc == 0) & (cys == 0)
    # soft penalty (lower is better); aromatic load penalised only above ~0.5
    penalty = (deamid + iso + 0.5 * oxid + lowcomplex
               + 2.0 * np.clip(aromfrac - 0.5, 0, None))
    return {
        "n_glyc": n_glyc, "deamid": deamid, "iso": iso, "cys": cys,
        "oxid": oxid, "aromatic_fraction": aromfrac, "low_complexity": lowcomplex,
        "hard_pass": hard_pass, "penalty": penalty.astype(np.float32),
    }
