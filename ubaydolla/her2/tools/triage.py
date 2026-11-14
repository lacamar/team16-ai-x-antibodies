#!/usr/bin/env python3
"""
CDRH3 triage for the HER2/trastuzumab optimisation track.

Two jobs, in this order of value:

  1. POSITION INFORMATION. From the 524k FACS CDRH3 variants, work out which
     CDRH3 positions actually carry the binding signal and which are tolerant.
     This tells you WHERE to spend your <70%-identity budget. Mutating the
     tolerant positions is close to free; mutating the high-information ones
     is how you lose axis 1 and axis 2 together.

  2. CALIBRATION. Score the ~190 SPR-measured optimisation designs with the
     PSSM built in step 1 and correlate against measured affinity. If this
     correlation is ~0, a sequence-only ranker is noise for your designs and
     you must lean on the structural scorers. If it is decent, you have a
     ranker that costs milliseconds and is independent of ipTM.

Schema-tolerant on purpose: it sniffs column names and prints what it chose,
so you are not blocked on knowing the Chinery et al. column headers.

Usage
-----
  python3 triage.py --facs path/to/facs_variants.csv \
                    [--spr path/to/spr_designs.csv] \
                    [--wt WGGDGFYAMDY] \
                    [--out-dir ./triage_out]

  # if it picks the wrong columns, pin them:
  python3 triage.py --facs f.csv --facs-seq-col cdrh3 --facs-label-col bin

Outputs (to --out-dir):
  position_information.csv   per-position entropy / information content
  pssm_logodds.csv           high-vs-low log-odds matrix, the ranker
  mutation_budget.md         human-readable: conserve these, mutate those
  spr_calibration.csv        per-design PSSM score vs measured affinity
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {a: i for i, a in enumerate(AA)}

# Motifs that cost you measured thermostability / developability for free.
LIABILITIES = {
    "NG": "deamidation",
    "NS": "deamidation (weaker)",
    "DG": "isomerisation",
    "DP": "acid cleavage",
    "DS": "isomerisation (weaker)",
}


# ---------------------------------------------------------------- loading


def load_any(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    sep = "\t" if ".tsv" in low or ".tab" in low else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def looks_like_protein(s: pd.Series, min_frac: float = 0.9) -> bool:
    """True if most non-null values are plausible short peptide strings."""
    sample = s.dropna().astype(str).head(2000)
    if sample.empty:
        return False
    ok = 0
    for v in sample:
        v = v.strip().upper()
        if not v or len(v) > 40 or len(v) < 3:
            continue
        if all(c in AA_IDX for c in v):
            ok += 1
    return (ok / len(sample)) >= min_frac


def pick_seq_col(df: pd.DataFrame, hint: str | None) -> str:
    if hint:
        if hint not in df.columns:
            sys.exit(f"[fatal] --*-seq-col {hint!r} not in columns: {list(df.columns)}")
        return hint
    preferred = [c for c in df.columns if any(
        k in c.lower() for k in ("cdrh3", "cdr_h3", "cdr3", "h3", "seq", "sequence", "aa")
    )]
    for c in preferred + list(df.columns):
        if df[c].dtype == object and looks_like_protein(df[c]):
            return c
    sys.exit(f"[fatal] no peptide-like column found. columns: {list(df.columns)}")


def pick_label_col(df: pd.DataFrame, seq_col: str, hint: str | None) -> str:
    if hint:
        if hint not in df.columns:
            sys.exit(f"[fatal] --*-label-col {hint!r} not in columns: {list(df.columns)}")
        return hint
    cands = [c for c in df.columns if c != seq_col and any(
        k in c.lower() for k in
        ("class", "label", "bin", "gate", "affinity", "kd", "score", "category", "tier")
    )]
    # prefer low-cardinality (a class label) over continuous
    cands.sort(key=lambda c: (df[c].nunique(dropna=True) > 12, df[c].nunique(dropna=True)))
    if cands:
        return cands[0]
    sys.exit(f"[fatal] no label-like column found. columns: {list(df.columns)}")


def pick_numeric_col(df: pd.DataFrame, seq_col: str, hint: str | None) -> str:
    if hint:
        if hint not in df.columns:
            sys.exit(f"[fatal] column {hint!r} not in columns: {list(df.columns)}")
        return hint
    num = [c for c in df.columns
           if c != seq_col and pd.api.types.is_numeric_dtype(df[c])]
    pref = [c for c in num if any(k in c.lower() for k in
            ("kd", "affinity", "ka", "kon", "koff", "ddg", "dg", "ec50", "ic50"))]
    if pref:
        return pref[0]
    if num:
        return num[0]
    sys.exit(f"[fatal] no numeric affinity column found. columns: {list(df.columns)}")


# ---------------------------------------------------------- core analysis


def order_classes(df: pd.DataFrame, label_col: str) -> list:
    """Return class values ordered worst -> best, as best we can infer."""
    vals = df[label_col].dropna().unique().tolist()
    if pd.api.types.is_numeric_dtype(df[label_col]):
        return sorted(vals)
    # string classes: try to order by common naming
    rank_hints = ["non", "neg", "no", "low", "weak", "mid", "med", "int",
                  "high", "strong", "tight", "pos"]
    def key(v):
        s = str(v).lower()
        for i, h in enumerate(rank_hints):
            if h in s:
                return i
        return len(rank_hints)
    return sorted(vals, key=key)


def modal_length(seqs: pd.Series) -> int:
    return Counter(seqs.str.len()).most_common(1)[0][0]


def freq_matrix(seqs: list[str], L: int, pseudo: float = 0.5) -> np.ndarray:
    M = np.full((L, len(AA)), pseudo, dtype=float)
    for s in seqs:
        for i, c in enumerate(s):
            j = AA_IDX.get(c)
            if j is not None:
                M[i, j] += 1.0
    return M / M.sum(axis=1, keepdims=True)


def information_content(freqs: np.ndarray) -> np.ndarray:
    """bits per position, max log2(20) ~= 4.32"""
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(freqs > 0, freqs * np.log2(freqs), 0.0), axis=1)
    return math.log2(len(AA)) - ent


def score_with_pssm(seq: str, pssm: np.ndarray) -> float | None:
    if len(seq) != pssm.shape[0]:
        return None
    tot = 0.0
    for i, c in enumerate(seq):
        j = AA_IDX.get(c)
        if j is None:
            return None
        tot += pssm[i, j]
    return tot


def scan_liabilities(seq: str) -> list[str]:
    hits = []
    for i in range(len(seq) - 1):
        m = seq[i:i + 2]
        if m in LIABILITIES:
            hits.append(f"{m}@{i + 1}:{LIABILITIES[m]}")
    for i, c in enumerate(seq):
        if c in "MWC":
            note = "free cysteine" if c == "C" else "oxidation-prone"
            hits.append(f"{c}@{i + 1}:{note}")
    return hits


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 8:
        return float("nan"), n
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = np.corrcoef(rx, ry)[0, 1]
    return float(rho), n


# ------------------------------------------------------------------ main


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--facs", required=True, help="524k CDRH3 FACS variants")
    p.add_argument("--spr", help="~190 optimisation designs with SPR affinity")
    p.add_argument("--wt", default=None, help="wild-type trastuzumab CDRH3 (verify from 1N8Z!)")
    p.add_argument("--facs-seq-col"), p.add_argument("--facs-label-col")
    p.add_argument("--spr-seq-col"), p.add_argument("--spr-affinity-col")
    p.add_argument("--affinity-lower-is-better", action="store_true",
                   help="set if the SPR column is KD (lower = tighter)")
    p.add_argument("--out-dir", default="./triage_out")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    # ---- 1. FACS: where is the signal?
    print(f"[load] {a.facs}")
    facs = load_any(a.facs)
    sc = pick_seq_col(facs, a.facs_seq_col)
    lc = pick_label_col(facs, sc, a.facs_label_col)
    print(f"[cols] sequence={sc!r}  label={lc!r}   (pin with --facs-seq-col/--facs-label-col)")

    facs = facs[[sc, lc]].dropna()
    facs[sc] = facs[sc].astype(str).str.strip().str.upper()
    facs = facs[facs[sc].str.fullmatch(f"[{''.join(AA)}]+")]

    L = modal_length(facs[sc])
    n_all = len(facs)
    facs = facs[facs[sc].str.len() == L]
    print(f"[len ] modal CDRH3 length {L} keeps {len(facs):,}/{n_all:,} variants "
          f"({len(facs)/max(n_all,1):.1%}). Fixed-length analysis only.")

    classes = order_classes(facs, lc)
    print(f"[cls ] inferred worst->best: {classes}")
    print(facs[lc].value_counts().to_string())
    if len(classes) < 2:
        sys.exit("[fatal] need >=2 affinity classes")

    lo_seqs = facs.loc[facs[lc] == classes[0], sc].tolist()
    hi_seqs = facs.loc[facs[lc] == classes[-1], sc].tolist()
    print(f"[sets] low={len(lo_seqs):,}  high={len(hi_seqs):,}")

    f_hi = freq_matrix(hi_seqs, L)
    f_lo = freq_matrix(lo_seqs, L)
    pssm = np.log2(f_hi / f_lo)                      # positive = enriched in binders

    ic_hi = information_content(f_hi)
    jsd = np.abs(f_hi - f_lo).sum(axis=1) / 2.0      # total-variation per position

    wt = (a.wt or "").strip().upper() or None
    if wt and len(wt) != L:
        print(f"[warn] --wt length {len(wt)} != modal length {L}; ignoring for per-position report")
        wt = None

    rows = []
    for i in range(L):
        best = AA[int(np.argmax(pssm[i]))]
        worst = AA[int(np.argmin(pssm[i]))]
        rows.append({
            "position": i + 1,
            "wt_aa": wt[i] if wt else "",
            "info_bits_high_class": round(float(ic_hi[i]), 3),
            "divergence_high_vs_low": round(float(jsd[i]), 3),
            "best_aa": best,
            "best_logodds": round(float(pssm[i].max()), 3),
            "worst_aa": worst,
            "wt_logodds": round(float(pssm[i, AA_IDX[wt[i]]]), 3) if wt and wt[i] in AA_IDX else "",
            "top5_enriched": ",".join(AA[j] for j in np.argsort(-pssm[i])[:5]),
        })
    pos = pd.DataFrame(rows)
    pos.to_csv(os.path.join(a.out_dir, "position_information.csv"), index=False)

    pd.DataFrame(pssm, columns=AA, index=[f"pos{i+1}" for i in range(L)]).to_csv(
        os.path.join(a.out_dir, "pssm_logodds.csv"))

    # the actual decision: conserve vs mutate
    cons = pos.sort_values("divergence_high_vs_low", ascending=False)
    n_mut_needed = math.ceil(L * 0.31)   # >30% divergence => <70% identity
    tolerant = cons.tail(max(n_mut_needed + 2, 4))

    md = [
        "# CDRH3 mutation budget",
        "",
        f"- modal CDRH3 length: **{L}**",
        f"- to clear the <70% identity rule you need at least **{n_mut_needed}** "
        f"substitutions ({n_mut_needed}/{L} = {n_mut_needed/L:.1%} divergence). "
        f"Aim for {n_mut_needed + 1}-{n_mut_needed + 2} so abguard's identity "
        f"calculation disagreeing with yours does not reject the design.",
        "",
        "## Conserve (highest high-vs-low divergence = carries the binding signal)",
        "",
        cons.head(max(3, L - n_mut_needed - 2))[
            ["position", "wt_aa", "divergence_high_vs_low", "best_aa", "top5_enriched"]
        ].to_markdown(index=False),
        "",
        "## Spend your identity budget here (most tolerant positions)",
        "",
        tolerant[["position", "wt_aa", "divergence_high_vs_low", "best_aa", "top5_enriched"]]
        .to_markdown(index=False),
        "",
        "Pick substitutions from `top5_enriched` at the tolerant positions: you then "
        "satisfy the novelty rule using residues the FACS screen says binders prefer, "
        "instead of random drift away from the epitope.",
    ]
    if wt:
        hits = scan_liabilities(wt)
        md += ["", "## Liability motifs in the WT CDRH3",
               "", ("\n".join(f"- `{h}`" for h in hits) if hits else "- none found"),
               "", "Removing these at tolerant positions is a free thermostability "
               "hypothesis: it spends identity budget you must spend anyway."]
    open(os.path.join(a.out_dir, "mutation_budget.md"), "w").write("\n".join(md))
    print(f"[out ] {a.out_dir}/mutation_budget.md  <- read this one first")

    # ---- 2. SPR: is a sequence ranker worth anything?
    if a.spr:
        print(f"\n[load] {a.spr}")
        spr = load_any(a.spr)
        ssc = pick_seq_col(spr, a.spr_seq_col)
        sac = pick_numeric_col(spr, ssc, a.spr_affinity_col)
        print(f"[cols] sequence={ssc!r}  affinity={sac!r}")

        d = spr[[ssc, sac]].dropna().copy()
        d[ssc] = d[ssc].astype(str).str.strip().str.upper()
        d["pssm_score"] = [score_with_pssm(s, pssm) for s in d[ssc]]
        d["len"] = d[ssc].str.len()
        d["liabilities"] = [";".join(scan_liabilities(s)) for s in d[ssc]]
        d.to_csv(os.path.join(a.out_dir, "spr_calibration.csv"), index=False)

        usable = d["pssm_score"].notna().sum()
        print(f"[calib] {usable}/{len(d)} designs scorable at length {L} "
              f"(others differ in CDRH3 length)")

        y = d[sac].to_numpy(dtype=float)
        if a.affinity_lower_is_better:
            y = -y
        rho, n = spearman(d["pssm_score"].to_numpy(dtype=float), y)
        print(f"[calib] Spearman(PSSM, measured affinity) = {rho:+.3f}  (n={n})")
        print("[calib] interpretation:")
        print("        |rho| < 0.15  -> sequence ranker is noise HERE. Do not rank on it.")
        print("        0.15-0.35     -> weak but real. Use as a tiebreak / veto only.")
        print("        > 0.35        -> use it as a primary ranker; it is free and")
        print("                         independent of ipTM, so consensus is meaningful.")
        print("\n[next] rerun this correlation with your ipTM / min_ipae column on the")
        print("       same 190 designs. Whichever metric ranks THEM is the metric that")
        print("       ranks YOURS. If neither does, rank on thermostability consensus")
        print("       and say so in the pitch - that is a defensible call, guessing is not.")


if __name__ == "__main__":
    main()
