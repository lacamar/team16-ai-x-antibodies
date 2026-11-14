"""
Final ranking. Assembles every axis into one table, applies the hard gates,
and emits results/ranked_designs.csv + the rejection ledger.

Design of the ranking, and why it is this way:

AXIS 1 (binding) is ranked on a CONSENSUS of four scorers that do not share a
fit: two trained on measurement (HistGradientBoosting on 367k clonotype-split
FACS labels; RandomForest on 180 quantified SPR KDs), the additive ridge that
proposed the shell, and Boltz-2 binder-target interface ipTM from the fold.
Finding F2 is the reason: single-model argmax walked into a region where two
models that agree at rho=0.72 in-distribution anti-correlate at -0.27. A
candidate is promoted only when independent scorers agree.

AXIS 2 (epitope) is a measured structural quantity, not a prior: Jaccard
overlap of contacted HER2 residues against the trastuzumab baseline folded in
the SAME run, plus retention of all four conditioned hotspot atoms.

AXIS 3 (thermostability) has no Tm predictor on either lane (Blocker 2). We do
not invent one. We rank on what is defensible: the framework is identical to
the clinical molecule so Tm is near-constant by construction, and we score the
loop's own aggregation / chemical-liability / charge contribution
deterministically, plus ESM-2 likelihood and AntiFold backbone perplexity as
learned naturalness proxies. Every such column is labelled a proxy.

Gates are hard. A design that fails novelty is kept as a rejection row with a
reason, never silently dropped.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from developability import TZ, panel

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def pct(s):
    """Percentile rank, higher = better. NaN-safe."""
    return s.rank(pct=True, na_option="keep")


def load():
    m = pd.read_csv(f"{HERE}/runs/shell_merged.csv")
    ep = pd.read_csv(f"{RES}/epitope_shell.csv")
    df = m.merge(ep, on="query_id", how="left")

    # ESM-2 pseudo-likelihood (lane B, op-07b3495e1f4d104a)
    esm = None
    for p in os.listdir(f"{HERE}/nebius-runs/esm2-out"):
        if p.endswith(".csv"):
            esm = pd.read_csv(f"{HERE}/nebius-runs/esm2-out/{p}")
    if esm is not None:
        esm["query_id"] = esm["id"].str.split("|").str[0]
        esm = esm[["query_id", "mean_score"]].rename(
            columns={"mean_score": "esm2_pll"}).drop_duplicates("query_id")
        df = df.merge(esm, on="query_id", how="left")

    # AntiFold backbone perplexity (lane B), if it has landed
    af = f"{RES}/antifold_scores.csv"
    if os.path.exists(af):
        df = df.merge(pd.read_csv(af), on="query_id", how="left")

    # StaB-ddG binding ddG (lane B), if it has landed
    sd = f"{RES}/stabddg_scores.csv"
    if os.path.exists(sd):
        df = df.merge(pd.read_csv(sd), on="query_id", how="left")

    # deterministic developability panel
    dev = pd.DataFrame([panel(c) for c in df.cdrh3]).drop(columns=["cdrh3", "n_mut"])
    df = pd.concat([df.reset_index(drop=True), dev.reset_index(drop=True)], axis=1)
    return df


def gate(df):
    """Hard gates. abguard is absent from this VM (Blocker 1); these are our
    reimplementation of its four documented checks plus the two track rules."""
    novel = json.load(open(f"{HERE}/runs/novelty_shell.json")) \
        if os.path.exists(f"{HERE}/runs/novelty_shell.json") else None
    reasons = []
    for r in df.itertuples():
        bad = []
        vh = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRY"
              "ADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR" + r.cdrh3 + "YWGQGTLVTVSS")
        vl = ("DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPS"
              "RFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK")
        # format: scFv = VH + VL, length <= 250 aa
        if len(vh) + len(vl) > 250:
            bad.append(f"length {len(vh)+len(vl)}>250")
        # no unpaired cysteines: each domain has exactly its one internal pair
        if (vh.count("C") % 2) or (vl.count("C") % 2):
            bad.append("unpaired Cys")
        # track rule: >2 point mutations from any provided antibody
        if r.query_id == "BASELINE_trastuzumab":
            bad.append("baseline control, not a submission")
        elif r.n_mut <= 2:
            bad.append(f"only {r.n_mut} CDRH3 mutations, rule needs >2")
        if novel is not None and r.cdrh3 in novel.get("fail", []):
            bad.append("within Hamming 2 of a provided CDRH3")
        reasons.append("; ".join(bad))
    df["reject_reason"] = reasons
    df["passes_gates"] = df.reject_reason == ""
    return df


def rank(df):
    base = df[df.query_id == "BASELINE_trastuzumab"]
    b = base.iloc[0] if len(base) else None

    # ---- axis 1: consensus of five scorers that share no fit
    #   kd_rf              RandomForest on 180 quantified SPR KDs
    #   facs_histgb        HistGradientBoosting on 367k clonotype-split FACS
    #   ridge_score        additive ridge that proposed the shell
    #   iptm_binder_target Boltz-2 binder-target interface ipTM (structural)
    #   stabddg_better     StaB-ddG binding ddG, sign-flipped (higher better)
    # These do NOT agree with each other. Measured on this shortlist (n=40):
    #   spearman(stabddg, kd_rf) = -0.405 (p=0.009)
    #   spearman(stabddg, iptm)  = -0.338 (p=0.033)
    #   spearman(iptm,    kd_rf) = +0.343 (p<0.001, from Finding F4)
    # Two pairs agree, two anti-correlate. No single ranking is therefore
    # trustworthy, and a mean of percentiles would silently average away the
    # disagreement. So the ranking key is UNANIMITY OF SIGN against the
    # in-run trastuzumab baseline (axis1_n_beat_baseline), with the percentile
    # consensus used only to order designs that tie on unanimity.
    if "stabddg" in df:
        df["stabddg_better"] = -df["stabddg"]
    a1 = ["kd_rf", "facs_histgb", "ridge_score", "iptm_binder_target",
          "stabddg_better"]
    a1 = [c for c in a1 if c in df]
    df["axis1_consensus"] = np.nanmean([pct(df[c]) for c in a1], axis=0)

    # how many independent scorers put this design above trastuzumab.
    # The baseline has no kd_rf/ridge/stabddg of its own (stabddg is defined
    # AS the difference from it, so its baseline value is exactly 0, and the
    # empirical models' trastuzumab values are measured separately).
    TZ_REF = {"kd_rf": 0.127,          # measured, NOTES.md axis-1 bar
              "facs_histgb": 0.80,     # used as a floor, not a ceiling (F3)
              "ridge_score": -0.287,   # NOTES.md F2
              "stabddg_better": 0.0}   # ddG is relative to trastuzumab
    if b is not None:
        TZ_REF["iptm_binder_target"] = float(b["iptm_binder_target"])
    n = np.zeros(len(df))
    for c in a1:
        n += (df[c] > TZ_REF[c]).astype(float).values
    df["axis1_n_beat_baseline"] = n.astype(int)
    df["axis1_n_scorers"] = len(a1)

    # ---- axis 2: epitope preservation
    df["axis2_score"] = np.nanmean(
        [pct(df.epitope_jaccard), pct(df.hot_contacts)], axis=0)

    # ---- axis 3: developability proxies (lower agg / fewer liabilities better)
    a3parts = [pct(-df.d_agg_vs_tz), pct(-df.d_liabilities_vs_tz)]
    if "esm2_pll" in df:
        a3parts.append(pct(df.esm2_pll))
    if "antifold_score" in df:
        a3parts.append(pct(df.antifold_score))
    df["axis3_score"] = np.nanmean(a3parts, axis=0)

    # ---- overall: S-tier requires all three to improve, so weight the
    # weakest axis, never the mean alone. We rank on the minimum of the three
    # normalised axes (a design that is excellent on one axis and poor on
    # another is explicitly NOT what the judging rewards), tie-broken by mean.
    ax = df[["axis1_consensus", "axis2_score", "axis3_score"]]
    df["worst_axis"] = ax.min(axis=1)
    df["mean_axis"] = ax.mean(axis=1)

    # S-tier requires improving ALL THREE axes, so the primary key is how many
    # independent axis-1 scorers beat trastuzumab (unanimity), then the weakest
    # axis. A design excellent on one axis and poor on another is explicitly
    # not what the judging rewards, so the mean is only a tie-break.
    df["overall"] = (df.axis1_n_beat_baseline
                     + df.worst_axis
                     + 0.001 * df.mean_axis)
    return df


def main():
    df = rank(gate(load()))
    os.makedirs(RES, exist_ok=True)
    df.to_csv(f"{RES}/all_designs_scored.csv", index=False)

    ok = df[df.passes_gates].sort_values("overall", ascending=False)
    rej = df[~df.passes_gates]
    ok.head(20).to_csv(f"{RES}/ranked_designs.csv", index=False)
    rej.to_csv(f"{RES}/rejected_designs.csv", index=False)

    cols = ["query_id", "cdrh3", "mutations", "n_mut", "axis1_consensus",
            "axis2_score", "axis3_score", "worst_axis",
            "iptm_binder_target", "epitope_jaccard", "hot_contacts",
            "kd_rf", "d_liabilities_vs_tz", "d_agg_vs_tz"]
    cols = [c for c in cols if c in ok]
    print(f"\n{len(ok)} designs pass all gates; {len(rej)} rejected "
          f"(ledger: results/rejected_designs.csv)\n")
    print(ok[cols].head(20).round(3).to_string(index=False))
    b = df[df.query_id == "BASELINE_trastuzumab"]
    if len(b):
        print("\nBASELINE trastuzumab (same fold run, same scorers):")
        print(b[cols].round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
