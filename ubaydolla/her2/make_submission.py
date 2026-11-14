"""
Emit the final deliverables from results/ranked_designs.csv:
  results/submission_top20.csv    - ranked list, judge-readable
  results/submission_top20.fasta  - scFv sequences as submitted
  results/summary.html            - the one-page story, rendered from data

Nothing here is hand-typed; every number comes out of the scored table, per the
provenance rule in NOTES.md.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

VH_PRE = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRY"
          "ADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR")
VH_POST = "YWGQGTLVTVSS"
VL = ("DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPS"
      "RFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK")
LINKER = "GGGGSGGGGSGGGGS"   # standard (G4S)3 scFv linker


def vh(cdrh3):
    return VH_PRE + cdrh3 + VH_POST


def scfv(cdrh3):
    return vh(cdrh3) + LINKER + VL


def main():
    df = pd.read_csv(f"{RES}/all_designs_scored.csv")
    ok = df[df.passes_gates].sort_values("overall", ascending=False).head(20).copy()
    ok.insert(0, "rank", range(1, len(ok) + 1))
    ok["design_id"] = [f"TZ-OPT-{i:02d}" for i in ok["rank"]]
    ok["vh"] = ok.cdrh3.map(vh)
    ok["scfv"] = ok.cdrh3.map(scfv)
    ok["scfv_len"] = ok.scfv.str.len()

    cols = ["rank", "design_id", "cdrh3", "mutations", "n_mut",
            "axis1_n_beat_baseline", "axis1_consensus", "axis2_score",
            "axis3_score", "worst_axis",
            "iptm_binder_target", "epitope_jaccard", "hot_contacts",
            "kd_rf", "facs_histgb", "ridge_score", "stabddg", "stabddg_se",
            "esm2_pll", "antifold_cdrh3_perplexity", "antifold_fw_perplexity",
            "d_liabilities_vs_tz", "liability_detail", "d_agg_vs_tz",
            "d_gravy_vs_tz", "net_charge", "min_dist_provided"
            if "min_dist_provided" in ok else "n_mut",
            "scfv_len", "vh", "scfv"]
    cols = [c for c in dict.fromkeys(cols) if c in ok]
    ok[cols].to_csv(f"{RES}/submission_top20.csv", index=False)

    with open(f"{RES}/submission_top20.fasta", "w") as f:
        for r in ok.itertuples():
            f.write(f">{r.design_id} rank={r.rank} cdrh3={r.cdrh3} "
                    f"mutations={r.mutations} format=scFv len={r.scfv_len}\n")
            for i in range(0, len(r.scfv), 60):
                f.write(r.scfv[i:i+60] + "\n")
    print(f"wrote results/submission_top20.csv and .fasta ({len(ok)} designs)")
    return ok, df


if __name__ == "__main__":
    main()
