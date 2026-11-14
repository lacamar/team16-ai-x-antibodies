"""
Independent re-verification of the submission gates on the FINAL shortlist.

This is deliberately not reused from the shell-search code that proposed the
candidates: a filter applied by the generator is not evidence that the shipped
molecules pass. Here we recompute, from the raw provided data, the exact
minimum Hamming distance from every shipped CDRH3 to all 525,063 provided
CDRH3s, and re-apply all four documented abguard checks.

`abguard` itself is absent from this VM (NOTES.md Blocker 1): it is not on
PATH, not in pip, not on PyPI, not inside any Apheris image, and a recursive
search of s3://apheris-oslo-hackathon/ for it found nothing. This file is our
reimplementation of its four documented checks plus the two track rules. The
one check we CANNOT reproduce is immunum numbering confidence >= 0.1, which
needs ANARCI/ANARCII; neither installs here. That gap is declared, not hidden.

Writes runs/novelty_shell.json, which rank.py consumes as a hard gate.
"""
import csv
import io
import itertools
import json
import os
import zipfile

AA = "ACDEFGHIKLMNPQRSTVWY"
TZ = "WGGDGFYAMD"
DATA = "/home/hackathon/team16-ai-x-antibodies/data/her2"
HERE = os.path.dirname(os.path.abspath(__file__))

VH_PRE = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRY"
          "ADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR")
VH_POST = "YWGQGTLVTVSS"
VL = ("DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPS"
      "RFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK")


def load_provided():
    prov = set()
    with zipfile.ZipFile(f"{DATA}/tz_her2_facs_3classes_524346.csv.zip") as z:
        with z.open(z.namelist()[0]) as f:
            for r in csv.DictReader(io.TextIOWrapper(f, "utf-8")):
                s = (r.get("cdrh3") or "").strip().upper()
                if s:
                    prov.add(s)
    with open(f"{DATA}/trastuzumab_her2_spr_all_768.csv") as f:
        for r in csv.DictReader(f):
            s = (r.get("cdrh3") or "").strip().upper()
            if s:
                prov.add(s)
    return prov


def min_ham(s, prov):
    """Exact min Hamming distance to the provided set (radius 0,1,2 enumerated;
    returns 3 meaning '>=3'). Exact, not sampled."""
    if s in prov:
        return 0
    for rad in (1, 2):
        for pos in itertools.combinations(range(len(s)), rad):
            for subs in itertools.product(AA, repeat=rad):
                if any(s[p] == c for p, c in zip(pos, subs)):
                    continue
                t = list(s)
                for p, c in zip(pos, subs):
                    t[p] = c
                if "".join(t) in prov:
                    return rad
    return 3


def check(cdrh3, prov):
    vh = VH_PRE + cdrh3 + VH_POST
    scfv_len = len(vh) + len(VL)
    d = min_ham(cdrh3, prov)
    n_mut = sum(a != b for a, b in zip(cdrh3, TZ))
    fails = []
    if scfv_len > 250:
        fails.append(f"scFv length {scfv_len} > 250")
    if (vh.count("C") % 2) or (VL.count("C") % 2):
        fails.append("unpaired cysteine")
    if d < 3:
        fails.append(f"only {d} point mutations from a provided antibody, rule needs >2")
    # CDRH3 identity to trastuzumab must be UNDER 70%. On 10 designable
    # residues 3 changes leaves exactly 70.0%, which is not under 70%, so we
    # require >=4 (the strict reading). See NOTES.md Q1.
    if n_mut < 4:
        fails.append(f"CDRH3 identity {100*(10-n_mut)/10:.0f}% not under 70% (strict reading)")
    return {"cdrh3": cdrh3, "n_mut_vs_tz": n_mut, "min_dist_provided": d,
            "scfv_len": scfv_len, "format": "scFv",
            "numbering_confidence": None,   # cannot compute: no ANARCI/ANARCII
            "fails": fails, "pass": not fails}


def main():
    import pandas as pd
    prov = load_provided()
    print(f"provided CDRH3 reference set: {len(prov)}")
    df = pd.read_csv(f"{HERE}/runs/shell_merged.csv")
    rows = []
    for r in df.itertuples():
        c = check(r.cdrh3, prov)
        c["query_id"] = r.query_id
        rows.append(c)
    out = pd.DataFrame(rows)
    out.to_csv(f"{HERE}/results/gate_verification.csv", index=False)
    fail = sorted(out[~out["pass"]].cdrh3.unique().tolist())
    json.dump({"fail": fail,
               "n_checked": len(out),
               "n_pass": int(out["pass"].sum()),
               "provided_set_size": len(prov),
               "uncomputable_check": "immunum numbering confidence >= 0.1 "
                                     "(needs ANARCI/ANARCII, not installable here)"},
              open(f"{HERE}/runs/novelty_shell.json", "w"), indent=1)
    print(f"{out['pass'].sum()}/{len(out)} pass all computable gates")
    print("min_dist_provided counts:", out.min_dist_provided.value_counts().to_dict())
    print("n_mut_vs_tz counts:", out.n_mut_vs_tz.value_counts().to_dict())
    if fail:
        print("FAILURES:")
        print(out[~out["pass"]][["query_id", "cdrh3", "fails"]].to_string(index=False))


if __name__ == "__main__":
    main()
