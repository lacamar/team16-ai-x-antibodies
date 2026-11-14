"""
Axis-3 (thermostability / developability) panel for CDRH3 variants on a FIXED
trastuzumab framework.

Rationale, which is the whole reason this file is defensible:
every candidate we ship differs from trastuzumab ONLY at up to 5 of the 10
CDRH3 positions (B99-B108). The VH framework, the VL, the inter-domain packing
and both disulfides are byte-identical to the clinical molecule. Fv melting
temperature is dominated by framework and VH/VL interface packing, so these
designs are near-isothermal with trastuzumab BY CONSTRUCTION. No Tm predictor
was available on either lane (see NOTES.md Blocker 2), and we do not pretend to
have one.

What we can score honestly is the part we actually changed: the loop's own
contribution to aggregation propensity, chemical-degradation liabilities and
charge/solubility. Those are the failure modes that sink expression and
thermostability for an otherwise-clinical framework.

All descriptors are deterministic functions of the 10-mer. No fitting, no
extrapolation, so none of the scorer-adversarial failure of Finding F2 applies.
Every column is labelled proxy / deterministic in the output.
"""
import re
import pandas as pd

TZ = "WGGDGFYAMD"          # trastuzumab CDRH3, 10 designable residues B99-B108
CDRH3_START = 99

# Kyte-Doolittle hydropathy
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
      'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
      'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGE = {'D':-1,'E':-1,'K':1,'R':1,'H':0.1}


def gravy(s):
    return sum(KD[c] for c in s) / len(s)


def net_charge(s):
    return sum(CHARGE.get(c, 0) for c in s)


def liabilities(s):
    """Known CDR chemical-degradation motifs. Returns (count, list-of-names).

    These are the standard developability liabilities screened in antibody
    engineering. Note trastuzumab's own CDRH3 carries two of them (DG
    isomerisation at D102-G103, Met oxidation at M107) - so a design that
    removes one is genuinely better than the clinical baseline on that motif,
    not merely 'not worse'.
    """
    hits = []
    for m in re.finditer(r'N[^P][ST]', s):
        hits.append(f"N-glyc:N{CDRH3_START+m.start()}")
    for m in re.finditer(r'N[GS]', s):
        hits.append(f"deamid:N{CDRH3_START+m.start()}")
    for m in re.finditer(r'D[GSTD]', s):
        hits.append(f"isomer:D{CDRH3_START+m.start()}")
    for m in re.finditer(r'M', s):
        hits.append(f"oxid:M{CDRH3_START+m.start()}")
    for m in re.finditer(r'W', s):
        hits.append(f"oxid:W{CDRH3_START+m.start()}")
    for m in re.finditer(r'DP', s):
        hits.append(f"cleav:D{CDRH3_START+m.start()}")
    if s.count('C') % 2 == 1:
        hits.append("unpaired-Cys")
    return len(hits), ";".join(hits)


def agg_window(s, w=5):
    """Crude aggregation-prone-region surrogate: max mean hydropathy over a
    sliding window. A stand-in for Aggrescan3D/CamSol, which were absent on
    both lanes. Labelled as a proxy everywhere it is reported."""
    if len(s) < w:
        return gravy(s)
    return max(sum(KD[c] for c in s[i:i+w]) / w for i in range(len(s)-w+1))


def panel(cdrh3):
    n_lia, lia = liabilities(cdrh3)
    tz_n, _ = liabilities(TZ)
    return {
        "cdrh3": cdrh3,
        "n_mut": sum(a != b for a, b in zip(cdrh3, TZ)),
        "mutations": ",".join(
            f"{TZ[i]}{CDRH3_START+i}{cdrh3[i]}"
            for i in range(len(TZ)) if cdrh3[i] != TZ[i]),
        "gravy": round(gravy(cdrh3), 3),
        "d_gravy_vs_tz": round(gravy(cdrh3) - gravy(TZ), 3),
        "net_charge": net_charge(cdrh3),
        "d_charge_vs_tz": net_charge(cdrh3) - net_charge(TZ),
        "agg_window_proxy": round(agg_window(cdrh3), 3),
        "d_agg_vs_tz": round(agg_window(cdrh3) - agg_window(TZ), 3),
        "n_liabilities": n_lia,
        "d_liabilities_vs_tz": n_lia - tz_n,
        "liability_detail": lia,
    }


if __name__ == "__main__":
    import sys
    seqs = sys.argv[1:] or [TZ]
    print(pd.DataFrame([panel(s) for s in seqs]).to_string(index=False))
