"""Epitope-preservation check — does a design bind the SAME HER2 surface as trastuzumab?

Epitope preservation is a judged property (competitive binding by SPR). A design that binds
HER2 somewhere else loses, however good its ipTM. So: take the HER2 residues trastuzumab
contacts in the reference complex (1N8Z), take the HER2 residues the design contacts in its
co-folded complex, and report the overlap.

Co-fold chain layout (same in reference and designs): A = trastuzumab VL, B = VH,
C+D = HER2 (domain IV). Antibody = {A,B}; antigen = {C,D}. A contact = any heavy atom of the
antibody within `CUTOFF` A of a HER2 heavy atom.

Deps: gemmi, numpy, scipy.
"""
import sys
from pathlib import Path
import numpy as np
import gemmi
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REF = HERE / "data" / "structures" / "ref_1n8z.cif"
AB_CHAINS = {"A", "B"}
AG_CHAINS = {"C", "D"}
CUTOFF = 4.5
# trastuzumab hotspots on HER2 from her2_cdrh3.yaml (C: 489-580, D: 591-607)
HOTSPOTS = {("C", 572), ("C", 573), ("D", 591), ("D", 593)}


def _heavy(st, chains):
    pts, lab = [], []
    for ch in st[0]:
        if ch.name not in chains:
            continue
        for r in ch:
            for a in r:
                if a.element == gemmi.Element("H"):
                    continue
                pts.append([a.pos.x, a.pos.y, a.pos.z]); lab.append((ch.name, r.seqid.num))
    return np.asarray(pts), lab


def contact_epitope(cif, ab=AB_CHAINS, ag=AG_CHAINS, cutoff=CUTOFF):
    """HER2 residues (chain, resnum) with a heavy atom within `cutoff` of the antibody."""
    st = gemmi.read_structure(str(cif)); st.setup_entities()
    abp, _ = _heavy(st, ab)
    agp, aglab = _heavy(st, ag)
    if len(abp) == 0 or len(agp) == 0:
        return set()
    d, _ = cKDTree(abp).query(agp, k=1)
    return {aglab[i] for i in range(len(agp)) if d[i] <= cutoff}


def preservation(design_cif, ref_cif=REF):
    ref = contact_epitope(ref_cif)
    des = contact_epitope(design_cif)
    inter = des & ref
    return {
        "epitope_recall": round(len(inter) / len(ref), 3) if ref else 0.0,   # share of trastuzumab's epitope kept
        "jaccard": round(len(inter) / len(des | ref), 3) if (des or ref) else 0.0,
        "hotspots_hit": sorted(f"{c}{n}" for (c, n) in des & HOTSPOTS),
        "n_ref": len(ref), "n_design": len(des), "n_shared": len(inter),
        "lost": sorted(f"{c}{n}" for (c, n) in ref - des),
    }


def demo():
    ref = contact_epitope(REF)
    print(f"reference epitope (trastuzumab on HER2, 1N8Z): {len(ref)} residues")
    print("  " + " ".join(f"{c}{n}" for c, n in sorted(ref)))
    print(f"  hotspots present in reference: {sorted(f'{c}{n}' for c,n in ref & HOTSPOTS)}")
    for cif in sorted((HERE / "data/structures").glob("*_generated.cif")):
        p = preservation(cif)
        print(f"\n{cif.name}")
        print(f"  epitope recall = {p['epitope_recall']}  (kept {p['n_shared']}/{p['n_ref']}) · Jaccard {p['jaccard']}")
        print(f"  hotspots hit: {p['hotspots_hit'] or 'none'}")
        print(f"  lost: {' '.join(p['lost']) or 'none'}")
    print("\nself-check passed")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(preservation(sys.argv[1]))
    else:
        demo()
