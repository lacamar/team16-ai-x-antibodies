"""
Axis-2 (epitope preservation) measured structurally, per design, in one
folding model, with the trastuzumab baseline folded in the SAME run.

Axis 2 is judged experimentally by competitive binding against trastuzumab: a
design scores well only if it occupies the trastuzumab epitope, so that
trastuzumab competes it off. The computational stand-in is therefore not
'does it bind somewhere' but 'does it contact the same HER2 residues'.

The operational epitope is the four hotspot atoms the RFD4 HER2 spec conditions
on (Data/12-her2/her2_cdrh3.yaml, C_HOT):
    HER2 Pro572 CD, Phe573 CD1   (chain C)
    HER2 Ile591 CG2, Lys593 NZ   (chain D)
plus, more robustly, the full set of HER2 residues the baseline trastuzumab Fv
actually touches in its own fold. We report:

  hot_contacts    - heavy-atom contacts <5 A to the 4 hotspot atoms
  hot_min_dist    - closest approach to each hotspot
  epitope_jaccard - Jaccard overlap of the contacted-HER2-residue set with the
                    in-run BASELINE_trastuzumab fold. This is the headline
                    axis-2 number: 1.0 means the design engages exactly the
                    residues trastuzumab engages.
  cdrh3_contacts  - how much of the interface the redesigned loop itself carries

Comparisons are only ever made inside one folding model (Boltz-2, same
template, same seed), per the provenance rule in NOTES.md.
"""
import glob
import os
import sys

import gemmi
import pandas as pd

HOTSPOTS = [("C", 572, "CD"), ("C", 573, "CD1"), ("D", 591, "CG2"), ("D", 593, "NZ")]
TARGET_CHAINS = ("C", "D")
BINDER_CHAINS = ("A", "B")
CDRH3 = ("B", range(99, 109))
CUTOFF = 5.0


def contacts(path):
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    m = st[0]
    chains = {c.name: c for c in m}

    binder_atoms = []          # (chain, resnum, atom, pos)
    for cn in BINDER_CHAINS:
        if cn not in chains:
            continue
        for r in chains[cn]:
            for a in r:
                binder_atoms.append((cn, r.seqid.num, a.name, a.pos))

    # HER2 residues contacted by the binder
    contacted = set()
    cdrh3_n = 0
    for cn in TARGET_CHAINS:
        if cn not in chains:
            continue
        for r in chains[cn]:
            for a in r:
                for (bc, bn, _an, bp) in binder_atoms:
                    if a.pos.dist(bp) < CUTOFF:
                        contacted.add(f"{cn}{r.seqid.num}")
                        if bc == CDRH3[0] and bn in CDRH3[1]:
                            cdrh3_n += 1
                        break

    row = {"n_her2_contacted": len(contacted),
           "contacted_set": ";".join(sorted(contacted)),
           "cdrh3_contact_atoms": cdrh3_n}

    # hotspot geometry
    hot_hits = 0
    for (cn, rn, an) in HOTSPOTS:
        d = None
        if cn in chains:
            for r in chains[cn]:
                if r.seqid.num != rn:
                    continue
                for a in r:
                    if a.name != an:
                        continue
                    d = min((a.pos.dist(bp) for (_c, _n, _a, bp) in binder_atoms),
                            default=None)
        row[f"dist_{cn}{rn}_{an}"] = round(d, 2) if d is not None else None
        if d is not None and d < CUTOFF:
            hot_hits += 1
    row["hot_contacts"] = hot_hits
    row["hot_min_dist"] = min(
        [v for k, v in row.items() if k.startswith("dist_") and v is not None],
        default=None)
    return row


def main(root, out):
    rows = []
    for p in sorted(glob.glob(f"{root}/*/seed_*/*_model.cif")):
        qid = p.split(os.sep)[-3]
        r = contacts(p)
        r["query_id"] = qid
        rows.append(r)
    df = pd.DataFrame(rows)

    base = df[df.query_id == "BASELINE_trastuzumab"]
    if len(base):
        bset = set(base.iloc[0]["contacted_set"].split(";"))

        def jac(s):
            a = set(s.split(";"))
            return round(len(a & bset) / len(a | bset), 3) if (a | bset) else 0.0

        df["epitope_jaccard"] = df.contacted_set.map(jac)
    cols = ["query_id", "epitope_jaccard", "hot_contacts", "hot_min_dist",
            "n_her2_contacted", "cdrh3_contact_atoms"]
    df = df[cols + [c for c in df.columns if c not in cols]]
    df.to_csv(out, index=False)
    print(f"epitope analysis -> {out}  ({len(df)} structures)")
    print(df[cols].sort_values("epitope_jaccard", ascending=False).head(15).to_string(index=False))
    return df


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
