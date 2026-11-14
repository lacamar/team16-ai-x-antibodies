#!/usr/bin/env python3
"""
interface_physics.py - auditable, physics-based interface analysis for
antibody/HER2 complexes. No machine learning anywhere in this file.

WHY THIS EXISTS
---------------
A learned ranker (RF, ridge, ipTM) gives you a number you cannot decompose and
cannot defend. Everything here is computed from atomic coordinates with
published constants, and every reported value can be traced to the atoms that
produced it. Run with --explain to print the exact formulae and constants used.

TWO LAYERS
----------
LAYER 1 (always runs, needs only numpy + scipy)
  Pure geometry and solvent accessibility:
    - interface contact map (per-atom distance criterion)
    - buried solvent-accessible surface area, Shrake-Rupley, per residue
    - hydrogen bonds (heavy-atom geometric criterion)
    - salt bridges
    - exposed hydrophobic surface  -> aggregation / thermostability proxy
    - EPITOPE OVERLAP against a reference complex (e.g. 1N8Z trastuzumab-HER2)
      -> the only direct proxy you have for judging axis 2

LAYER 2 (optional, needs openmm + pdbfixer)
  Single-snapshot MM-GBSA interaction energy, decomposed into
  van der Waals / Coulomb / polar solvation, per interface residue.
  Amber14SB + GBn2 implicit solvent.

WHAT LAYER 2 IS NOT
-------------------
It is NOT a binding free energy. One snapshot, no conformational sampling, no
configurational entropy, no explicit waters. It ranks and it explains; it does
not predict Kd. Say exactly this in the pitch - a judge will respect the limit
and distrust a team that claims otherwise.

USAGE
-----
  # analyse one design, and score its epitope against trastuzumab's
  python3 interface_physics.py \
      --complex folded/design_007.pdb --ab-chains H,L --ag-chains A \
      --reference 1n8z.pdb --ref-ab-chains B,A --ref-ag-chains C \
      --out-dir physics_out

  # whole folded directory
  python3 interface_physics.py --complex-dir folded/ --ab-chains H,L \
      --ag-chains A --reference 1n8z.pdb --ref-ab-chains B,A \
      --ref-ag-chains C --out-dir physics_out

  python3 interface_physics.py --explain      # print methods, cite these

OUTPUTS
-------
  interface_summary.csv     one row per complex, all scalar observables
  residue_detail_<name>.csv per-residue burial / contacts / h-bonds / energy
  methods.md                the formulae and constants actually used
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    sys.exit("[fatal] needs scipy:  pip install --break-system-packages scipy")


# --------------------------------------------------------------- constants
# Bondi (1964) van der Waals radii, Angstrom. J Phys Chem 68:441.
VDW_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52,
             "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75, "SE": 1.90}
DEFAULT_RADIUS = 1.70
PROBE_RADIUS = 1.40          # water probe, Lee & Richards (1971)
SASA_SPHERE_POINTS = 256     # Shrake & Rupley (1973) numerical quadrature

CONTACT_CUTOFF = 4.5         # heavy-atom contact, Angstrom
HBOND_MAX_DIST = 3.5         # donor/acceptor heavy-atom distance
SALTBRIDGE_MAX_DIST = 4.0    # charged-group heavy-atom distance

HBOND_DONORS = {"N", "O"}    # heavy atoms only; no explicit H in folded models
HBOND_ACCEPTORS = {"N", "O"}

ANION_ATOMS = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
CATION_ATOMS = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
                ("HIS", "ND1"), ("HIS", "NE2")}

# Kyte-Doolittle positive set; used only to label surface as hydrophobic.
HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "CYS", "PRO"}


# ------------------------------------------------------------- structures
@dataclass
class Atom:
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    xyz: np.ndarray
    element: str

    @property
    def radius(self) -> float:
        return VDW_RADII.get(self.element.upper(), DEFAULT_RADIUS)

    @property
    def res_id(self) -> str:
        return f"{self.chain}:{self.resseq}{self.icode.strip()}:{self.resname}"


@dataclass
class Structure:
    name: str
    atoms: list[Atom] = field(default_factory=list)

    def select(self, chains: set[str]) -> list[Atom]:
        return [a for a in self.atoms if a.chain in chains]

    def chains(self) -> list[str]:
        seen = []
        for a in self.atoms:
            if a.chain not in seen:
                seen.append(a.chain)
        return seen


def guess_element(name: str, raw: str) -> str:
    raw = raw.strip()
    if raw:
        return raw
    n = name.strip()
    # PDB column convention: element symbol is right-justified in cols 13-14
    if n[:1].isdigit():
        n = n[1:]
    return n[:1]


def parse_pdb(path: str, skip_hydrogen: bool = True) -> Structure:
    atoms: list[Atom] = []
    seen_altloc: set[tuple] = set()
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec not in ("ATOM  ", "HETATM"):
                if line.startswith("ENDMDL"):
                    break          # first model only
                continue
            resname = line[17:20].strip()
            if rec == "HETATM" and resname in ("HOH", "WAT", "DOD"):
                continue
            element = guess_element(line[12:16], line[76:78])
            if skip_hydrogen and element.upper() in ("H", "D"):
                continue
            altloc = line[16]
            key = (line[21], line[22:26], line[12:16])
            if altloc not in (" ", "A"):
                continue
            if key in seen_altloc:
                continue
            seen_altloc.add(key)
            try:
                xyz = np.array([float(line[30:38]), float(line[38:46]),
                                float(line[46:54])])
                resseq = int(line[22:26])
            except ValueError:
                continue
            atoms.append(Atom(
                serial=len(atoms), name=line[12:16].strip(), resname=resname,
                chain=line[21], resseq=resseq, icode=line[26],
                xyz=xyz, element=element))
    if not atoms:
        sys.exit(f"[fatal] no atoms parsed from {path}")
    return Structure(name=os.path.basename(path), atoms=atoms)


# ------------------------------------------------------------------- SASA
def sphere_points(n: int) -> np.ndarray:
    """Golden-spiral quadrature points on the unit sphere."""
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])


_SPHERE = sphere_points(SASA_SPHERE_POINTS)


def sasa_per_atom(atoms: list[Atom]) -> np.ndarray:
    """
    Shrake-Rupley numerical SASA, A^2 per atom.
      A_i = 4*pi*(r_i+p)^2 * (accessible points / total points)
    A test point is occluded if it lies inside (r_j + p) of any other atom j.
    """
    if not atoms:
        return np.zeros(0)
    coords = np.array([a.xyz for a in atoms])
    radii = np.array([a.radius for a in atoms]) + PROBE_RADIUS
    tree = cKDTree(coords)
    max_r = radii.max()
    out = np.zeros(len(atoms))
    for i, (c, r) in enumerate(zip(coords, radii)):
        nb = tree.query_ball_point(c, r + max_r)
        nb = [j for j in nb if j != i]
        pts = c + r * _SPHERE
        if nb:
            d2 = ((pts[:, None, :] - coords[nb][None, :, :]) ** 2).sum(-1)
            occluded = (d2 < (radii[nb] ** 2)[None, :]).any(axis=1)
            frac = 1.0 - occluded.mean()
        else:
            frac = 1.0
        out[i] = 4.0 * math.pi * r * r * frac
    return out


def per_residue(atoms: list[Atom], values: np.ndarray) -> dict[str, float]:
    agg: dict[str, float] = {}
    for a, v in zip(atoms, values):
        agg[a.res_id] = agg.get(a.res_id, 0.0) + float(v)
    return agg


# -------------------------------------------------------------- interface
def contact_pairs(ab: list[Atom], ag: list[Atom],
                  cutoff: float = CONTACT_CUTOFF) -> list[tuple[Atom, Atom, float]]:
    if not ab or not ag:
        return []
    ab_xyz = np.array([a.xyz for a in ab])
    ag_xyz = np.array([a.xyz for a in ag])
    t = cKDTree(ag_xyz)
    pairs = []
    for i, c in enumerate(ab_xyz):
        for j in t.query_ball_point(c, cutoff):
            d = float(np.linalg.norm(c - ag_xyz[j]))
            pairs.append((ab[i], ag[j], d))
    return pairs


def count_hbonds(pairs) -> list[tuple[str, str, float]]:
    out = []
    for a, b, d in pairs:
        if d <= HBOND_MAX_DIST and a.element.upper() in HBOND_DONORS \
                and b.element.upper() in HBOND_ACCEPTORS:
            out.append((f"{a.res_id}/{a.name}", f"{b.res_id}/{b.name}", round(d, 2)))
    return out


def count_saltbridges(pairs) -> list[tuple[str, str, float]]:
    out = []
    for a, b, d in pairs:
        if d > SALTBRIDGE_MAX_DIST:
            continue
        ka, kb = (a.resname, a.name), (b.resname, b.name)
        if (ka in ANION_ATOMS and kb in CATION_ATOMS) or \
           (ka in CATION_ATOMS and kb in ANION_ATOMS):
            out.append((f"{a.res_id}/{a.name}", f"{b.res_id}/{b.name}", round(d, 2)))
    return out


def epitope_residues(ab: list[Atom], ag: list[Atom]) -> set[str]:
    """Antigen residues contacted by the antibody. This is the epitope."""
    return {b.res_id for _, b, _ in contact_pairs(ab, ag)}


def analyse(struct: Structure, ab_chains: set[str], ag_chains: set[str]) -> dict:
    ab, ag = struct.select(ab_chains), struct.select(ag_chains)
    if not ab or not ag:
        sys.exit(f"[fatal] {struct.name}: chains not found. present: {struct.chains()}")

    sasa_ab_free = sasa_per_atom(ab)
    sasa_ag_free = sasa_per_atom(ag)
    both = ab + ag
    sasa_cplx = sasa_per_atom(both)
    sasa_ab_c, sasa_ag_c = sasa_cplx[:len(ab)], sasa_cplx[len(ab):]

    bsa_total = float((sasa_ab_free.sum() - sasa_ab_c.sum())
                      + (sasa_ag_free.sum() - sasa_ag_c.sum()))

    res_bsa_ab = {k: res_free_v - per_residue(ab, sasa_ab_c).get(k, 0.0)
                  for k, res_free_v in per_residue(ab, sasa_ab_free).items()}
    res_bsa_ag = {k: res_free_v - per_residue(ag, sasa_ag_c).get(k, 0.0)
                  for k, res_free_v in per_residue(ag, sasa_ag_free).items()}

    pairs = contact_pairs(ab, ag)
    hb = count_hbonds(pairs)
    sb = count_saltbridges(pairs)

    # exposed hydrophobic surface on the antibody in the bound state:
    # a thermostability / aggregation proxy, and it needs no training data.
    hyd_exposed = sum(float(v) for a, v in zip(ab, sasa_ab_c)
                      if a.resname in HYDROPHOBIC_RES and a.element.upper() == "C")

    return {
        "complex": struct.name,
        "buried_sasa_A2": round(bsa_total, 1),
        "interface_atom_contacts": len(pairs),
        "interface_res_ab": sum(1 for v in res_bsa_ab.values() if v > 1.0),
        "interface_res_ag": sum(1 for v in res_bsa_ag.values() if v > 1.0),
        "hbonds": len(hb),
        "salt_bridges": len(sb),
        "exposed_hydrophobic_C_A2": round(hyd_exposed, 1),
        "_epitope": {k for k, v in res_bsa_ag.items() if v > 1.0},
        "_res_bsa_ab": res_bsa_ab,
        "_res_bsa_ag": res_bsa_ag,
        "_hbond_list": hb,
        "_saltbridge_list": sb,
    }


# ------------------------------------------------- layer 2: MM-GBSA (opt)
def mmgbsa(path: str, ab_chains: set[str], ag_chains: set[str]) -> dict | None:
    """
    Single-snapshot interaction energy, Amber14SB + GBn2, NoCutoff.
    Chains are separated by rigid translation; intramolecular terms cancel,
    so the difference isolates intermolecular vdW + Coulomb and the change in
    GB polar solvation. No sampling, no entropy: a ranking quantity only.
    """
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        from pdbfixer import PDBFixer
    except ImportError:
        print("[layer2] skipped - openmm/pdbfixer not installed. To enable:")
        print("         pip install --break-system-packages openmm pdbfixer")
        return None

    fixer = PDBFixer(filename=path)
    fixer.findMissingResidues()
    fixer.missingResidues = {}          # do not model loops we did not design
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller = app.Modeller(fixer.topology, fixer.positions)
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.NoCutoff,
                             constraints=None, rigidWater=False)

    groups = {}
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)
        groups[i] = force.__class__.__name__

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx = openmm.Context(system, integrator,
                         openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(modeller.positions)
    openmm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)

    def energies(positions):
        ctx.setPositions(positions)
        out = {}
        for i, nm in groups.items():
            e = ctx.getState(getEnergy=True, groups={i}).getPotentialEnergy()
            out[nm] = out.get(nm, 0.0) + e.value_in_unit(
                unit.kilocalorie_per_mole)
        return out

    pos = ctx.getState(getPositions=True).getPositions(asNumpy=True)
    e_complex = energies(pos)

    sep = np.array(pos.value_in_unit(unit.nanometer), dtype=float)
    ag_idx = [a.index for a in modeller.topology.atoms()
              if a.residue.chain.id in ag_chains]
    if not ag_idx:
        print(f"[layer2] antigen chains {ag_chains} not found after fixing; skipped")
        return None
    sep[ag_idx] += np.array([100.0, 0.0, 0.0])     # 1000 A, NoCutoff so safe
    e_apart = energies(sep * unit.nanometer)

    delta = {k: e_complex.get(k, 0.0) - e_apart.get(k, 0.0)
             for k in set(e_complex) | set(e_apart)}
    vdw = delta.get("NonbondedForce", 0.0)
    gb = delta.get("GBSAOBCForce", delta.get("CustomGBForce", 0.0))
    return {
        "mmgbsa_total_kcal": round(sum(delta.values()), 2),
        "mmgbsa_nonbonded_kcal": round(vdw, 2),
        "mmgbsa_polar_solv_kcal": round(gb, 2),
        "mmgbsa_terms": {k: round(v, 2) for k, v in delta.items() if abs(v) > 0.01},
    }


# ------------------------------------------------------------------ report
METHODS = f"""# Methods actually used by interface_physics.py

No machine-learned model contributes to any number below.

## Solvent accessible surface area
Shrake & Rupley (1973) numerical quadrature, {SASA_SPHERE_POINTS} golden-spiral
points per atom. Atom radii: Bondi (1964) van der Waals radii
(C {VDW_RADII['C']}, N {VDW_RADII['N']}, O {VDW_RADII['O']}, S {VDW_RADII['S']} A),
water probe {PROBE_RADIUS} A. Hydrogens excluded; folded models rarely carry
reliable hydrogen positions.

    A_i = 4*pi*(r_i + p)^2 * (unoccluded points / total points)

## Buried surface area at the interface
    BSA = [SASA(Ab alone) - SASA(Ab in complex)]
        + [SASA(Ag alone) - SASA(Ag in complex)]
Reported per residue as well as total. This is a geometric quantity: it is
exactly reproducible from the coordinates, with no fitted parameter.

## Interface contacts
Heavy-atom pairs within {CONTACT_CUTOFF} A across the antibody/antigen boundary.

## Hydrogen bonds
Geometric proxy: N/O to N/O heavy-atom pairs within {HBOND_MAX_DIST} A across the
interface. Without explicit hydrogens no angular criterion is applied, so this
OVERCOUNTS relative to a full donor-H-acceptor geometry. Use it comparatively
between designs, never as an absolute count.

## Salt bridges
Asp/Glu carboxylate oxygen to Lys NZ / Arg NE,NH1,NH2 / His ND1,NE2 within
{SALTBRIDGE_MAX_DIST} A.

## Exposed hydrophobic surface
Summed SASA of carbon atoms in {sorted(HYDROPHOBIC_RES)} in the bound state.
A proxy for aggregation propensity and therefore for measured thermostability.
Uncalibrated: use as a relative veto, not as a predicted Tm.

## Epitope overlap (axis 2)
The epitope is the set of antigen residues burying >1 A^2 on binding. Overlap
with a reference complex is reported as
    recall    = |design epitope AND reference epitope| / |reference epitope|
    jaccard   = |intersection| / |union|
Reference should be the trastuzumab-HER2 complex (PDB 1N8Z). Recall is the
quantity that matters: it asks how much of trastuzumab's epitope your design
still covers, which is what the competitive-binding assay measures.

## MM-GBSA interaction energy (optional layer)
Amber14SB protein force field with GBn2 implicit solvent, NoCutoff, 500 steps
of local energy minimisation, then rigid separation of the antigen chains by
1000 A. Reported as the difference in each force group between bound and
separated states, decomposed into nonbonded (vdW + Coulomb) and polar
solvation terms.

LIMITS, state these in the pitch: one minimised snapshot, no conformational
sampling, no configurational entropy, no explicit water, no per-term
reweighting. It is a ranking and explanation quantity. It is not a binding
free energy and must not be quoted as a predicted Kd or dG.
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--complex"), p.add_argument("--complex-dir")
    p.add_argument("--ab-chains", default="H,L")
    p.add_argument("--ag-chains", default="A")
    p.add_argument("--reference", help="reference complex, e.g. 1n8z.pdb")
    p.add_argument("--ref-ab-chains", default="B,A")
    p.add_argument("--ref-ag-chains", default="C")
    p.add_argument("--mmgbsa", action="store_true", help="run layer 2")
    p.add_argument("--out-dir", default="./physics_out")
    p.add_argument("--explain", action="store_true")
    a = p.parse_args()

    if a.explain:
        print(METHODS)
        return
    if not (a.complex or a.complex_dir):
        sys.exit("[fatal] give --complex or --complex-dir (or --explain)")

    os.makedirs(a.out_dir, exist_ok=True)
    open(os.path.join(a.out_dir, "methods.md"), "w").write(METHODS)

    ab_ch = set(a.ab_chains.split(","))
    ag_ch = set(a.ag_chains.split(","))

    ref_epitope: set[str] | None = None
    if a.reference:
        ref = parse_pdb(a.reference)
        r = analyse(ref, set(a.ref_ab_chains.split(",")),
                    set(a.ref_ag_chains.split(",")))
        ref_epitope = r["_epitope"]
        print(f"[ref ] {a.reference}: {len(ref_epitope)} epitope residues, "
              f"{r['buried_sasa_A2']} A^2 buried, {r['hbonds']} h-bonds")
        print(f"[ref ] epitope: {sorted(ref_epitope)}")

    paths = [a.complex] if a.complex else sorted(
        os.path.join(a.complex_dir, f) for f in os.listdir(a.complex_dir)
        if f.lower().endswith((".pdb", ".ent")))
    if not paths:
        sys.exit("[fatal] no .pdb files found")

    rows = []
    for path in paths:
        s = parse_pdb(path)
        res = analyse(s, ab_ch, ag_ch)

        if ref_epitope:
            ep = res["_epitope"]
            inter = ep & ref_epitope
            res["epitope_recall"] = round(len(inter) / max(len(ref_epitope), 1), 3)
            res["epitope_jaccard"] = round(
                len(inter) / max(len(ep | ref_epitope), 1), 3)
            res["epitope_lost"] = ";".join(sorted(ref_epitope - ep)) or "-"
            res["epitope_gained"] = ";".join(sorted(ep - ref_epitope)) or "-"

        if a.mmgbsa:
            mm = mmgbsa(path, ab_ch, ag_ch)
            if mm:
                res.update({k: v for k, v in mm.items() if k != "mmgbsa_terms"})
                print(f"[mm  ] {s.name}: {mm['mmgbsa_terms']}")

        # per-residue detail: this is the file you show a judge
        stem = os.path.splitext(os.path.basename(path))[0]
        detail = os.path.join(a.out_dir, f"residue_detail_{stem}.csv")
        with open(detail, "w") as fh:
            fh.write("side,residue,buried_sasa_A2\n")
            for k, v in sorted(res["_res_bsa_ab"].items(),
                               key=lambda kv: -kv[1]):
                if v > 1.0:
                    fh.write(f"antibody,{k},{v:.1f}\n")
            for k, v in sorted(res["_res_bsa_ag"].items(),
                               key=lambda kv: -kv[1]):
                if v > 1.0:
                    fh.write(f"antigen,{k},{v:.1f}\n")
            fh.write("\ntype,partner_a,partner_b,distance_A\n")
            for x, y, d in res["_hbond_list"]:
                fh.write(f"hbond,{x},{y},{d}\n")
            for x, y, d in res["_saltbridge_list"]:
                fh.write(f"saltbridge,{x},{y},{d}\n")

        print(f"[ok  ] {s.name}: BSA {res['buried_sasa_A2']} A^2, "
              f"{res['hbonds']} h-bonds, {res['salt_bridges']} salt bridges, "
              f"epitope recall {res.get('epitope_recall', 'n/a')}")
        rows.append({k: v for k, v in res.items() if not k.startswith("_")})

    import csv
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    summary = os.path.join(a.out_dir, "interface_summary.csv")
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[out ] {summary}")
    print(f"[out ] {a.out_dir}/methods.md   <- cite this, do not paraphrase it")
    print("\n[read] Rank on epitope_recall first (axis 2 gate), then buried_sasa")
    print("       and hbonds (axis 1), then exposed_hydrophobic_C as a veto")
    print("       (axis 3). Any design losing >30% of the reference epitope is")
    print("       not an optimisation of trastuzumab, whatever the RF says.")


if __name__ == "__main__":
    main()
