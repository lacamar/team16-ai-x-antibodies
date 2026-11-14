# Methods actually used by interface_physics.py

No machine-learned model contributes to any number below.

## Solvent accessible surface area
Shrake & Rupley (1973) numerical quadrature, 256 golden-spiral
points per atom. Atom radii: Bondi (1964) van der Waals radii
(C 1.7, N 1.55, O 1.52, S 1.8 A),
water probe 1.4 A. Hydrogens excluded; folded models rarely carry
reliable hydrogen positions.

    A_i = 4*pi*(r_i + p)^2 * (unoccluded points / total points)

## Buried surface area at the interface
    BSA = [SASA(Ab alone) - SASA(Ab in complex)]
        + [SASA(Ag alone) - SASA(Ag in complex)]
Reported per residue as well as total. This is a geometric quantity: it is
exactly reproducible from the coordinates, with no fitted parameter.

## Interface contacts
Heavy-atom pairs within 4.5 A across the antibody/antigen boundary.

## Hydrogen bonds
Geometric proxy: N/O to N/O heavy-atom pairs within 3.5 A across the
interface. Without explicit hydrogens no angular criterion is applied, so this
OVERCOUNTS relative to a full donor-H-acceptor geometry. Use it comparatively
between designs, never as an absolute count.

## Salt bridges
Asp/Glu carboxylate oxygen to Lys NZ / Arg NE,NH1,NH2 / His ND1,NE2 within
4.0 A.

## Exposed hydrophobic surface
Summed SASA of carbon atoms in ['ALA', 'CYS', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'TRP', 'VAL'] in the bound state.
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
