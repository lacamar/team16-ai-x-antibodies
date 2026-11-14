"""Build + fire the StaB-ddG run for the diversified (joint-search) pool.

Why: StaB-ddG is physics-based and shares no fit with any of our ML scorers.
F7 measured it ANTI-correlating with the SPR RF (rho=-0.405) and with Boltz-2
ipTM (-0.338) on the first shortlist. The joint pool is uniformly worse than
trastuzumab on the SPR RF by construction (joint_search.py: 0 of 731 beat it),
so StaB-ddG is the one scorer that could independently support the diversified
tail -- or refute it. Either answer is worth 12 minutes of lane B, and lane B
does not contend for the teammate-held H100.

Includes three AntiFold-argmax sequences as declared NEGATIVE controls: they
are what one generator wants and the other three models reject, so if
StaB-ddG rates them well that is evidence about StaB-ddG, not about them.

Reuses the exact entry/param shape of op-79006e6f713984a7 (semantic type
stab-ddg-workload-tar/v1; partner_a = antibody A+B, partner_b = HER2 C+D).
"""
import json
import os
import subprocess
import tarfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FORGE = "/home/hackathon/ubaydolla/nebius-kit/agent/nebius-scientific-ai/forge.py"
CREDS = "/home/hackathon/ubaydolla/nebius-kit/credentials.private.json"
TZ = "WGGDGFYAMD"
RES = list(range(99, 109))
LABEL = "stabddg-joint42"

# AntiFold-argmax sequences: declared negative controls (see docstring)
AF_CONTROLS = ["EGGIYGYALD", "EGGIYGYAMD", "YGGIYGYALD"]


def muts(cdrh3):
    return [{"chain": "B", "residue_number": RES[p], "insertion_code": "",
             "wild_type": TZ[p], "mutant": cdrh3[p]}
            for p in range(10) if cdrh3[p] != TZ[p]]


def main():
    div = pd.read_csv(f"{HERE}/runs/joint_candidates.csv")
    variants, manifest = [], []
    for i, r in enumerate(div.itertuples()):
        vid = f"joint_{i:02d}"
        variants.append({"id": vid, "mutations": muts(r.cdrh3)})
        manifest.append({"variant_id": vid, "cdrh3": r.cdrh3,
                         "mut_positions": r.mut_positions, "role": "candidate"})
    for s in AF_CONTROLS:
        vid = f"CTRL_afargmax_{s}"
        variants.append({"id": vid, "mutations": muts(s)})
        manifest.append({"variant_id": vid, "cdrh3": s,
                         "mut_positions": "+".join(
                             str(RES[p]) for p in range(10) if s[p] != TZ[p]),
                         "role": "negative_control_antifold_argmax"})

    params = {"pdb_file": "complex.pdb", "partner_a": ["A", "B"],
              "partner_b": ["C", "D"], "variants": variants}
    pj = f"{HERE}/nebius-runs/stabddg_joint_params.json"
    json.dump(params, open(pj, "w"), indent=1)
    pd.DataFrame(manifest).to_csv(f"{HERE}/nebius-runs/stabddg_joint_manifest.csv",
                                  index=False)
    print(f"{len(variants)} variants ({len(div)} candidates + "
          f"{len(AF_CONTROLS)} controls) -> {pj}")

    tar = f"{HERE}/nebius-runs/inputs/stabddg_joint.tar"
    with tarfile.open(tar, "w") as t:
        t.add(f"{HERE}/nebius-runs/inputs/complex.pdb", arcname="complex.pdb")
    print(f"wrote {tar}")

    cmd = ["python3", FORGE, "--creds", CREDS,
           "submit", "stab-ddg",
           "--entry", f"workload|stab-ddg-workload-tar/v1|{tar}",
           "--params", json.dumps(params),
           "--operation", "predict-binding-ddg",
           "--tool", "submit_stab_ddg",
           "--label", LABEL, "--no-wait"]
    print("\n$ forge submit stab-ddg ... --label", LABEL)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode:
        print("STDERR:", r.stderr[-3000:])
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
