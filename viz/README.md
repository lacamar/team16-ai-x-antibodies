# 3D dashboard (static)

`dashboard.html` — self-contained page (3Dmol.js from cdnjs). Serve this folder and open it
(`python3 -m http.server` then the dashboard.html), so the viewer can fetch the structures in `cif/`.

Design selector switches between 18 proteins: the optimisation top-20 (those with a folded
structure) plus the de-novo VHH WASSSEYYFDY. Each shows the antibody (VL blue / VH violet)
docked to HER2 (orange), hotspots C572/C573/D591/D593 in red, and its scores.

`designs.json` is the metadata; `cif/optNN.cif` / `cif/denovo_wasss.cif` are the structures.
