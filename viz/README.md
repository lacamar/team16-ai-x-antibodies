# 3D dashboard (static)

`dashboard.html` — self-contained (3Dmol.js from cdnjs). Open in any browser.
Design selector switches between the folded structures in `designs.json`
(antibody chains A/B docked to HER2 C/D, hotspots C572/C573/D591/D593).

Rebuild after editing `designs.json`: `python3 rebuild.py` (patches DESIGNS + render in place).
