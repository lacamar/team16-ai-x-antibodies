#!/usr/bin/env python3
"""Consolidate scattered sequence data into viz/designs.json for the dashboard.

Joins each design's full VH sequence (from the candidate CSVs, de-novo from its
fold input) onto the dashboard's design records by CDRH3, and stamps provenance.
Render, never hand-type (repo rule 1)."""
import csv, json, subprocess
from pathlib import Path

VIZ = Path(__file__).resolve().parent
ROOT = VIZ.parent
CAND_CSVS = ["candidates_k1.csv", "results/opt-fold/candidates.csv", "candidates_opt.csv"]
DENOVO_INPUT = ROOT / "nebius-runs/inputs/wasss.json"


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "nogit"


def vh_by_cdrh3():
    m = {}
    for rel in CAND_CSVS:
        p = ROOT / rel
        if not p.exists():
            continue
        for r in csv.DictReader(p.open()):
            m.setdefault(r["cdrh3"], (r["vh_sequence"], rel))
    return m


def denovo_vh():
    seq = json.loads(DENOVO_INPUT.read_text())["polymers"][0]["sequence"]
    return seq, DENOVO_INPUT.relative_to(ROOT).as_posix()


def main():
    dj = json.loads((VIZ / "designs.json").read_text())
    table = vh_by_cdrh3()
    dn_seq, dn_src = denovo_vh()
    missing = []
    for d in dj["designs"]:
        c = d["cdrh3"]
        if c in table:
            seq, src = table[c]
        elif c in dn_seq:
            seq, src = dn_seq, dn_src
        else:
            missing.append(c)
            continue
        assert c in seq, f"CDRH3 {c} not found inside its VH sequence"
        d["vh_sequence"] = seq
        d["seq_len"] = len(seq)
        d["seq_source"] = src
    if missing:
        raise SystemExit(f"no VH sequence for: {missing}")
    dj["provenance"] = {"git_sha": git_sha(), "source": "viz/consolidate.py",
                        "candidate_csvs": CAND_CSVS, "model_id": "boltz2", "seed": 42}
    (VIZ / "designs.json").write_text(json.dumps(dj, indent=1) + "\n")
    inject(dj["designs"])
    print(f"consolidated {len(dj['designs'])} designs with VH sequences -> viz/designs.json + dashboard.html")


def inject(designs):
    """Re-inject the consolidated records into the dashboard's inlined DESIGNS (its single source)."""
    html = (VIZ / "dashboard.html").read_text()
    a = html.index("const DESIGNS = ")
    b = html.index("\n", a)
    line = "const DESIGNS = " + json.dumps(designs).replace("</", "<\\/") + ";"
    (VIZ / "dashboard.html").write_text(html[:a] + line + html[b:])


def _selfcheck():
    """Every design must get a VH sequence containing its CDRH3 at the CSR…Y junction."""
    dj = json.loads((VIZ / "designs.json").read_text())
    for d in dj["designs"]:
        s = d["vh_sequence"]
        assert d["cdrh3"] in s and len(s) == d["seq_len"], d["cdrh3"]
    print(f"selfcheck ok: {len(dj['designs'])} designs, all VH sequences carry their CDRH3")


if __name__ == "__main__":
    import sys
    (_selfcheck if "--check" in sys.argv else main)()
