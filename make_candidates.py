"""refold_metrics.csv (Boltz-2 refold-score) -> candidates.csv for rank.py.
Usage: python make_candidates.py refold_metrics.csv [candidates.csv]
CDRH3 rule: residues after the last `C-x-[RK]` and before `YWG.G` (FR4 start; the Y is
residue 102, excluded to match trastuzumab's WGGDGFYAMD). Override with --pattern.
"""
import argparse, csv, re
from pathlib import Path

PATTERN = r"^.*C.[RK](.+?)YWG.G"


def extract_cdrh3(vh, pattern=PATTERN):
    m = re.match(pattern, vh)
    return m.group(1) if m else ""


def resolve_cif(design, metrics):
    p = Path(design)
    if p.is_absolute():
        return str(p)
    for base in [metrics.parent, *metrics.parent.parents]:
        if (base / p).exists():
            return str((base / p).resolve())
    return design


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("out", nargs="?", default="candidates.csv")
    ap.add_argument("--pattern", default=PATTERN, help="regex; group 1 = CDRH3")
    a = ap.parse_args()
    metrics = Path(a.metrics).resolve()
    rows = []
    for r in csv.DictReader(open(metrics)):
        vh = (r.get("binder_sequence") or "").strip().upper()
        if r.get("status") != "predicted" or not vh:
            continue
        cdr = extract_cdrh3(vh, a.pattern)
        if not cdr:
            continue
        rows.append({"cdrh3": cdr, "vh_sequence": vh, "iptm": r["iptm"], "min_ipae": r["min_ipae"],
                     "design_cif": resolve_cif(r["design"], metrics)})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, ["cdrh3", "vh_sequence", "iptm", "min_ipae", "design_cif"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} candidates -> {a.out}")


if __name__ == "__main__":
    TZ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
    assert extract_cdrh3(TZ) == "WGGDGFYAMD"
    main()
