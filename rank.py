"""Combined re-ranker: gates -> surrogate (primary) + epitope bonus -> diversity dedup -> top-20.

CSV columns: cdrh3, vh_sequence, [iptm, min_ipae, design_cif]
Usage: python rank.py candidates.csv [out.csv]   |   python rank.py   (demo)
"""
import csv, sys
from pathlib import Path

import novelty_filter as nf
from affinity_surrogate import AffinitySurrogate
from epitope_check import preservation
from thermostability import stability_score

HERE = Path(__file__).resolve().parent
FACS = HERE / "data/her2/tz_her2_facs_3classes_524346.csv.zip"
SPR = HERE / "data/her2/trastuzumab_her2_spr_all_768.csv"

TOP_N = 20
IPTM_FLOOR = 0.6          # geometry gate only (Spearman(ipTM, affinity)=0.285: weak)
MAX_IPAE = None           # optional gate on min_ipae (lower better), e.g. 15.0
W_SURROGATE = 1.0         # primary ranker (Spearman 0.72)
W_EPITOPE = 1.5           # bonus: * epitope_recall
W_HOTSPOT = 0.25          # bonus per hit of the 4 hotspots (C572/C573/D591/D593)
EPITOPE_MIN_RECALL = 0.3  # below this: epitope penalty
EPITOPE_PENALTY = 1.0
W_IPTM = 0.2              # tiny tie-break: * (iptm - IPTM_FLOOR)
W_STABILITY = 0.75        # penalise destabilising loops: -W * (1 - stability_score)
MIN_DIST = 2              # diversity: min CDRH3 hamming between two picks


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rank(cands, top_n=TOP_N, surrogate=None):
    by_len = nf.load_provided_cdrh3(FACS, SPR)
    rows = []
    for c in cands:
        r = {"cdrh3": c["cdrh3"].strip().upper(), "selected": False, "final_score": None, "reason": ""}
        iptm, ipae = _f(c.get("iptm")), _f(c.get("min_ipae"))
        ok, why = nf.is_novel(r["cdrh3"], by_len)
        if not ok:
            r["reason"] = "rejected: " + why
        else:
            ok, why = nf.hard_checks(c["vh_sequence"])
            if not ok:
                r["reason"] = "rejected: " + why
            elif iptm is not None and iptm < IPTM_FLOOR:
                r["reason"] = f"rejected: ipTM {iptm:.2f} < {IPTM_FLOOR} (bad fold)"
            elif MAX_IPAE is not None and ipae is not None and ipae > MAX_IPAE:
                r["reason"] = f"rejected: min_ipAE {ipae:.1f} > {MAX_IPAE}"
        r["_gated"] = not r["reason"]
        r["_c"], r["_iptm"] = c, iptm
        rows.append(r)

    live = [r for r in rows if r["_gated"]]
    if live:
        surrogate = surrogate or AffinitySurrogate().fit()
    for r in live:
        s = surrogate.score(r["cdrh3"])
        score = W_SURROGATE * s["surrogate"]
        note = f"surrogate {s['surrogate']:.2f} (pKd {s['pkd']:.2f})"
        cif = r["_c"].get("design_cif")
        if cif:
            p = preservation(cif)
            score += W_EPITOPE * p["epitope_recall"] + W_HOTSPOT * len(p["hotspots_hit"])
            note += f", epitope recall {p['epitope_recall']:.2f}, hotspots {len(p['hotspots_hit'])}/4"
            if p["epitope_recall"] < EPITOPE_MIN_RECALL:
                score -= EPITOPE_PENALTY
                note += " [off-epitope penalty]"
        else:
            note += ", no structure (epitope unchecked)"
        stab = stability_score(r["cdrh3"])
        score -= W_STABILITY * (1.0 - stab)
        note += f", stability {stab:.2f}"
        if r["_iptm"] is not None:
            score += W_IPTM * (r["_iptm"] - IPTM_FLOOR)
        r["final_score"], r["_note"] = round(score, 3), note

    picked = []
    for r in sorted(live, key=lambda r: (-r["final_score"], r["cdrh3"])):
        near = next((p for p in picked if nf.hamming(r["cdrh3"], p["cdrh3"]) < MIN_DIST), None)
        if near:
            r["reason"] = f"rejected: near-duplicate of {near['cdrh3']} (hamming<{MIN_DIST})"
        elif len(picked) >= top_n:
            r["reason"] = f"rejected: below top-{top_n} cutoff; {r['_note']}"
        else:
            picked.append(r)
            r["selected"] = True
            r["reason"] = f"pick #{len(picked)}: {r['_note']}"
    rows.sort(key=lambda r: (not r["selected"], -(r["final_score"] if r["final_score"] is not None else -1e9), r["cdrh3"]))
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def table(rows):
    print(f"{'sel':3} {'score':>7}  {'cdrh3':20} reason")
    for r in rows:
        sc = "-" if r["final_score"] is None else f"{r['final_score']:.3f}"
        print(f"{'Y' if r['selected'] else 'n':3} {sc:>7}  {r['cdrh3']:20} {r['reason']}")


def demo():
    spr = list(csv.DictReader(open(SPR)))
    tz = next(r for r in spr if r["cdrh3"] == nf.TRASTUZUMAB_CDRH3)["vh_sequence"]
    vh = lambda c: tz.replace(nf.TRASTUZUMAB_CDRH3, c)
    cif = lambda i: str(HERE / f"data/structures/her2_cdrh3_{i}_generated.cif")
    spr_known = next(r["cdrh3"] for r in spr if r["cdrh3"] != nf.TRASTUZUMAB_CDRH3 and r["kd_nM"])
    cands = [
        {"cdrh3": nf.TRASTUZUMAB_CDRH3, "vh_sequence": tz, "iptm": 0.9},
        {"cdrh3": spr_known, "vh_sequence": vh(spr_known), "iptm": 0.8},
        {"cdrh3": "WASSSEYYFD", "vh_sequence": vh("WASSSEYYFD"), "iptm": 0.75, "design_cif": cif(0)},
        {"cdrh3": "WGPPYFYAMD", "vh_sequence": vh("WGPPYFYAMD"), "iptm": 0.70, "design_cif": cif(1)},
        {"cdrh3": "WGPPYFYAMN", "vh_sequence": vh("WGPPYFYAMN"), "iptm": 0.72},
        {"cdrh3": "WSSPDWYAMD", "vh_sequence": vh("WSSPDWYAMD"), "iptm": 0.45},
    ]
    rows = rank(cands)
    table(rows)
    by = {r["cdrh3"]: r for r in rows}
    assert not by[nf.TRASTUZUMAB_CDRH3]["selected"] and "rejected" in by[nf.TRASTUZUMAB_CDRH3]["reason"]
    assert not by[spr_known]["selected"]
    assert "ipTM" in by["WSSPDWYAMD"]["reason"]
    assert "near-duplicate" in by["WGPPYFYAMN"]["reason"] or by["WGPPYFYAMN"]["selected"]
    assert rows == rank(cands), "must be deterministic"
    assert any(r["selected"] for r in rows)
    print("self-check passed")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        out = rank(list(csv.DictReader(open(sys.argv[1]))))
        table(out)
        if len(sys.argv) > 2:
            with open(sys.argv[2], "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(out[0]))
                w.writeheader(); w.writerows(out)
    else:
        demo()
