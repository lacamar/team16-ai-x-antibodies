"""De-novo VHH (nanobody) ranker for the HER2 de-novo track.

NOT the trastuzumab affinity surrogate: affinity_surrogate.py is trained only on trastuzumab
CDRH3 variants (one framework, 10-mer loops) and does not transfer to a VHH on a different
framework with three free CDRs. Ranking here is structure/interface + epitope + developability
+ novelty only.

Hard gates: CDR3 novelty (novelty_filter.is_novel), novelty_filter.hard_checks, global identity
< MAX_IDENT to every known reference, >= MIN_HALLMARKS VHH FR2 hallmarks, ipTM floor.
Score: iptm - 0.05*min_ipae + epitope recall vs 1N8Z trastuzumab epitope + hotspots
       + CDR developability - liabilities - hallmark/identity shortfall.

Input: refold_metrics.csv straight from refold-score, or CSV with columns id, vhh_sequence, [iptm, min_ipae, design_cif]   (design_cif = RFD4 *_generated.cif,
which keeps HER2 numbering 489-607 on chains C/D; any other chain is taken as the VHH).
Usage: python denovo_rank.py candidates.csv [out.csv]   |   python denovo_rank.py   (self-check)
Deps: gemmi, numpy, scipy (scribe venv).
"""
import csv, re, sys
from pathlib import Path

import gemmi

import novelty_filter as nf
from epitope_check import REF, HOTSPOTS, contact_epitope
from thermostability import stability_score

HERE = Path(__file__).resolve().parent
FACS = HERE / "data/her2/tz_her2_facs_3classes_524346.csv.zip"
SPR = HERE / "data/her2/trastuzumab_her2_spr_all_768.csv"
KNOWN_FASTA = HERE / "data/denovo/known_antibodies.fasta"   # optional: add SAbDab/INDI/OAS seqs here

FR = ["EVQLVESGGGLVQPGGSLRLSC", "WFRQAPGKGRELVA", "YPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYC", "WGQGTQVTVS"]
SPLIT = re.compile(r"^(.{18,26}?C)(.{8,14}?)(W.RQ.{10})(.{8,11}?)(.{30,40}?C)(.{10,24}?)(W[GA]QG.*)$")
HALLMARKS = {1: "FY", 8: "EQ", 9: "RK", 11: "FGLY"}   # FR2 idx of Kabat 37/44/45/47: VHH-like (hydrophilic) residues

TOP_N = 20
MAX_IDENT = 0.70          # task rule: <70% identity to ANY known antibody
MIN_HALLMARKS = 2
IPTM_FLOOR = 0.6
W_IPTM, W_IPAE = 1.0, 0.05
W_EPITOPE, W_HOTSPOT = 1.5, 0.25
EPITOPE_MIN_RECALL, EPITOPE_PENALTY = 0.3, 1.0
W_DEV, W_LIAB, W_HALL, W_IDENT = 0.75, 0.2, 0.15, 2.0
IDENT_SOFT = 0.60         # identity above this is penalised before the hard gate bites
MIN_DIST = 3              # CDR3 diversity between two picks
AG = {"C", "D"}


def parse_vhh(seq):
    m = SPLIT.match(seq)
    return dict(zip(("fr1", "c1", "fr2", "c2", "fr3", "c3", "fr4"), m.groups())) if m else None


def identity(q, r):
    """Global alignment (match+1, mismatch-1, gap-0.5): matches / len(q)."""
    n, m = len(q), len(r)
    S = [[(0.0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        S[i][0] = (-0.5 * i, 0)
    for j in range(1, m + 1):
        S[0][j] = (-0.5 * j, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            eq = q[i - 1] == r[j - 1]
            d = (S[i - 1][j - 1][0] + (1 if eq else -1), S[i - 1][j - 1][1] + eq)
            u = (S[i - 1][j][0] - 0.5, S[i - 1][j][1])
            l = (S[i][j - 1][0] - 0.5, S[i][j - 1][1])
            S[i][j] = max(d, u, l)
    return S[n][m][1] / n


def references():
    refs = {"template_framework": "".join(FR)}
    st = gemmi.read_structure(str(REF))
    refs["trastuzumab_VH"] = gemmi.one_letter_code([r.name for r in st[0]["B"]])
    if KNOWN_FASTA.exists():
        name = None
        for ln in KNOWN_FASTA.read_text().split("\n"):
            if ln.startswith(">"):
                name = ln[1:].strip(); refs[name] = ""
            elif name:
                refs[name] += ln.strip().upper()
    return refs


def hallmarks(fr2):
    return sum(fr2[i] in ok for i, ok in HALLMARKS.items())


def liabilities(*cdrs):
    s = "-".join(cdrs)
    return 2 * len(re.findall(r"N[^P][ST]", s)) + len(re.findall(r"NG|DG|DP", s))


def epitope(cif):
    st = gemmi.read_structure(str(cif))
    ab = {c.name for c in st[0]} - AG
    ref, des = contact_epitope(REF), contact_epitope(cif, ab=ab, ag=AG)
    return len(des & ref) / len(ref) if ref else 0.0, len(des & HOTSPOTS)


def load(path):
    rows = list(csv.DictReader(open(path)))
    if rows and "binder_sequence" in rows[0]:
        from make_candidates import resolve_cif
        path = Path(path).resolve()
        return [{"id": r["query_id"], "vhh_sequence": r["binder_sequence"], "iptm": r["iptm"], "min_ipae": r["min_ipae"],
                 "design_cif": resolve_cif(r["design"], path.parent.parent / "design" / "x")} for r in rows if r.get("status") == "predicted"]
    return rows


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rank(cands, top_n=TOP_N):
    by_len, refs = nf.load_provided_cdrh3(FACS, SPR), references()
    rows = []
    for c in cands:
        seq = c["vhh_sequence"].strip().upper()
        r = {"id": c["id"], "vhh_sequence": seq, "selected": False, "final_score": None, "reason": ""}
        iptm, ipae = _f(c.get("iptm")), _f(c.get("min_ipae"))
        p = parse_vhh(seq)
        why = ""
        if not p:
            why = "cannot parse FR/CDR layout"
        else:
            ok, why = nf.is_novel(p["c3"], by_len)
            why = "" if ok else "CDR3 " + why
            if not why:
                ok, why = nf.hard_checks(seq)
                why = "" if ok else why
            qk = {seq[i:i + 3] for i in range(len(seq) - 2)}
            near = sorted(refs, key=lambda k: -len(qk & {refs[k][i:i + 3] for i in range(len(refs[k]) - 2)}))[:40]  # k-mer prefilter; full alignment vs all refs is too slow
            ids = {k: identity(seq, refs[k]) for k in near}
            top = max(ids, key=ids.get)
            hall = hallmarks(p["fr2"])
            if not why and hall < MIN_HALLMARKS:
                why = f"VHH hallmarks {hall}/4 < {MIN_HALLMARKS} (VH-like FR2)"
            if not why and ids[top] >= MAX_IDENT:
                why = f"identity {ids[top]:.2f} >= {MAX_IDENT} to {top}"
            if not why and iptm is not None and iptm < IPTM_FLOOR:
                why = f"ipTM {iptm:.2f} < {IPTM_FLOOR}"
        r["reason"] = "rejected: " + why if why else ""
        r["_p"], r["_iptm"], r["_ipae"], r["_cif"] = p, iptm, ipae, c.get("design_cif")
        if not why:
            r["_ident"], r["_hall"] = ids[top], hall
        rows.append(r)

    live = [r for r in rows if not r["reason"]]
    for r in live:
        p = r["_p"]
        dev = sum(stability_score(p[k]) for k in ("c1", "c2", "c3")) / 3
        liab = liabilities(p["c1"], p["c2"], p["c3"])
        score = (W_IPTM * (r["_iptm"] or 0) - W_IPAE * (r["_ipae"] or 0)
                 + W_DEV * (dev - 1) - W_LIAB * liab - W_HALL * (4 - r["_hall"])
                 - W_IDENT * max(0.0, r["_ident"] - IDENT_SOFT))
        note = (f"iptm {r['_iptm']}, min_ipae {r['_ipae']}, dev {dev:.2f}, liab {liab}, "
                f"hallmarks {r['_hall']}/4, ident {r['_ident']:.2f}")
        if r["_cif"] and Path(r["_cif"]).exists():
            rec, hits = epitope(r["_cif"])
            score += W_EPITOPE * rec + W_HOTSPOT * hits
            note += f", epitope recall {rec:.2f}, hotspots {hits}/4"
            if rec < EPITOPE_MIN_RECALL:
                score -= EPITOPE_PENALTY
                note += " [off-epitope penalty]"
        else:
            note += ", no structure (epitope unchecked)"
        r["final_score"], r["_note"] = round(score, 3), note

    picked = []
    for r in sorted(live, key=lambda r: (-r["final_score"], r["id"])):
        near = next((q for q in picked if nf.hamming(r["_p"]["c3"], q["_p"]["c3"]) < MIN_DIST), None)
        if near:
            r["reason"] = f"rejected: CDR3 near-duplicate of {near['id']}"
        elif len(picked) >= top_n:
            r["reason"] = f"rejected: below top-{top_n} cutoff; {r['_note']}"
        else:
            picked.append(r)
            r["selected"] = True
            r["reason"] = f"pick #{len(picked)}: {r['_note']}"
    rows.sort(key=lambda r: (not r["selected"], -(r["final_score"] if r["final_score"] is not None else -1e9), r["id"]))
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def table(rows):
    for r in rows:
        sc = "-" if r["final_score"] is None else f"{r['final_score']:.3f}"
        print(f"{'Y' if r['selected'] else 'n'} {sc:>7} {r['id']:16} {r['reason']}")


def demo():
    mk = lambda fr1, c1, fr2, c2, fr3, c3, fr4="WGQGTQVTVS": fr1 + c1 + fr2 + c2 + fr3 + c3 + fr4
    fr1, fr2, fr3 = FR[0], FR[1], FR[2]
    d1, d3 = "QVKLQESGGGSVQAGGSLRLTC", "YADSVKGRFTISRDNAKNTVYLQMNNLKPEDTAIYYC"
    d1b, d3b = "QVQLVESGGGSVQAGGSLKLSC", "YTDSVKGRFTISRENAKNTLYLQMNNLKPEDTAMYYC"
    cif = str(HERE / "data/structures/her2_cdrh3_0_generated.cif")
    known10 = sorted(nf.load_provided_cdrh3(FACS, SPR)[10])[0]
    cands = [
        {"id": "good1", "vhh_sequence": mk(d1, "GRTFSSYGMG", fr2, "AISWSGGST", d3, "AADLRGYSSSWYDY"), "iptm": 0.8, "min_ipae": 8, "design_cif": cif},
        {"id": "good2", "vhh_sequence": mk(d1b, "GFTLDYYAIG", fr2, "SISSSDGST", d3b, "AVRPYGSGWYEYDY"), "iptm": 0.7, "min_ipae": 12},
        {"id": "fixed_fw_short", "vhh_sequence": mk(fr1, "GRTFSSYGMG", fr2, "AISWSGGST", fr3, "AADLRGYSSSWY"), "iptm": 0.9, "min_ipae": 5},
        {"id": "odd_cys", "vhh_sequence": mk(d1, "GRTFSSYGMG", fr2, "AISWSGGST", d3, "AADLCGYSSSWYDY"), "iptm": 0.9, "min_ipae": 5},
        {"id": "vh_like", "vhh_sequence": mk(d1, "GRTFSSYGMG", "WVRQAPGKGLEWVS", "AISWSGGST", d3, "AADLRGYSSSWYDY"), "iptm": 0.9, "min_ipae": 5},
        {"id": "known_cdr3", "vhh_sequence": mk(d1, "GRTFSSYGMG", fr2, "AISWSGGST", d3, known10), "iptm": 0.9, "min_ipae": 5},
        {"id": "low_iptm", "vhh_sequence": mk(d1, "GRTFSSYGMG", fr2, "AISWSGGST", d3, "AAKLRGYSSSWYDY"), "iptm": 0.4, "min_ipae": 5},
    ]
    rows = rank(cands)
    table(rows)
    by = {r["id"]: r for r in rows}
    assert by["good1"]["selected"] and "epitope recall" in by["good1"]["reason"]
    assert by["good2"]["selected"]
    for k, w in [("fixed_fw_short", "identity"), ("odd_cys", "cysteine"), ("vh_like", "hallmarks"),
                 ("known_cdr3", "CDR3"), ("low_iptm", "ipTM")]:
        assert not by[k]["selected"] and w in by[k]["reason"], (k, by[k]["reason"])
    assert identity("ACDEF", "ACDEF") == 1.0 and identity("ACDEF", "ACXEF") == 0.8
    assert rows == rank(cands), "must be deterministic"
    print("self-check passed")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        out = rank(load(sys.argv[1]))
        table(out)
        if len(sys.argv) > 2:
            with open(sys.argv[2], "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(out[0]))
                w.writeheader(); w.writerows(out)
    else:
        demo()
