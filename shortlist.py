"""IgLM top-K -> Boltz-2 fold -> rank.py. Steps (see docs/IGLM_LOOP.md):
  prep    topk.csv outdir          gate + write outdir/queries.json (Boltz-2 input; VL/HER2 from data/boltz_query_base.json)
  metrics boltz2_dir out.csv       Boltz-2 scores.json -> refold_metrics.csv (feeds make_candidates.py)
topk.csv columns: cdrh3, vh_sequence, [reward]
"""
import csv, json, sys
from pathlib import Path

import novelty_filter as nf
from affinity_surrogate import DATA

HERE = Path(__file__).resolve().parent
BASE = HERE / "data/boltz_query_base.json"
VH_CHAIN = "B"


def prep(topk, out):
    by_len = nf.load_provided_cdrh3(DATA / "tz_her2_facs_3classes_524346.csv.zip",
                                    DATA / "trastuzumab_her2_spr_all_768.csv")
    base = json.load(open(BASE))
    queries, seen = [], set()
    for r in csv.DictReader(open(topk)):
        c, vh = r["cdrh3"].strip().upper(), r["vh_sequence"].strip().upper()
        if c in seen or not nf.is_novel(c, by_len)[0] or not nf.hard_checks(vh)[0]:
            continue
        seen.add(c)
        q = json.loads(json.dumps(base))
        q["query_id"] = f"iglm_{len(queries):03d}"
        next(ch for ch in q["chains"] if ch["chain_id"] == VH_CHAIN)["sequence"] = vh
        queries.append(q)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"queries": queries}, open(out / "queries.json", "w"))
    print(f"{len(queries)} queries -> {out/'queries.json'}")


def metrics(boltz_dir, out_csv, queries_json):
    qs = {q["query_id"]: q for q in json.load(open(queries_json))["queries"]}
    rows = []
    for qid, q in sorted(qs.items()):
        d = Path(boltz_dir) / qid / "seed_42"
        sc, cif = d / f"{qid}_seed_42_sample_1_scores.json", d / f"{qid}_seed_42_sample_1_model.cif"
        if not (sc.exists() and cif.exists()):
            continue
        vh = next(ch for ch in q["chains"] if ch["chain_id"] == VH_CHAIN)["sequence"]
        rows.append({"design": str(cif.resolve()), "status": "predicted", "binder_sequence": vh,
                     "iptm": json.load(open(sc))["interface_predicted_tm_score"], "min_ipae": ""})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, ["design", "status", "binder_sequence", "iptm", "min_ipae"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} folded -> {out_csv}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "prep":
        prep(a[1], a[2])
    elif len(a) == 4 and a[0] == "metrics":
        metrics(a[1], a[2], a[3])
    else:
        sys.exit(__doc__)
