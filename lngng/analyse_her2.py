"""Stdlib-only analysis of the HER2 hackathon Group-1 tables."""
import csv, json, math, collections, statistics

D = "/home/hackathon/data/her2-trastuzumab"
TZ = "WGGDGFYAMD"
out = {}

def ham(a, b):
    return sum(x != y for x, y in zip(a, b))

# ---------- FACS ----------
facs = []
with open(f"{D}/tz_her2_facs_3classes_524346.csv") as fh:
    for r in csv.DictReader(fh):
        facs.append((r["cdrh3"], r["binding_class"], int(r["edit_distance"])))

out["facs_n"] = len(facs)
out["facs_classes"] = dict(collections.Counter(c for _, c, _ in facs))
out["facs_unique_cdrh3"] = len({s for s, _, _ in facs})
out["facs_len_check"] = dict(collections.Counter(len(s) for s, _, _ in facs))

# edit_distance column vs recomputed Hamming to trastuzumab (sanity)
mismatch = sum(1 for s, _, e in facs if ham(s, TZ) != e)
out["facs_editdist_mismatch_vs_hamming"] = mismatch

# P(high) by edit distance
by_ed = collections.defaultdict(collections.Counter)
for s, c, e in facs:
    by_ed[e][c] += 1
rows = []
for e in sorted(by_ed):
    cc = by_ed[e]
    n = sum(cc.values())
    rows.append({"edit_distance": e, "n": n,
                 "high": cc["high"], "medium": cc["medium"], "low": cc["low"],
                 "pct_high": round(100 * cc["high"] / n, 1),
                 "pct_low": round(100 * cc["low"] / n, 1)})
out["facs_by_edit_distance"] = rows

# eligibility arithmetic: provided seqs that are >=4 from trastuzumab
ge4 = sum(1 for s, _, e in facs if e >= 4)
high_ge4 = sum(1 for s, c, e in facs if e >= 4 and c == "high")
out["facs_ge4_from_tz"] = ge4
out["facs_high_ge4_from_tz"] = high_ge4
out["facs_high_total"] = out["facs_classes"]["high"]

# position-wise: AA frequency in high vs low, as enrichment
hi = [s for s, c, _ in facs if c == "high"]
lo = [s for s, c, _ in facs if c == "low"]
pos_rows = []
for i in range(10):
    ch = collections.Counter(s[i] for s in hi)
    cl = collections.Counter(s[i] for s in lo)
    nh, nl = sum(ch.values()), sum(cl.values())
    # top enriched residue in high binders at this position
    best = max(ch, key=lambda a: (ch[a] / nh) / max(cl[a] / nl, 1e-9))
    pos_rows.append({
        "pos": i + 1, "tz_residue": TZ[i],
        "pct_conserved_tz_high": round(100 * ch[TZ[i]] / nh, 1),
        "pct_conserved_tz_low": round(100 * cl[TZ[i]] / nl, 1),
        "most_enriched_in_high": best,
        "enrichment": round((ch[best] / nh) / max(cl[best] / nl, 1e-9), 2),
    })
out["facs_positions"] = pos_rows

# ---------- SPR ----------
spr = list(csv.DictReader(open(f"{D}/trastuzumab_her2_spr_all_768.csv")))
out["spr_n"] = len(spr)
out["spr_results"] = dict(collections.Counter(r["binding_result"] for r in spr))
foreign = [r for r in spr if r["cdrh3"] == "ARDRIVDDYYYYGMDV"]
out["spr_foreign_rows"] = len(foreign)
tz_rows = [r for r in spr if r["cdrh3"] == TZ]
tz_kd = sorted(float(r["kd_nM"]) for r in tz_rows if r["kd_nM"])
out["spr_trastuzumab_rows"] = len(tz_rows)
out["spr_trastuzumab_kd"] = {"n": len(tz_kd), "min": tz_kd[0], "max": tz_kd[-1],
                             "median": statistics.median(tz_kd)}

kd = [(r["cdrh3"], float(r["kd_nM"])) for r in spr
      if r["kd_nM"] and r["cdrh3"] != "ARDRIVDDYYYYGMDV"]
vals = sorted(v for _, v in kd)
out["spr_kd_n"] = len(kd)
out["spr_kd_stats"] = {"min": vals[0], "max": vals[-1],
                       "median": statistics.median(vals),
                       "geomean_nM": round(math.exp(statistics.fmean(math.log(v) for v in vals)), 2)}

# how many beat trastuzumab's best / worst reference
best_tz, worst_tz = tz_kd[0], tz_kd[-1]
out["spr_beat_tz_best"] = sum(1 for _, v in kd if v < best_tz)
out["spr_beat_tz_worst"] = sum(1 for _, v in kd if v < worst_tz)
out["spr_tz_reference_window"] = [best_tz, worst_tz]

# KD vs edit distance (10-mers only, vs trastuzumab)
kd10 = [(s, v, ham(s, TZ)) for s, v in kd if len(s) == 10]
grp = collections.defaultdict(list)
for s, v, e in kd10:
    grp[e].append(v)
out["spr_kd_by_edit_distance"] = [
    {"edit_distance": e, "n": len(grp[e]),
     "median_kd_nM": round(statistics.median(grp[e]), 2),
     "best_kd_nM": round(min(grp[e]), 3)}
    for e in sorted(grp)]

# eligibility of the measured binders themselves
facs_set = {s for s, _, _ in facs}
elig = []
for s, v, e in kd10:
    in_provided = s in facs_set
    elig.append({"ge4_from_tz": e >= 4, "in_facs": in_provided})
out["spr_kd10_n"] = len(kd10)
out["spr_kd10_ge4_from_tz"] = sum(1 for x in elig if x["ge4_from_tz"])
out["spr_kd10_in_facs"] = sum(1 for x in elig if x["in_facs"])
# best KD among those >=4 from tz
ge4_kd = [v for _, v, e in kd10 if e >= 4]
out["spr_best_kd_ge4"] = round(min(ge4_kd), 3) if ge4_kd else None
lt4_kd = [v for _, v, e in kd10 if e < 4]
out["spr_best_kd_lt4"] = round(min(lt4_kd), 3) if lt4_kd else None
out["spr_median_kd_ge4"] = round(statistics.median(ge4_kd), 2) if ge4_kd else None
out["spr_median_kd_lt4"] = round(statistics.median(lt4_kd), 2) if lt4_kd else None

# overlap between SPR and FACS
out["spr_cdrh3_in_facs"] = len({r["cdrh3"] for r in spr} & facs_set)

# ---------- design-space arithmetic ----------
space = 20 ** 10
ball2 = 1 + 10 * 19 + (10 * 9 // 2) * 19 ** 2   # Hamming ball radius 2
out["space_total_10mers"] = space
out["space_provided_fraction"] = out["facs_unique_cdrh3"] / space
out["space_excluded_upper_bound"] = out["facs_unique_cdrh3"] * ball2
out["space_excluded_fraction_upper"] = out["facs_unique_cdrh3"] * ball2 / space

print(json.dumps(out, indent=2, default=str))
