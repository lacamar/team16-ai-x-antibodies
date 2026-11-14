"""Seed the design shell on the MEASURED champion, not on trastuzumab.

The find
--------
Of the 182 unique CDRH3s with a quantified SPR KD, exactly 6 are tighter than
trastuzumab's median (0.135 nM, 8 replicates). The best is

    WGSQGFYAYD   0.0234 nM   = G101S, D102Q, M107Y   (5.8x tighter than tz)

Recurring substitutions across those 6 winners: G101S (x2), D102Q (x2), A106L (x2).
Our ridge, fitted on these same 180 rows, independently ranked D102Q its single
best substitution (+0.583 log10 units) and position 101 among its top three --
so the measured champion and the model agree on 2 of 3 positions. That is a
consistency check, NOT independent validation: these rows are in the ridge's
training set. What IS independent is the measurement itself.

Why this changes the generator
------------------------------
Every search so far (shell_search.py, joint_search.py) expanded around
trastuzumab, a molecule measured at 0.135 nM. WGSQGFYAYD is measured at
0.0234 nM. Expanding around it starts 0.76 log units ahead, from a point whose
affinity is known rather than predicted.

It also fixes the novelty problem that F5/F9 ran into. WGSQGFYAYD is already 3
mutations from trastuzumab but sits IN the provided library (min_dist 0), so it
cannot be submitted. Adding 1-3 further mutations at the OTHER seven positions
keeps all three measured-winning substitutions, reaches 4-6 mutations from
trastuzumab (clearing the <70%-identity rule under either reading of Q1), and
can clear Hamming >=3 from every one of the 525,063 provided sequences.

So: hold 101S/102Q/107Y fixed, enumerate every combination of 1-3 mutations
across the remaining seven positions, gate with verify_gates.check() unchanged,
and score with all four models. Candidates are then required to beat
TRASTUZUMAB on the nonlinear SPR RF -- the gate that F9 showed the
AntiFold-intersection pool could not clear even once.

Writes runs/champion_candidates.csv.
"""
import itertools
import json
import os
import pickle

import numpy as np
import pandas as pd

import antifold_generate as AF
import verify_gates as VG

HERE = os.path.dirname(os.path.abspath(__file__))
AA = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(AA)}
L = 10
TZ = "WGGDGFYAMD"
CHAMP = "WGSQGFYAYD"
CHAMP_KD = 0.0234
TZ_KD = 0.135
RES = list(range(99, 109))
# positions carrying the measured champion's winning substitutions -- held fixed
FIXED = [p for p in range(L) if CHAMP[p] != TZ[p]]
FREE = [p for p in range(L) if p not in FIXED]
MAX_PER_POSSET = 3


def main():
    print(f"trastuzumab  {TZ}  measured {TZ_KD:.4f} nM")
    print(f"champion     {CHAMP}  measured {CHAMP_KD:.4f} nM  "
          f"({TZ_KD/CHAMP_KD:.1f}x tighter)")
    print("held fixed (measured winning substitutions): "
          + ", ".join(f"{TZ[p]}{RES[p]}{CHAMP[p]}" for p in FIXED))
    print("free positions: " + ", ".join(f"{RES[p]}{TZ[p]}" for p in FREE))

    # ---- scorers -------------------------------------------------------
    M = pickle.load(open(f"{HERE}/models/kd_reg.pkl", "rb"))
    ridge, rf = M["ridge"], M["rf"]
    clf = pickle.load(open(f"{HERE}/models/binding_clf.pkl", "rb"))["model"]
    Waf, _ = AF.antifold_matrix()

    def onehot(seqs):
        X = np.zeros((len(seqs), L * 20), dtype=np.float32)
        for n, s in enumerate(seqs):
            for p, c in enumerate(s[:L]):
                X[n, p * 20 + IDX[c]] = 1.0
        return X

    def af_score(s):
        return float(sum(Waf[p, IDX[c]] for p, c in enumerate(s)))

    # ---- enumerate 1-3 extra mutations at the free positions ------------
    seqs = []
    for k in (1, 2, 3):
        for pos in itertools.combinations(FREE, k):
            for subs in itertools.product(AA, repeat=k):
                if any(CHAMP[p] == c for p, c in zip(pos, subs)):
                    continue
                t = list(CHAMP)
                for p, c in zip(pos, subs):
                    t[p] = c
                seqs.append("".join(t))
    seqs = sorted(set(seqs))
    print(f"\nenumerated {len(seqs):,} sequences within 3 mutations of the champion "
          f"(champion's 3 winning substitutions retained in all of them)")

    df = pd.DataFrame({"cdrh3": seqs})
    df["n_mut_vs_tz"] = [sum(a != b for a, b in zip(s, TZ)) for s in seqs]
    df["n_mut_vs_champ"] = [sum(a != b for a, b in zip(s, CHAMP)) for s in seqs]
    # the <70%-identity-to-trastuzumab rule needs >=4 under the strict reading
    df = df[df.n_mut_vs_tz >= 4]
    print(f"with >=4 mutations from trastuzumab (strict identity rule): {len(df):,}")

    X = onehot(df.cdrh3.tolist() + [TZ, CHAMP])
    df["ridge_score"] = ridge.predict(X)[:-2]
    df["kd_rf"] = rf.predict(X)[:-2]
    df["facs_histgb"] = clf.predict_proba(X)[:-2, 1]
    tz_ridge, ch_ridge = ridge.predict(X)[-2:]
    tz_rf, ch_rf = rf.predict(X)[-2:]
    tz_facs, ch_facs = clf.predict_proba(X)[-2:, 1]
    print(f"\nbaselines:  trastuzumab ridge={tz_ridge:+.3f} kd_rf={tz_rf:+.3f} "
          f"facs={tz_facs:.4f}")
    print(f"            champion    ridge={ch_ridge:+.3f} kd_rf={ch_rf:+.3f} "
          f"facs={ch_facs:.4f}")

    # hard requirement: beat TRASTUZUMAB on the nonlinear SPR model, and clear
    # the FACS floor (F3: trastuzumab saturates the classifier, so it is a
    # floor not a ceiling)
    df["beats_tz_rf"] = df.kd_rf > tz_rf
    df["beats_champ_rf"] = df.kd_rf > ch_rf
    df["gate_facs_floor"] = df.facs_histgb > 0.8
    print(f"\nbeat trastuzumab on nonlinear SPR RF: {int(df.beats_tz_rf.sum()):,}"
          f"   | also clear FACS floor: "
          f"{int((df.beats_tz_rf & df.gate_facs_floor).sum()):,}"
          f"   | also beat the CHAMPION: "
          f"{int((df.beats_champ_rf & df.gate_facs_floor).sum()):,}")

    pool = df[df.beats_tz_rf & df.gate_facs_floor].copy()
    if pool.empty:
        print("EMPTY -- report as finding.")
        return
    pool["af_score"] = [af_score(s) for s in pool.cdrh3]
    pool["n_beat_tz_of4"] = (
        (pool.ridge_score > tz_ridge).astype(int)
        + (pool.kd_rf > tz_rf).astype(int)
        + (pool.facs_histgb > tz_facs).astype(int)
        + (pool.af_score > af_score(TZ)).astype(int))
    pool["mut_positions"] = ["+".join(str(RES[p]) for p in range(L) if s[p] != TZ[p])
                             for s in pool.cdrh3]
    pool["extra_muts_vs_champ"] = [
        ",".join(f"{CHAMP[p]}{RES[p]}{s[p]}" for p in range(L) if s[p] != CHAMP[p])
        for s in pool.cdrh3]
    pool = pool.sort_values(["kd_rf", "n_beat_tz_of4"], ascending=False)

    # ---- novelty gate, shipped code, in rank order ----------------------
    prov = VG.load_provided()
    keep = []
    for r in pool.itertuples():
        c = VG.check(r.cdrh3, prov)
        if c["pass"]:
            keep.append({**{k: getattr(r, k) for k in
                            ("cdrh3", "n_mut_vs_tz", "n_mut_vs_champ",
                             "extra_muts_vs_champ", "mut_positions",
                             "kd_rf", "ridge_score", "facs_histgb", "af_score",
                             "n_beat_tz_of4", "beats_champ_rf")},
                         "min_dist_provided": c["min_dist_provided"]})
            if len(keep) >= 400:
                break
    out = pd.DataFrame(keep)
    print(f"\nNOVELTY-COMPLIANT champion-seeded candidates: {len(out)}")
    if out.empty:
        print("EMPTY after novelty gate -- that is the finding.")
        return
    print(f"  of which also beat the measured champion on the SPR RF: "
          f"{int(out.beats_champ_rf.sum())}")

    sel, seen = [], {}
    for r in out.itertuples():
        n = seen.get(r.mut_positions, 0)
        if n >= MAX_PER_POSSET:
            continue
        seen[r.mut_positions] = n + 1
        sel.append(r.Index)
    div = out.loc[sel]
    div.to_csv(f"{HERE}/runs/champion_candidates.csv", index=False)
    json.dump({"seed": CHAMP, "seed_measured_kd_nM": CHAMP_KD,
               "trastuzumab_measured_kd_nM": TZ_KD,
               "fixed_substitutions": [f"{TZ[p]}{RES[p]}{CHAMP[p]}" for p in FIXED],
               "n_enumerated": len(seqs),
               "n_beat_tz_rf_and_facs": int(len(pool)),
               "n_novelty_compliant": int(len(out)),
               "n_beat_champion_rf": int(out.beats_champ_rf.sum()),
               "n_after_diversity_cap": int(len(div)),
               "baselines": {"tz_ridge": float(tz_ridge), "tz_kd_rf": float(tz_rf),
                             "tz_facs": float(tz_facs),
                             "champ_ridge": float(ch_ridge),
                             "champ_kd_rf": float(ch_rf),
                             "champ_facs": float(ch_facs)}},
              open(f"{HERE}/runs/champion_shell.json", "w"), indent=1)
    print(f"after diversity cap (<={MAX_PER_POSSET} per position-set): {len(div)}"
          f"  over {div.mut_positions.nunique()} distinct position-sets")
    print("\n=== top 25 champion-seeded, novelty-compliant candidates ===")
    cols = ["cdrh3", "extra_muts_vs_champ", "n_mut_vs_tz", "min_dist_provided",
            "kd_rf", "facs_histgb", "ridge_score", "n_beat_tz_of4"]
    print(div.head(25)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
