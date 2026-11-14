"""Exact search of the INTERSECTION of two independent generators (RISK 1).

Context
-------
F2 established the rule: single-model argmax is never evidence. antifold_generate.py
then demonstrated the rule cuts both ways -- AntiFold's own argmax wants positions
99/102/103/104/107 and lands on sequences the SPR-trained models call ~2 log units
WORSE than trastuzumab, just as the ridge's argmax landed on sequences AntiFold and
the FACS model dislike. Two generators, two disjoint optima, no overlap in the top 20
of either.

So the shipping criterion cannot be either generator's ranking. It is the
intersection: novelty-compliant 10-mers that BOTH the SPR ridge (fitted on 180
measured KDs) and AntiFold (Fv-backbone inverse folding, never shown our data)
score above trastuzumab. Both models are additive over positions, so this
intersection can be enumerated EXACTLY rather than sampled:

    af(s)    = tz_af    + sum_p  Daf[p, s_p]
    ridge(s) = tz_ridge + sum_p  Dr[p, s_p]

and the requirement is simply that both delta-sums are positive. Survivors are
then scored by the two NONLINEAR models (HistGB on 367k FACS rows, RF on the
SPR KDs) which share no fit with either generator, gated by verify_gates.check()
unchanged, and finally selected for POSITION-SET DIVERSITY so the shipped list
is not N replicates of one hypothesis.

Writes runs/joint_candidates.csv.
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
RES = list(range(99, 109))
N_ALT = 6          # top alternatives per position, PER MODEL, then unioned
MAX_PER_POSSET = 3  # diversity cap: at most this many shipped per position-set


def deltas():
    """Per-position substitution deltas for both additive models, plus the
    trastuzumab reference score of each."""
    Waf, perp = AF.antifold_matrix()
    Daf = Waf - Waf[np.arange(L), [IDX[c] for c in TZ]][:, None]
    tz_af = float(sum(Waf[p, IDX[c]] for p, c in enumerate(TZ)))

    M = pickle.load(open(f"{HERE}/models/kd_reg.pkl", "rb"))
    ridge = M["ridge"]
    Wr = ridge.coef_.reshape(L, 20)
    Dr = Wr - Wr[np.arange(L), [IDX[c] for c in TZ]][:, None]
    tz_ridge = float(ridge.intercept_ + sum(Wr[p, IDX[c]] for p, c in enumerate(TZ)))
    return Daf, Dr, tz_af, tz_ridge, perp


def main():
    Daf, Dr, tz_af, tz_ridge, perp = deltas()
    print(f"trastuzumab: af={tz_af:.3f}  ridge={tz_ridge:.3f}")

    # union of each model's preferred alternatives -> neither model's blind spot
    alts = {}
    for p in range(L):
        a = {AA[i] for i in np.argsort(-Daf[p]) if AA[i] != TZ[p]}
        b = {AA[i] for i in np.argsort(-Dr[p]) if AA[i] != TZ[p]}
        top_af = [AA[i] for i in np.argsort(-Daf[p]) if AA[i] != TZ[p]][:N_ALT]
        top_r = [AA[i] for i in np.argsort(-Dr[p]) if AA[i] != TZ[p]][:N_ALT]
        alts[p] = sorted(set(top_af) | set(top_r), key=lambda c: IDX[c])
    print("alternatives per position (union of both models' top "
          f"{N_ALT}):")
    for p in range(L):
        print(f"  res{RES[p]} {TZ[p]}: {''.join(alts[p])} ({len(alts[p])})")

    # exact enumeration of shells k=4,5, filtered in-loop on BOTH models
    found = []
    n_enum = 0
    for k in (4, 5):
        for pos in itertools.combinations(range(L), k):
            codes = [np.array([IDX[c] for c in alts[p]]) for p in pos]
            # additive sums by successive broadcasting
            saf = Daf[pos[0], codes[0]]
            sr = Dr[pos[0], codes[0]]
            for j in range(1, k):
                saf = np.add.outer(saf, Daf[pos[j], codes[j]]).ravel()
                sr = np.add.outer(sr, Dr[pos[j], codes[j]]).ravel()
            n_enum += saf.size
            hit = np.nonzero((saf > 0) & (sr > 0))[0]
            if not hit.size:
                continue
            sizes = [len(c) for c in codes]
            for h in hit:
                rem, picks = int(h), []
                for s in reversed(sizes):
                    picks.append(rem % s)
                    rem //= s
                picks.reverse()
                t = list(TZ)
                for p, pk in zip(pos, picks):
                    t[p] = alts[p][pk]
                found.append(("".join(t), k, float(saf[h]), float(sr[h])))
        print(f"shell k={k}: enumerated {n_enum:,} cumulative, "
              f"{len(found):,} pass both-models-beat-tz")

    df = pd.DataFrame(found, columns=["cdrh3", "n_mut", "d_af", "d_ridge"])
    df = df.drop_duplicates("cdrh3")
    df["af_score"] = tz_af + df.d_af
    df["ridge_score"] = tz_ridge + df.d_ridge
    print(f"\nunique sequences beating trastuzumab on BOTH additive models: {len(df):,}")
    if df.empty:
        print("INTERSECTION IS EMPTY -- report that as the finding.")
        return

    # nonlinear models: share no fit with either generator
    def onehot(seqs):
        X = np.zeros((len(seqs), L * 20), dtype=np.float32)
        for n, s in enumerate(seqs):
            for p, c in enumerate(s[:L]):
                X[n, p * 20 + IDX[c]] = 1.0
        return X

    Xo = onehot(df.cdrh3.tolist() + [TZ])
    clf = pickle.load(open(f"{HERE}/models/binding_clf.pkl", "rb"))["model"]
    facs = clf.predict_proba(Xo)[:, 1]
    rf = pickle.load(open(f"{HERE}/models/kd_reg.pkl", "rb"))["rf"]
    rfs = rf.predict(Xo)
    df["facs_histgb"], tz_facs = facs[:-1], float(facs[-1])
    df["kd_rf"], tz_rf = rfs[:-1], float(rfs[-1])
    print(f"baseline: facs={tz_facs:.4f} kd_rf={tz_rf:.4f}")

    # FACS is a FLOOR not a ceiling (F3): trastuzumab saturates the classifier
    df["gate_facs_floor"] = df.facs_histgb > 0.8
    df["beats_kd_rf"] = df.kd_rf > tz_rf
    df["n_beat_tz_of4"] = (2 + df.beats_kd_rf.astype(int)
                           + (df.facs_histgb > tz_facs).astype(int))
    df["mut_positions"] = ["+".join(str(RES[p]) for p in range(L) if s[p] != TZ[p])
                           for s in df.cdrh3]
    print("\nof those, pass FACS floor >0.8:", int(df.gate_facs_floor.sum()),
          "| also beat tz on nonlinear SPR RF:",
          int((df.gate_facs_floor & df.beats_kd_rf).sum()))

    # rank: 4-of-4 scorers first, then nonlinear SPR margin
    df = df.sort_values(["n_beat_tz_of4", "kd_rf"], ascending=False)
    pool = df[df.gate_facs_floor].copy()

    # novelty gate, shipped code, in rank order
    prov = VG.load_provided()
    keep = []
    for r in pool.itertuples():
        c = VG.check(r.cdrh3, prov)
        if c["pass"]:
            keep.append({**{k: getattr(r, k) for k in
                            ("cdrh3", "n_mut", "af_score", "ridge_score",
                             "facs_histgb", "kd_rf", "n_beat_tz_of4",
                             "mut_positions")},
                         "min_dist_provided": c["min_dist_provided"]})
            if len(keep) >= 300:
                break
    out = pd.DataFrame(keep)
    print(f"novelty-compliant joint candidates: {len(out)}")
    if out.empty:
        print("EMPTY after novelty gate -- that is itself the finding.")
        return

    # diversity selection: cap replicates of any one position-set
    sel, seen = [], {}
    for r in out.itertuples():
        n = seen.get(r.mut_positions, 0)
        if n >= MAX_PER_POSSET:
            continue
        seen[r.mut_positions] = n + 1
        sel.append(r.Index)
    div = out.loc[sel]
    div.to_csv(f"{HERE}/runs/joint_candidates.csv", index=False)
    json.dump({"baseline": {"af_score": tz_af, "ridge_score": tz_ridge,
                            "facs_histgb": tz_facs, "kd_rf": tz_rf},
               "n_enumerated": int(n_enum),
               "n_both_additive_beat_tz": int(len(df)),
               "n_novelty_compliant": int(len(out)),
               "n_after_diversity_cap": int(len(div)),
               "max_per_position_set": MAX_PER_POSSET,
               "antifold_op": "op-441067223204e140"},
              open(f"{HERE}/runs/joint_search.json", "w"), indent=1)

    print(f"\nafter diversity cap (<= {MAX_PER_POSSET} per position-set): {len(div)}")
    print("\n=== distinct position-sets available ===")
    print(out.mut_positions.value_counts().head(15).to_string())
    print("\n=== top 25 joint candidates (diversity-capped) ===")
    cols = ["cdrh3", "mut_positions", "n_mut", "n_beat_tz_of4", "kd_rf",
            "facs_histgb", "ridge_score", "af_score"]
    print(div.head(25)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
