"""Second, INDEPENDENT generator for novelty-compliant CDRH3s (NOTES.md RISK 1).

Why this exists
---------------
All 20 designs in results/submission_top20.csv came from one generator: the
additive-ridge shell enumeration, whose per-position weights were fitted on
180 quantified SPR KDs. 19 of 20 carry the identical A106E substitution and
there are only 4 distinct position-sets in the whole list. If the ridge's
preference is an artifact of that small fit, all 20 fail together. Ranking
inside a correlated pool cannot detect that the pool is correlated.

AntiFold (lane B, op-441067223204e140) is a genuinely different inductive
bias: antibody inverse folding, conditioned on the Fv BACKBONE, with no
antigen chain and no exposure to our SPR data. Its per-residue log-probs are
read directly at chain B residues 99-108 (= our 10 designable CDRH3
positions, verified: the baseline row sequence there is WGGDGFYAMD).

NOTE on AntiFold's own region labels: run with skip_anarcii_numbering=true
(the only way it would accept our Boltz-2 chains at all), so `assumed_region`
is wrong -- it labels chain B CDRH3 as "FWL3"/"CDRL3". We never use that
column; positions come from pdb_posins, which is correct.

Method: AntiFold one-shot scoring is additive over positions, exactly like
the ridge, so score(s) = sum_p logp[p, s_p] can be enumerated EXACTLY over
the k=4 and k=5 mutational shells -- no sampling, same machinery as
shell_search.py, different weight matrix. Candidates are then gated by
verify_gates.check() UNCHANGED, and cross-scored with the three ridge-side
models so D5's "independent models must agree" rule can be applied.
"""
import itertools
import json
import os
import pickle

import numpy as np
import pandas as pd

import verify_gates as VG

HERE = os.path.dirname(os.path.abspath(__file__))
AA = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(AA)}
L = 10
TZ = "WGGDGFYAMD"
CDRH3_RES = list(range(99, 109))          # chain B residue numbers
AF_CSV = (f"{HERE}/nebius-runs/antifold-skipnum-out/named/"
          "BASELINE_trastuzumab_AB.csv")
N_ALT = 6        # alternative residues considered per position
N_KEEP = 400     # novelty-passing candidates to retain


def antifold_matrix():
    """(10,20) log-prob matrix at chain B res 99-108, from the trastuzumab
    backbone fold. Asserts the native residues are what we think they are."""
    df = pd.read_csv(AF_CSV)
    df = df[(df.pdb_chain == "B") & (df.pdb_posins.isin(CDRH3_RES))]
    df = df.sort_values("pdb_posins")
    native = "".join(df.pdb_res.tolist())
    assert native == TZ, f"CDRH3 at B99-108 is {native!r}, expected {TZ!r}"
    W = df[list(AA)].to_numpy(dtype=np.float64)
    assert W.shape == (L, 20)
    # rows are log-softmax; confirm before using them as log-probs
    tot = np.exp(W).sum(axis=1)
    assert np.allclose(tot, 1.0, atol=2e-2), f"rows not normalised: {tot}"
    return W, df.perplexity.to_numpy()


def main():
    W, perp = antifold_matrix()
    af = lambda s: float(sum(W[p, IDX[c]] for p, c in enumerate(s)))
    tz_af = af(TZ)
    print(f"AntiFold CDRH3 log-likelihood: trastuzumab = {tz_af:.3f}")
    print("\nper-position AntiFold view (delta vs native, nats; + = preferred):")
    delta = W - W[np.arange(L), [IDX[c] for c in TZ]][:, None]
    for p in range(L):
        order = np.argsort(-delta[p])
        top = [(AA[i], round(float(delta[p, i]), 2))
               for i in order if AA[i] != TZ[p]][:3]
        print(f"  res{CDRH3_RES[p]} {TZ[p]} (perplexity {perp[p]:5.2f}): {top}")

    cand = {p: [AA[i] for i in np.argsort(-delta[p]) if AA[i] != TZ[p]][:N_ALT]
            for p in range(L)}

    rows = []
    for k in (4, 5):
        n = 0
        for pos in itertools.combinations(range(L), k):
            for subs in itertools.product(*[cand[p] for p in pos]):
                t = list(TZ)
                for p, c in zip(pos, subs):
                    t[p] = c
                rows.append(("".join(t), k))
                n += 1
        print(f"shell k={k}: enumerated {n:,}")

    df = pd.DataFrame(rows, columns=["cdrh3", "n_mut"]).drop_duplicates("cdrh3")
    df["af_score"] = [af(s) for s in df.cdrh3]
    df = df.sort_values("af_score", ascending=False).reset_index(drop=True)
    print(f"total unique: {len(df):,}; best AntiFold score {df.af_score.max():.3f} "
          f"vs trastuzumab {tz_af:.3f}")

    # --- gate, in descending AntiFold order, with the SHIPPED gate code ---
    prov = VG.load_provided()
    print(f"provided CDRH3 reference set: {len(prov)}")
    keep = []
    for r in df.itertuples():
        c = VG.check(r.cdrh3, prov)
        if c["pass"]:
            keep.append({"cdrh3": r.cdrh3, "af_score": r.af_score,
                         "n_mut": r.n_mut,
                         "min_dist_provided": c["min_dist_provided"]})
            if len(keep) >= N_KEEP:
                break
    out = pd.DataFrame(keep)
    print(f"novelty-passing AntiFold candidates kept: {len(out)}")

    # --- cross-score with the ridge-side models (D5: agreement, not argmax) ---
    def onehot(seqs):
        X = np.zeros((len(seqs), L * 20), dtype=np.float32)
        for n, s in enumerate(seqs):
            for p, c in enumerate(s[:L]):
                X[n, p * 20 + IDX[c]] = 1.0
        return X

    Xo = onehot(out.cdrh3.tolist() + [TZ])
    clf = pickle.load(open(f"{HERE}/models/binding_clf.pkl", "rb"))["model"]
    facs = clf.predict_proba(Xo)[:, 1]
    M = pickle.load(open(f"{HERE}/models/kd_reg.pkl", "rb"))
    ridge, rf = M["ridge"], M["rf"]
    rs, rfs = ridge.predict(Xo), rf.predict(Xo)
    out["facs_histgb"], tz_facs = facs[:-1], facs[-1]
    out["ridge_score"], tz_ridge = rs[:-1], rs[-1]
    out["kd_rf"], tz_rf = rfs[:-1], rfs[-1]
    out["af_beats_tz"] = out.af_score > tz_af
    out["mut_positions"] = [
        "+".join(str(CDRH3_RES[p]) for p in range(L) if s[p] != TZ[p])
        for s in out.cdrh3]

    base = {"af_score": tz_af, "facs_histgb": float(tz_facs),
            "ridge_score": float(tz_ridge), "kd_rf": float(tz_rf)}
    print("\ntrastuzumab baseline through the same scorers:",
          {k: round(v, 4) for k, v in base.items()})
    n_beat = sum((out[k] > base[k]).astype(int) for k in
                 ("af_score", "facs_histgb", "ridge_score", "kd_rf"))
    out["n_beat_tz_of4"] = n_beat
    out = out.sort_values(["n_beat_tz_of4", "af_score"], ascending=False)
    out.to_csv(f"{HERE}/runs/antifold_candidates.csv", index=False)
    json.dump({"baseline": base, "n_kept": len(out),
               "source_op": "op-441067223204e140",
               "n_alt_per_position": N_ALT,
               "note": "AntiFold Fv-backbone inverse folding, no antigen "
                       "chain, no exposure to SPR/FACS data"},
              open(f"{HERE}/runs/antifold_gen.json", "w"), indent=1)

    print("\n=== position-sets AntiFold proposes (top 60 by AntiFold score) ===")
    print(out.head(60).mut_positions.value_counts().head(12).to_string())
    print("\n=== top 20 novelty-compliant AntiFold candidates ===")
    cols = ["cdrh3", "mut_positions", "af_score", "kd_rf", "facs_histgb",
            "ridge_score", "n_beat_tz_of4"]
    print(out.head(20)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
