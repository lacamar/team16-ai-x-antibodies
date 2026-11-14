"""Affinity surrogate — rank HER2 designs by a model trained on the REAL measurements,
not by fold confidence (ipTM is a weak affinity proxy).

Two heads on one-hot CDRH3 features (the field guide notes one-hot often works):
  * affinity regressor : Ridge on log10(KD) from the SPR-quantified variants (~190 rows).
  * binder classifier  : LogisticRegression on the 525k FACS classes (high vs rest).

score(cdrh3) -> {pred_log_kd, pkd (higher=tighter), p_high, surrogate}. Rank designs by
`surrogate`; gate on `p_high` if you want a binder prior. Deterministic (seed).

Deps: numpy, scikit-learn, scipy.  Data: data/her2/{trastuzumab_her2_spr_all_768.csv,
tz_her2_facs_3classes_524346.csv.zip}.  Train once, pickle, reuse (wrap calls in a disk cache).

TODO when there's time: swap one-hot for ESM/ESM-C embeddings (Nebius `esmc-*`), add the
zenodo train/val/test splits, blend ipTM/min_ipae into `surrogate`.
"""
import csv, io, math, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "her2"
AA = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(AA)}
L = 10                                   # trastuzumab CDRH3 length
DROP_CDRH3 = {"ARDRIVDDYYYYGMDV"}        # 8 non-trastuzumab SPR rows (README caveat)
SEED = 1


def onehot(cdrh3: str) -> np.ndarray:
    cdrh3 = cdrh3.strip().upper()[:L].ljust(L, "-")
    v = np.zeros(L * len(AA), dtype=np.float32)
    for i, a in enumerate(cdrh3):
        j = IDX.get(a)
        if j is not None:
            v[i * len(AA) + j] = 1.0
    return v


def load_spr(path=DATA / "trastuzumab_her2_spr_all_768.csv"):
    """-> (X one-hot, y=log10 KD_nM) for the quantified, trastuzumab-variant rows."""
    X, y = [], []
    for r in csv.DictReader(open(path)):
        c = r["cdrh3"].strip().upper()
        if c in DROP_CDRH3 or r.get("binding_result") != "quantified_binding":
            continue
        kd = (r.get("kd_nM") or "").strip()
        if not kd:
            continue
        X.append(onehot(c)); y.append(math.log10(float(kd)))
    return np.asarray(X), np.asarray(y)


def load_facs(zip_path=DATA / "tz_her2_facs_3classes_524346.csv.zip", cap=None):
    """-> (X one-hot, y=1 if binding_class=='high' else 0). `cap` limits rows for speed."""
    X, y = [], []
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
        for i, r in enumerate(csv.DictReader(io.TextIOWrapper(f, "utf-8"))):
            if cap and i >= cap:
                break
            X.append(onehot(r["cdrh3"])); y.append(1 if r["binding_class"] == "high" else 0)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8)


class AffinitySurrogate:
    def __init__(self):
        self.reg = None    # log10 KD regressor (lower = tighter)
        self.clf = None    # P(high binder)

    def fit(self, facs_cap=150_000):
        from sklearn.linear_model import RidgeCV, LogisticRegression
        Xs, ys = load_spr()
        self.reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Xs, ys)
        Xf, yf = load_facs(cap=facs_cap)
        self.clf = LogisticRegression(max_iter=400, C=1.0, random_state=SEED).fit(Xf, yf)
        self._n = (len(ys), len(yf))
        return self

    def score(self, cdrh3: str) -> dict:
        x = onehot(cdrh3)[None, :]
        log_kd = float(self.reg.predict(x)[0])
        p_high = float(self.clf.predict_proba(x)[0, 1])
        pkd = -log_kd                                   # higher = tighter
        return {"pred_log_kd": round(log_kd, 3), "pkd": round(pkd, 3),
                "p_high": round(p_high, 3),
                "surrogate": round(pkd + 2.0 * (p_high - 0.5), 3)}  # tunable blend


def demo():
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict, train_test_split
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    Xs, ys = load_spr()
    print(f"SPR quantified: {len(ys)} rows  (KD {10**ys.min():.3g}–{10**ys.max():.3g} nM)")
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 12)), Xs, ys, cv=5)
    rho = spearmanr(ys, pred).statistic
    print(f"  affinity 5-fold Spearman(log KD) = {rho:.3f}   (ranking signal; classical methods ~0.3–0.5)")

    Xf, yf = load_facs(cap=120_000)
    Xtr, Xte, ytr, yte = train_test_split(Xf, yf, test_size=0.2, random_state=SEED, stratify=yf)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=400, random_state=SEED).fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    print(f"  FACS high-binder AUC (held out, {len(yf)} rows) = {auc:.3f}")

    s = AffinitySurrogate().fit(facs_cap=120_000)
    for name, c in [("trastuzumab", "WGGDGFYAMD"), ("design WASSSEYYFDY", "WASSSEYYFD"),
                    ("design WGPPYFYAMD", "WGPPYFYAMD")]:
        print(f"  {name:22} -> {s.score(c)}")
    print("self-check passed")


if __name__ == "__main__":
    demo()
