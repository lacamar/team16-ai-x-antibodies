import numpy as np, pandas as pd, pickle
from scipy.stats import spearmanr
AA="ACDEFGHIKLMNPQRSTVWY"; IDX={a:i for i,a in enumerate(AA)}; L=10; TZ="WGGDGFYAMD"
def onehot(seqs):
    X=np.zeros((len(seqs),L*20),dtype=np.float32)
    for n,s in enumerate(seqs):
        for p,c in enumerate(s[:L]):
            if c in IDX: X[n,p*20+IDX[c]]=1.0
    return X
clf=pickle.load(open("models/binding_clf.pkl","rb"))["model"]      # HistGB, 367k FACS rows
kd=pickle.load(open("models/kd_reg.pkl","rb")); rf=kd["rf"]        # RF, nonlinear, SPR
nd=pd.read_csv("runs/shell_candidates.csv")
X=onehot(nd.cdrh3)
nd["facs_histgb"]=clf.predict_proba(X)[:,1]
nd["kd_rf"]=rf.predict(X)
tzX=onehot([TZ])
tz_facs=float(clf.predict_proba(tzX)[:,1][0]); tz_rf=float(rf.predict(tzX)[0])
print(f"TRASTUZUMAB: facs_histgb={tz_facs:.4f}  kd_rf={tz_rf:.3f}  ridge={-0.287:.3f}")
print(f"\nshell candidates (n={len(nd)}):")
print(f"  facs_histgb: mean={nd.facs_histgb.mean():.3f} max={nd.facs_histgb.max():.3f}  >= tz: {int((nd.facs_histgb>=tz_facs).sum())}")
print(f"  kd_rf      : mean={nd.kd_rf.mean():.3f} max={nd.kd_rf.max():.3f}  >= tz: {int((nd.kd_rf>=tz_rf).sum())}")
print(f"\nAGREEMENT between ridge(additive) and the two independent models, across the 400 shell candidates:")
print(f"  spearman(ridge, facs_histgb) = {spearmanr(nd.ridge_score,nd.facs_histgb).statistic:.3f}")
print(f"  spearman(ridge, kd_rf)       = {spearmanr(nd.ridge_score,nd.kd_rf).statistic:.3f}")
# consensus: require all three to beat trastuzumab
nd["beats_all3"]=(nd.ridge_score>-0.287)&(nd.facs_histgb>=tz_facs)&(nd.kd_rf>=tz_rf)
print(f"\ncandidates beating trastuzumab on ALL THREE models: {int(nd.beats_all3.sum())}")
nd["consensus"]=nd[["ridge_score","kd_rf"]].rank(pct=True).mean(axis=1)*0.5+nd.facs_histgb.rank(pct=True)*0.5
nd=nd.sort_values("consensus",ascending=False)
nd.to_csv("runs/shell_candidates.csv",index=False)
print("\n=== top 20 by CONSENSUS of three models ===")
print(nd[["cdrh3","n_mut","ridge_score","kd_rf","facs_histgb","consensus","beats_all3"]].head(20).to_string(index=False))
# and the RFD4 pool for comparison, same three models
d=pd.read_csv("runs/prod100/scored.csv"); ps=d[d.PASS_strict].copy()
Xp=onehot(ps.cdrh3); ps["facs_histgb"]=clf.predict_proba(Xp)[:,1]; ps["kd_rf"]=rf.predict(Xp)
print(f"\n=== RFD4 pool (n={len(ps)} novelty-passing) for comparison ===")
print(f"  facs_histgb: max={ps.facs_histgb.max():.3f} (tz {tz_facs:.3f})   >= tz: {int((ps.facs_histgb>=tz_facs).sum())}")
print(f"  kd_rf      : max={ps.kd_rf.max():.3f} (tz {tz_rf:.3f})   >= tz: {int((ps.kd_rf>=tz_rf).sum())}")
