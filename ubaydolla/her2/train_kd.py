import numpy as np, pandas as pd, pickle, json
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, pearsonr
AA="ACDEFGHIKLMNPQRSTVWY"; IDX={a:i for i,a in enumerate(AA)}; L=10
def onehot(seqs):
    X=np.zeros((len(seqs),L*20),dtype=np.float32)
    for n,s in enumerate(seqs):
        for p,c in enumerate(s[:L]):
            if c in IDX: X[n,p*20+IDX[c]]=1.0
    return X
TZ="WGGDGFYAMD"
spr=pd.read_csv("/home/hackathon/team16-ai-x-antibodies/data/her2/trastuzumab_her2_spr_all_768.csv")
spr=spr[(spr.binding_result=="quantified_binding")&spr.kd_nM.notna()]
spr["cdrh3"]=spr.cdrh3.str.upper()
spr=spr[spr.cdrh3.str.len()==L].copy()
spr["y"]=-np.log10(spr.kd_nM)          # higher = tighter
# trastuzumab reference rows
tzr=spr[spr.cdrh3==TZ]
print(f"n={len(spr)} quantified 10-mers; trastuzumab rows={len(tzr)} KD={sorted(tzr.kd_nM.round(3))} -> y={tzr.y.mean():.3f}")
# collapse duplicate sequences (median y)
g=spr.groupby("cdrh3",as_index=False).agg(y=("y","median"),n=("y","size"))
print(f"unique sequences: {len(g)}")
X=onehot(g.cdrh3); y=g.y.values
cv=KFold(5,shuffle=True,random_state=0)
for name,mdl in [("ridge",RidgeCV(alphas=np.logspace(-2,3,20))),
                 ("rf",RandomForestRegressor(n_estimators=600,min_samples_leaf=2,random_state=0,n_jobs=-1))]:
    p=cross_val_predict(mdl,X,y,cv=cv)
    print(f"{name}: 5-fold CV spearman={spearmanr(p,y).statistic:.3f} pearson={pearsonr(p,y)[0]:.3f} MAE={np.abs(p-y).mean():.3f} log10 units")
rf=RandomForestRegressor(n_estimators=600,min_samples_leaf=2,random_state=0,n_jobs=-1).fit(X,y)
ridge=RidgeCV(alphas=np.logspace(-2,3,20)).fit(X,y)
pickle.dump({"rf":rf,"ridge":ridge,"AA":AA,"L":L,"tz_y":float(tzr.y.mean())},open("models/kd_reg.pkl","wb"))
# score designs
d=pd.read_csv("runs/prod100/scored.csv")
Xd=onehot(d.cdrh3)
d["pred_y_rf"]=rf.predict(Xd); d["pred_y_ridge"]=ridge.predict(Xd)
d["pred_kd_nM_rf"]=10**(-d.pred_y_rf)
tz_pred_rf=float(rf.predict(onehot([TZ]))[0]); tz_pred_ridge=float(ridge.predict(onehot([TZ]))[0])
print(f"\ntrastuzumab: measured y={tzr.y.mean():.3f} (KD {10**-tzr.y.mean():.3f} nM); model predicts rf={tz_pred_rf:.3f} ridge={tz_pred_ridge:.3f}")
d["beats_tz_rf"]=d.pred_y_rf>tz_pred_rf
d["beats_tz_ridge"]=d.pred_y_ridge>tz_pred_ridge
d.to_csv("runs/prod100/scored.csv",index=False)
ps=d[d.PASS_strict]
print(f"\nof {len(ps)} novelty-passing designs: beat tz by rf={int(ps.beats_tz_rf.sum())}, by ridge={int(ps.beats_tz_ridge.sum())}")
print("\n=== top 15 novelty-passing, by predicted affinity (rf) ===")
c=["design_id","cdrh3","pred_y_rf","pred_kd_nM_rf","bind_score","n_mut_tz","hotspot_obedience","interface_hbonds"]
print(ps.sort_values("pred_y_rf",ascending=False)[c].head(15).to_string(index=False))
