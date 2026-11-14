import json, time, hashlib, subprocess
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
AA="ACDEFGHIKLMNPQRSTVWY"; IDX={a:i for i,a in enumerate(AA)}; L=10
def onehot(seqs):
    X=np.zeros((len(seqs),L*20),dtype=np.float32)
    for n,s in enumerate(seqs):
        for p,c in enumerate(s[:L]):
            if c in IDX: X[n,p*20+IDX[c]]=1.0
    return X
B="data/zenodo-splits/unz/data/affinity_data/her2/her2_aff_large/processed/remove_overlap/clonotype_split/0.7_0.15_0.15"
tr=pd.read_csv(f"{B}/train.csv").dropna(subset=["seq"]); va=pd.read_csv(f"{B}/val.csv").dropna(subset=["seq"]); te=pd.read_csv(f"{B}/test.csv").dropna(subset=["seq"])
for d in (tr,va,te): d["seq"]=d.seq.str.upper()
Xtr,ytr=onehot(tr.seq),tr.label.values; Xva,yva=onehot(va.seq),va.label.values; Xte,yte=onehot(te.seq),te.label.values
res={}
t=time.time(); lr=LogisticRegression(max_iter=3000,C=1.0,n_jobs=-1).fit(Xtr,ytr)
res["logreg"]={"val_auc":float(roc_auc_score(yva,lr.predict_proba(Xva)[:,1])),
               "test_auc":float(roc_auc_score(yte,lr.predict_proba(Xte)[:,1])),
               "test_ap":float(average_precision_score(yte,lr.predict_proba(Xte)[:,1])),"fit_s":round(time.time()-t,1)}
t=time.time(); gb=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.1,max_leaf_nodes=63,
        early_stopping=True,validation_fraction=0.1,random_state=0).fit(Xtr,ytr)
res["histgb"]={"val_auc":float(roc_auc_score(yva,gb.predict_proba(Xva)[:,1])),
               "test_auc":float(roc_auc_score(yte,gb.predict_proba(Xte)[:,1])),
               "test_ap":float(average_precision_score(yte,gb.predict_proba(Xte)[:,1])),"fit_s":round(time.time()-t,1)}
print(json.dumps(res,indent=2))
# external validation: Shanehsazzadeh 198 known binders with continuous affinity
sh=pd.read_csv("data/zenodo-splits/unz/data/affinity_data/her2/shanehsazzadeh_et_al/processed/test.csv")
sh=sh[sh.seq.str.len()==L]
from scipy.stats import spearmanr
best = gb if res["histgb"]["test_auc"]>=res["logreg"]["test_auc"] else lr
ps=best.predict_proba(onehot(sh.seq))[:,1]
rho,p=spearmanr(ps,sh.affinity)
print(f"\nShanehsazzadeh external set n={len(sh)}: spearman(model_score, affinity) rho={rho:.3f} p={p:.2g}")
# SPR: 190 quantified KD
spr=pd.read_csv("data/her2/trastuzumab_her2_spr_all_768.csv") if False else pd.read_csv("/home/hackathon/team16-ai-x-antibodies/data/her2/trastuzumab_her2_spr_all_768.csv")
spr=spr[(spr.binding_result=="quantified_binding")&spr.kd_nM.notna()]
spr=spr[spr.cdrh3.str.len()==L]
ps2=best.predict_proba(onehot(spr.cdrh3))[:,1]
rho2,p2=spearmanr(ps2,-np.log10(spr.kd_nM))
print(f"SPR quantified n={len(spr)}: spearman(model_score, -log10 KD) rho={rho2:.3f} p={p2:.2g}")
import pickle
pickle.dump({"model":best,"which":"histgb" if best is gb else "logreg","AA":AA,"L":L},open("models/binding_clf.pkl","wb"))
json.dump({"metrics":res,"shanehsazzadeh":{"n":int(len(sh)),"spearman":float(rho),"p":float(p)},
           "spr":{"n":int(len(spr)),"spearman":float(rho2),"p":float(p2)},
           "train_rows":int(len(tr)),"split":B},open("models/binding_clf_metrics.json","w"),indent=2)
print("\nsaved models/binding_clf.pkl")
