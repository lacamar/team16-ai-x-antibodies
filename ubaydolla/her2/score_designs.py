import pickle, numpy as np, pandas as pd, csv, io, zipfile
from pathlib import Path
M=pickle.load(open("models/binding_clf.pkl","rb")); best=M["model"]; AA=M["AA"]; L=M["L"]
IDX={a:i for i,a in enumerate(AA)}
def onehot(seqs):
    X=np.zeros((len(seqs),L*20),dtype=np.float32)
    for n,s in enumerate(seqs):
        for p,c in enumerate(s[:L]):
            if c in IDX: X[n,p*20+IDX[c]]=1.0
    return X
TZ="WGGDGFYAMD"
cd=pd.read_csv("runs/prod100/cdrh3.csv")
sel=pd.read_csv("runs/prod100/refold_prep/selection.csv")
sel["design_id"]=sel.query_id
cd["design_id"]=cd.design
df=cd.merge(sel[["design_id","rank","hotspot_obedience","interface_hbonds","clash_fraction","pre_oracle_pass"]],on="design_id",how="left")
df["bind_score"]=best.predict_proba(onehot(df.cdrh3))[:,1]
df["n_mut_tz"]=[sum(a!=b for a,b in zip(s,TZ)) for s in df.cdrh3]
# trastuzumab baseline through the same scorer
tz_score=float(best.predict_proba(onehot([TZ]))[:,1][0])
print(f"TRASTUZUMAB baseline bind_score = {tz_score:.4f}")
df["beats_tz"]=df.bind_score>tz_score
# novelty vs all provided CDRH3 (525,063)
prov=set()
with zipfile.ZipFile("/home/hackathon/team16-ai-x-antibodies/data/her2/tz_her2_facs_3classes_524346.csv.zip") as z:
    with z.open(z.namelist()[0]) as f:
        for r in csv.DictReader(io.TextIOWrapper(f,"utf-8")):
            s=(r.get("cdrh3") or "").strip().upper()
            if s: prov.add(s)
for r in csv.DictReader(open("/home/hackathon/team16-ai-x-antibodies/data/her2/trastuzumab_her2_spr_all_768.csv")):
    s=(r.get("cdrh3") or "").strip().upper()
    if s: prov.add(s)
print(f"provided CDRH3 set: {len(prov)}")
def min_ham_to_provided(s, maxcheck=3):
    """exact min Hamming distance to provided set, searching radius 0,1,2"""
    if s in prov: return 0
    import itertools
    for rad in (1,2):
        for pos in itertools.combinations(range(len(s)),rad):
            for subs in itertools.product(AA,repeat=rad):
                if any(s[p]==c for p,c in zip(pos,subs)): continue
                t=list(s)
                for p,c in zip(pos,subs): t[p]=c
                if "".join(t) in prov: return rad
    return 3  # >=3
df["min_dist_provided"]=[min_ham_to_provided(s) for s in df.cdrh3]
# gates
df["gate_novel_provided"]=df.min_dist_provided>=3          # strict: >2 point mutations
df["gate_tz_strict"]=df.n_mut_tz>=4                        # identity <70% strict
df["gate_tz_lenient"]=df.n_mut_tz>=3
df["gate_cys"]=[(s.count("C")%2)==0 for s in df.cdrh3]
df["PASS_strict"]=df.gate_novel_provided&df.gate_tz_strict&df.gate_cys
df["PASS_lenient"]=df.gate_novel_provided&df.gate_tz_lenient&df.gate_cys
df=df.sort_values("bind_score",ascending=False)
df.to_csv("runs/prod100/scored.csv",index=False)
print("\n=== gate summary (n=100) ===")
print("novel vs provided (>=3 diffs):", int(df.gate_novel_provided.sum()))
print("min_dist_provided counts:", df.min_dist_provided.value_counts().to_dict())
print("PASS_strict:", int(df.PASS_strict.sum()), " PASS_lenient:", int(df.PASS_lenient.sum()))
print("beats trastuzumab bind_score:", int(df.beats_tz.sum()))
print("\n=== top 15 by bind_score (strict-passing only) ===")
c=["design_id","cdrh3","bind_score","n_mut_tz","min_dist_provided","hotspot_obedience","interface_hbonds","clash_fraction","rank"]
print(df[df.PASS_strict][c].head(15).to_string(index=False))
