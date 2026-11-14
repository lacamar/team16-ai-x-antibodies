"""Exact search of the novelty-compliant mutational shell closest to trastuzumab.
Ridge on one-hot is ADDITIVE, so score(s) = intercept + sum_p w[p, s[p]].
Per-position substitution deltas therefore rank exactly; no sampling needed."""
import numpy as np, pandas as pd, pickle, itertools, csv, io, zipfile, json
AA="ACDEFGHIKLMNPQRSTVWY"; IDX={a:i for i,a in enumerate(AA)}; L=10; TZ="WGGDGFYAMD"
M=pickle.load(open("models/kd_reg.pkl","rb")); ridge=M["ridge"]
W=ridge.coef_.reshape(L,20); b=float(ridge.intercept_)
def add_score(s): return b+sum(W[p,IDX[c]] for p,c in enumerate(s))
tz_s=add_score(TZ)
print(f"ridge additive score: trastuzumab={tz_s:.3f}")
delta=np.zeros((L,20))
for p in range(L):
    for a,i in IDX.items(): delta[p,i]=W[p,i]-W[p,IDX[TZ[p]]]
print("\nbest substitution at each position (delta in log10 KD units, + = tighter):")
for p in range(L):
    order=np.argsort(-delta[p]); top=[(AA[i],round(float(delta[p,i]),3)) for i in order[:3] if AA[i]!=TZ[p]]
    print(f"  pos{p+1} {TZ[p]}: {top}")
# provided set
prov=set()
with zipfile.ZipFile("/home/hackathon/team16-ai-x-antibodies/data/her2/tz_her2_facs_3classes_524346.csv.zip") as z:
    with z.open(z.namelist()[0]) as f:
        for r in csv.DictReader(io.TextIOWrapper(f,"utf-8")):
            s=(r.get("cdrh3") or "").strip().upper()
            if s: prov.add(s)
for r in csv.DictReader(open("/home/hackathon/team16-ai-x-antibodies/data/her2/trastuzumab_her2_spr_all_768.csv")):
    s=(r.get("cdrh3") or "").strip().upper()
    if s: prov.add(s)
prov10={s for s in prov if len(s)==L}
print(f"\nprovided 10-mers: {len(prov10)}")
def min_dist_ge3(s):
    """True iff Hamming distance to every provided 10-mer is >= 3."""
    if s in prov10: return False
    for rad in (1,2):
        for pos in itertools.combinations(range(L),rad):
            for subs in itertools.product(AA,repeat=rad):
                if any(s[p]==c for p,c in zip(pos,subs)): continue
                t=list(s)
                for p,c in zip(pos,subs): t[p]=c
                if "".join(t) in prov10: return False
    return True
# enumerate shells k=4,5 keeping the best substitutions per position
results=[]
for k in (4,5):
    # candidate substitutions: top 6 per position by delta
    cand={p:[AA[i] for i in np.argsort(-delta[p])[:7] if AA[i]!=TZ[p]][:6] for p in range(L)}
    n=0
    for pos in itertools.combinations(range(L),k):
        for subs in itertools.product(*[cand[p] for p in pos]):
            t=list(TZ)
            for p,c in zip(pos,subs): t[p]=c
            s="".join(t); n+=1
            results.append((s,add_score(s),k))
    print(f"shell k={k}: enumerated {n:,} candidates")
df=pd.DataFrame(results,columns=["cdrh3","ridge_score","n_mut"]).drop_duplicates("cdrh3")
df=df.sort_values("ridge_score",ascending=False)
print(f"total unique: {len(df):,}; best additive score {df.ridge_score.max():.3f} vs trastuzumab {tz_s:.3f}")
# novelty-filter the top slice only (expensive check)
keep=[]
for s,sc,k in df[["cdrh3","ridge_score","n_mut"]].itertuples(index=False):
    if min_dist_ge3(s):
        keep.append((s,sc,k))
        if len(keep)>=400: break
nd=pd.DataFrame(keep,columns=["cdrh3","ridge_score","n_mut"])
nd["pred_kd_nM"]=10**(-nd.ridge_score)
nd.to_csv("runs/shell_candidates.csv",index=False)
print(f"\nnovelty-passing candidates kept: {len(nd)}")
print(f"trastuzumab additive-equivalent KD = {10**-tz_s:.3f} nM")
print("\n=== top 20 novelty-compliant shell candidates ===")
print(nd.head(20).to_string(index=False))
