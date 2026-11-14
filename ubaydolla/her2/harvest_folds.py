import json, glob, os, sys, pandas as pd
def harvest(root):
    rows=[]
    for sc in sorted(glob.glob(f"{root}/*/seed_*/*_scores.json")):
        qid=os.path.basename(os.path.dirname(os.path.dirname(sc)))
        d=json.load(open(sc))
        r={"query_id":qid,
           "plddt":d.get("average_predicted_local_distance_difference_test"),
           "pde":d.get("global_predicted_distance_error"),
           "ptm":d.get("predicted_tm_score"),
           "iptm":d.get("interface_predicted_tm_score"),
           "ranking":d.get("model_ranking_score")}
        cc=d.get("chain_chain_interface_predicted_tm_score")
        cp=d.get("chain_predicted_tm_score")
        # chains order A,B,C,D -> indices 0..3 ; binder = B (1), HER2 = C(2), D(3)
        def g(m,i,j):
            try: return m[i][j]
            except Exception: return None
        if isinstance(cc,list):
            r["iptm_B_C"]=g(cc,1,2); r["iptm_B_D"]=g(cc,1,3); r["iptm_A_C"]=g(cc,0,2)
        if isinstance(cp,list) and len(cp)>1: r["ptm_B"]=cp[1]
        rows.append(r)
    return pd.DataFrame(rows)
root=sys.argv[1]; out=sys.argv[2]
f=harvest(root)
f["iptm_binder_target"]=f[["iptm_B_C","iptm_B_D"]].max(axis=1)
f.to_csv(out,index=False)
print(f"harvested {len(f)} folds -> {out}")
print(f.describe().loc[["mean","min","max"]][["plddt","ptm","iptm","iptm_B_C","iptm_B_D","ranking"]].round(3).to_string())
