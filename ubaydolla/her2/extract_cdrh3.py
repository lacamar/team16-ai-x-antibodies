import re, sys, glob, csv, os
TZ_VH_PRE, TZ_VH_POST = "YYCSR", "YWGQGTLVTVSS"
TZ_CDRH3 = "WGGDGFYAMD"
def seqs_from_cif(p):
    txt = open(p).read()
    # entity_poly one-letter codes appear as ;SEQ; blocks
    blocks = re.findall(r";\s*([A-Z\n]+?)\s*;", txt)
    out = []
    for b in blocks:
        s = re.sub(r"\s+", "", b)
        if s and set(s) <= set("ACDEFGHIKLMNPQRSTVWY") and len(s) > 20:
            out.append(s)
    # dedupe preserving order
    seen=set(); u=[]
    for s in out:
        if s not in seen: seen.add(s); u.append(s)
    return u
rows=[]
for p in sorted(glob.glob("runs/prod100/design/generation/inference_samples/*/*_generated.cif")):
    name=os.path.basename(os.path.dirname(p))
    ss=seqs_from_cif(p)
    vh=[s for s in ss if TZ_VH_PRE in s and TZ_VH_POST in s]
    if not vh:
        rows.append({"design":name,"cdrh3":"","vh":"","n_mut":"","note":"VH not parsed"}); continue
    vh=vh[0]
    i=vh.index(TZ_VH_PRE)+len(TZ_VH_PRE); j=vh.index(TZ_VH_POST)
    cdrh3=vh[i:j]
    nm=sum(1 for a,b in zip(cdrh3,TZ_CDRH3) if a!=b) if len(cdrh3)==len(TZ_CDRH3) else None
    rows.append({"design":name,"cdrh3":cdrh3,"vh":vh,
                 "n_mut": nm if nm is not None else "len%d"%len(cdrh3),
                 "note":"" if len(cdrh3)==10 else "length %d"%len(cdrh3)})
with open("runs/prod100/cdrh3.csv","w",newline="") as f:
    w=csv.DictWriter(f,["design","cdrh3","n_mut","note","vh"]); w.writeheader(); w.writerows(rows)
ok=[r for r in rows if r["cdrh3"]]
print("parsed %d/%d"%(len(ok),len(rows)))
from collections import Counter
print("CDRH3 lengths:",Counter(len(r["cdrh3"]) for r in ok))
print("n_mut distribution:",Counter(r["n_mut"] for r in ok))
print("unique CDRH3:",len({r["cdrh3"] for r in ok}))
