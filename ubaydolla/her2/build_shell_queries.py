import json, pandas as pd
VL="DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
VH_PRE="EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR"
VH_POST="YWGQGTLVTVSS"
C="CHQLCARGHCWGPGPTQCVNCSQFLRGQECVEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVARCPS"
D="IWKFPDEEGACQPCPIN"
Cids=[str(i) for i in range(489,581)]; Dids=[str(i) for i in range(591,608)]
TZ="WGGDGFYAMD"
nd=pd.read_csv("runs/shell_candidates.csv").head(40)
# always include trastuzumab itself as the in-run baseline control
rows=[("BASELINE_trastuzumab",TZ)]+[(f"shell_{i:02d}",s) for i,s in enumerate(nd.cdrh3)]
qs=[]
for qid,cdr in rows:
    vh=VH_PRE+cdr+VH_POST
    qs.append({"query_id":qid,"chains":[
        {"chain_id":"A","sequence":VL,"msa_path":None},
        {"chain_id":"B","sequence":vh,"msa_path":None},
        {"chain_id":"C","sequence":C,"residue_ids":Cids,"msa_path":None},
        {"chain_id":"D","sequence":D,"residue_ids":Dids,"msa_path":None}]})
json.dump({"queries":qs},open("runs/shell_queries.json","w"),indent=1)
print(f"wrote {len(qs)} queries (incl. trastuzumab baseline control) -> runs/shell_queries.json")
print("VH length:",len(VH_PRE+TZ+VH_POST))
