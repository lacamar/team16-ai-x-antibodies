# IgLM generate -> score -> fold -> fine-tune loop

Reward = `reward.py` (ms, Spearman 0.72). Folding = last-stage gate on top-K only (~75 s each, shared GPU, ipTM rho 0.285).
Scorer reference: `SCORERS.md`.

```
fine-tune IgLM on the 524k CDRH3 data            (round 0)
repeat:
  1. generate   N variants (e.g. 5-50k)           optimisation: mutate trastuzumab CDRH3 | de novo: free sampling
  2. reward     rw.batch(cdrh3s, vhs)             FAST, all N; gated designs get -10
  3. keep top-K by reward, K ~ 20-30, diversity-deduped
  4. fold K     bash score_shortlist.sh topk.csv runs/roundN      SLOW (K x ~75 s), VM Boltz-2
  5. epitope + rank                               (inside score_shortlist.sh) -> runs/roundN/ranked.csv
  6. export     (sequence, reward) pairs for ALL ungated N  -> IgLM fine-tune data (reward-weighted / top-fraction)
  7. fine-tune IgLM; track best ranked.csv final_score  -> stop at plateau
```

```python
from reward import SeqReward, GATED_OUT
import csv, subprocess
rw = SeqReward()
for rnd in range(R):
    cdrh3s = iglm.generate(N)                              # teammate's side
    vhs = [FRAMEWORK_VH.replace(TZ_CDRH3, c) for c in cdrh3s]   # trastuzumab framework + new CDRH3
    rs = rw.batch(cdrh3s, vhs)
    scored = sorted(((r["reward"], c, v) for r, c, v in zip(rs, cdrh3s, vhs) if r["reward"] > GATED_OUT), reverse=True)

    topk, seen = [], []                                    # ponytail: greedy dedup, O(N*K)
    for s, c, v in scored:
        if all(sum(a != b for a, b in zip(c, o)) >= 2 for o in seen if len(o) == len(c)):
            topk.append((c, v, s)); seen.append(c)
        if len(topk) == K: break
    with open(f"topk{rnd}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cdrh3", "vh_sequence", "reward"]); w.writerows(topk)
    subprocess.run(["bash", "score_shortlist.sh", f"topk{rnd}.csv", f"runs/round{rnd}"], check=True)

    with open(f"finetune{rnd}.csv", "w", newline="") as f:  # step 6
        w = csv.writer(f); w.writerow(["cdrh3", "reward"]); w.writerows((c, s) for s, c, _ in scored)
    iglm.finetune(f"finetune{rnd}.csv")
```

## What `score_shortlist.sh` does (steps 4-5)
Each step reuses an existing module; the VM command is the Boltz-2 one from `docs/VM-RUN.md`.
```
shortlist.py prep      re-gate (novelty, hard_checks) -> queries.json (VL + VH + HER2 C/D, from data/boltz_query_base.json)
scp + docker boltz2    apheris_boltz.cli predict, template = runs/opt/prepared/holo/template.cif already on the VM
scp results back
shortlist.py metrics   Boltz-2 *_scores.json -> refold_metrics.csv
make_candidates.py     -> candidates.csv (cdrh3, vh, iptm, design_cif)
rank.py                gates, surrogate + epitope bonus (epitope_check.preservation) - stability -> ranked.csv
```
Final submission: `selected==True` rows of `ranked.csv` (max 20 per track).

Notes
- Requires the VM (`~/.ssh/team-16`), GPU is shared: check it is free; keep K small.
- Needs `~/lachlan/rfd4-docs/runs/opt/prepared/holo/template.cif` on the VM; change `TPL` in the script if moved.
- Verified end-to-end with K=2 (1m43s): prep -> fold -> metrics -> candidates -> rank.
