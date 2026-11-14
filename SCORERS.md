# Scorers for the IgLM loop

Run with `/home/lm/.local/src/ai-antibodies/scribe/.venv/bin/python` from the repo root (`PY` below).

**Rule: loop reward = fast sequence surrogate (`reward.py`). Fold only the final shortlist.**
ipTM vs real affinity is Spearman 0.285 and a fold is ~75 s on one shared GPU; the surrogate is Spearman 0.72 (trained on the same 524k FACS + 768 SPR data you fine-tune on) and takes ms.

| Scorer | Scores | Speed | Quality | Call |
|---|---|---|---|---|
| `reward.py` | surrogate - stability penalty; novelty + developability hard gates (gated = -10) | ms | rides the surrogate | `SeqReward().reward(cdrh3, vh)` / `.batch(cdrh3s, vhs)` |
| `affinity_surrogate.py` | pKd, P(high), surrogate; `ood` flag if length off-distribution | ms (fit ~15 s once) | Spearman 0.72 | `AffinitySurrogate().fit().score(cdrh3)` |
| `novelty_filter.py` | CDRH3 >=2/10 from every provided seq, >=3/10 from trastuzumab; VH len<=250, Cys parity | ms | hard rule | `is_novel(cdrh3, by_len)`, `hard_checks(vh)` |
| `thermostability.py` | loop developability flags -> 0..1 | us | heuristic | `stability_score(cdrh3)` |
| `epitope_check.py` | HER2 epitope overlap vs 1N8Z + 4 hotspots | needs a folded CIF | judged property | `preservation(cif)` |
| Boltz-2 fold (VM) | ipTM, pLDDT, structure | ~75 s/design | ipTM rho 0.285: gate only | `score_shortlist.sh` |
| `rank.py` | gates -> surrogate + epitope bonus - stability, diversity dedup, top-20 | s | final | `rank.py candidates.csv out.csv` |

## Score one sequence
```bash
PY=/home/lm/.local/src/ai-antibodies/scribe/.venv/bin/python
$PY -c "from reward import SeqReward; print(SeqReward().reward('FLPQDAYHMG', vh=None))"
```
```python
from reward import SeqReward, GATED_OUT
rw = SeqReward()                                   # fits once, ~15 s; reuse the object
rw.reward(cdrh3, vh)                               # dict: reward, pkd, p_high, surrogate, stability, novel, ood, gate
rs = rw.batch(cdrh3s, vhs)                         # list of dicts (vhs optional)
rewards = [r["reward"] for r in rs]
kept = [(c, r["reward"]) for c, r in zip(cdrh3s, rs) if r["reward"] > GATED_OUT]   # drop gated before fine-tuning
```
`vh` = full heavy chain (trastuzumab framework with the CDRH3 swapped in); omit it to skip the Cys/length check.

## Gotchas
- Most 1-2 mutants of trastuzumab are in the provided data, so a model fine-tuned on it will often regenerate known sequences. These score `GATED_OUT` (-10): filter them, do not train on them.
- `ood=True` (non-10-residue CDRH3): surrogate unreliable.
- Epitope recall from template-guided folds is near 1.0 for most designs (the template steers the pose): treat as a coarse gate, not a ranker.
- Boltz queries are VL + VH + HER2 C/D; `iptm` = `interface_predicted_tm_score`; `min_ipae` is blank on this path (only in the detailed CBOR).

## Score + fold a shortlist
```bash
bash score_shortlist.sh topk.csv runs/round1      # topk.csv: cdrh3,vh_sequence[,reward]  -> runs/round1/ranked.csv
```
Loop architecture and pseudocode: `docs/IGLM_LOOP.md`.
