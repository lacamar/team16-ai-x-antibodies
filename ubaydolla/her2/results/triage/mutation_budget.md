# CDRH3 mutation budget

- modal CDRH3 length: **10**
- to clear the <70% identity rule you need at least **4** substitutions (4/10 = 40.0% divergence). Aim for 5-6 so abguard's identity calculation disagreeing with yours does not reject the design.

## Conserve (highest high-vs-low divergence = carries the binding signal)

|   position | wt_aa   |   divergence_high_vs_low | best_aa   | top5_enriched   |
|-----------:|:--------|-------------------------:|:----------|:----------------|
|          3 | G       |                    0.427 | A         | A,G,R,P,D       |
|          8 | A       |                    0.412 | T         | T,S,K,N,A       |
|          5 | G       |                    0.385 | R         | R,P,M,G,T       |
|         10 | D       |                    0.328 | D         | D,N,Y,E,K       |

## Spend your identity budget here (most tolerant positions)

|   position | wt_aa   |   divergence_high_vs_low | best_aa   | top5_enriched   |
|-----------:|:--------|-------------------------:|:----------|:----------------|
|          4 | D       |                    0.297 | G         | G,P,A,W,T       |
|          2 | G       |                    0.285 | A         | A,G,D,P,Y       |
|          6 | F       |                    0.187 | G         | G,D,P,V,A       |
|          1 | W       |                    0.145 | N         | N,Y,H,R,W       |
|          7 | Y       |                    0.099 | W         | W,G,D,E,F       |
|          9 | M       |                    0.099 | A         | A,V,L,K,F       |

Pick substitutions from `top5_enriched` at the tolerant positions: you then satisfy the novelty rule using residues the FACS screen says binders prefer, instead of random drift away from the epitope.

## Liability motifs in the WT CDRH3

- `DG@4:isomerisation`
- `W@1:oxidation-prone`
- `M@9:oxidation-prone`

Removing these at tolerant positions is a free thermostability hypothesis: it spends identity budget you must spend anyway.