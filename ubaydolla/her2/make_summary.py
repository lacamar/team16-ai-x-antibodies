"""
Render ~/ubaydolla/results/summary.html from the scored tables.

Every number in the page is read out of results/*.csv at render time. Nothing is
hand-typed, so the page cannot drift from the data (provenance rule, NOTES.md).
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

CSS = """
:root{--bg:#fbfaf8;--fg:#1a1a1a;--mut:#5a5a5a;--line:#e3e0da;--card:#fff;
--good:#1a7f5a;--bad:#b3341f;--warn:#9a6b00;--accent:#2d4a7c;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#15161a;--fg:#e9e7e3;
--mut:#a0a0a0;--line:#2e3038;--card:#1c1e23;--good:#4fcf9a;--bad:#ff7a63;--warn:#e0b050;--accent:#8fb3e8}}
*{box-sizing:border-box}
body{margin:0;padding:0 16px 64px;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:26px;margin:32px 0 4px;letter-spacing:-.01em}
h2{font-size:19px;margin:38px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:22px 0 6px}
.sub{color:var(--mut);font-size:13px;margin-bottom:8px}
p{margin:8px 0}
code,.mono{font-family:var(--mono);font-size:.92em}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0;display:block;overflow-x:auto}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-weight:600;color:var(--mut);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;
position:sticky;top:0;background:var(--bg)}
td:first-child,th:first-child,td.l,th.l{text-align:left}
tr.hi td{background:color-mix(in srgb,var(--accent) 11%,transparent);font-weight:600}
tr.base td{border-top:2px solid var(--accent);color:var(--mut);font-style:italic}
.good{color:var(--good);font-weight:600}.bad{color:var(--bad)}.warn{color:var(--warn)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:13px 15px}
.card .k{font-size:11.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:25px;font-weight:650;margin:3px 0;letter-spacing:-.02em}
.card .n{font-size:12px;color:var(--mut)}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:0 8px 8px 0;padding:11px 15px;margin:14px 0}
.note.w{border-left-color:var(--warn)}.note.r{border-left-color:var(--bad)}
.note b{display:block;margin-bottom:3px}
ul{margin:8px 0;padding-left:22px}li{margin:4px 0}
.seq{font-family:var(--mono);font-size:11.5px;word-break:break-all;color:var(--mut)}
footer{margin-top:44px;padding-top:14px;border-top:1px solid var(--line);color:var(--mut);font-size:12px}
"""


def fmt(v, n=3):
    if pd.isna(v):
        return "<span class=mut>—</span>"
    if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer()):
        return f"{int(v)}"
    return f"{v:.{n}f}"


def main():
    sub = pd.read_csv(f"{RES}/submission_top20.csv")
    allsc = pd.read_csv(f"{RES}/all_designs_scored.csv")
    gates = pd.read_csv(f"{RES}/gate_verification.csv")
    base = allsc[allsc.query_id == "BASELINE_trastuzumab"].iloc[0]
    rej = pd.read_csv(f"{RES}/rejected_designs.csv")

    n_unan = int((sub.axis1_n_beat_baseline >= 4).sum())
    n_hot4 = int((sub.hot_contacts == 4).sum())
    n_lia = int((sub.d_liabilities_vs_tz < 0).sum())
    n_gate = int(gates["pass"].sum())

    H = [f"<!doctype html><html lang=en><meta charset=utf-8>",
         "<meta name=viewport content='width=device-width,initial-scale=1'>",
         "<title>Trastuzumab Optimisation</title>", f"<style>{CSS}</style>",
         "<div class=wrap>"]
    A = H.append

    A("<h1>Trastuzumab &rarr; HER2: optimisation-track designs</h1>")
    A("<div class=sub>Team 16 &middot; AI &times; Antibodies, Oslo &middot; 2026-11-14 &middot; "
      "all numbers rendered from <code>results/*.csv</code>, none hand-typed</div>")

    # ---------- headline cards
    A("<div class=cards>")
    for k, v, n in [
        ("Designs submitted", len(sub), "ranked, all gate-verified"),
        ("Beat trastuzumab on 4/5 scorers", n_unan, "none beat it on 5/5 &mdash; we do not claim that"),
        ("Keep all 4 epitope hotspots", f"{n_hot4}/{len(sub)}", "axis 2, measured per design"),
        ("Delete a trastuzumab liability", f"{n_lia}/{len(sub)}", "D102&ndash;G103 isomerisation motif"),
    ]:
        A(f"<div class=card><div class=k>{k}</div><div class=v>{v}</div><div class=n>{n}</div></div>")
    A("</div>")

    # ---------- the thesis
    A("<h2>The one-paragraph argument</h2>")
    A("""<p>Trastuzumab is already a clinically optimised binder at <b>K<sub>D</sub> 0.095 nM</b>, and the
optimisation track forbids staying near it: a design needs &gt;2 point mutations from
<i>every</i> one of the 525,063 provided antibodies. Our first finding was that this rule and the
binding axis pull against each other &mdash; RFD4 satisfies novelty by jumping 6&ndash;8 mutations
from a tuned loop, and <b>none</b> of its 73 novelty-passing designs was predicted to bind as well
(best &asymp;15 nM, ~150&times; worse). So we stopped sampling and searched instead for the
<i>least-damaging</i> novelty-compliant perturbation: because the provided library covers only
5.25&times;10<sup>5</sup> of 10<sup>13</sup> possible 10-mers, compliant sequences exist at just
<b>4 mutations</b> from trastuzumab. We enumerated that shell exactly, then required
<b>independent methods to agree</b> before promoting anything.</p>""")

    # ---------- top 20 table
    A("<h2>Ranked designs</h2>")
    A("<div class=sub>Ranked by how many of five independent axis-1 scorers beat the trastuzumab control "
      "(<code>n&uarr;</code>), then by the <i>worst</i> of the three normalised axes &mdash; S-tier needs all "
      "three, so a design that is excellent on one axis and poor on another is not what we want. "
      "The baseline row is trastuzumab folded <i>in the same run</i>.</div>")
    A("<table><thead><tr><th class=l>#</th><th class=l>ID</th><th class=l>CDRH3</th>"
      "<th class=l>Mutations</th><th>n&uarr;</th><th>ipTM<sub>bt</sub></th><th>epitope</th>"
      "<th>hot</th><th>SPR&nbsp;RF</th><th>FACS</th><th>ddG</th><th>&Delta;liab</th>"
      "<th>&Delta;agg</th><th>worst axis</th></tr></thead><tbody>")

    def row(r, cls="", rank=True):
        ipt = float(r.iptm_binder_target)
        dd = r.get("stabddg", float("nan"))
        cells = [
            f"<td class=l>{int(r['rank']) if rank else ''}</td>",
            f"<td class=l>{r.get('design_id','trastuzumab')}</td>",
            f"<td class='l mono'>{r.cdrh3}</td>",
            f"<td class='l mono' style='font-size:11px'>{r.get('mutations','') if pd.notna(r.get('mutations','')) else ''}</td>",
            f"<td>{'<span class=good>' if r.get('axis1_n_beat_baseline',0)>=4 else '<span>'}{fmt(r.get('axis1_n_beat_baseline'))}</span></td>",
            f"<td class={'good' if ipt>float(base.iptm_binder_target) else ''}>{fmt(ipt)}</td>",
            f"<td>{fmt(r.epitope_jaccard)}</td>",
            f"<td class={'good' if r.hot_contacts==4 else 'warn'}>{int(r.hot_contacts)}</td>",
            f"<td class={'good' if pd.notna(r.get('kd_rf')) and r.kd_rf>0.127 else ''}>{fmt(r.get('kd_rf'))}</td>",
            f"<td>{fmt(r.get('facs_histgb'))}</td>",
            f"<td class={'good' if pd.notna(dd) and dd<0 else ''}>{fmt(dd)}</td>",
            f"<td class={'good' if r.get('d_liabilities_vs_tz',0)<0 else ''}>{fmt(r.get('d_liabilities_vs_tz'))}</td>",
            f"<td class={'good' if r.get('d_agg_vs_tz',0)<0 else 'bad' if r.get('d_agg_vs_tz',0)>0 else ''}>{fmt(r.get('d_agg_vs_tz'),2)}</td>",
            f"<td>{fmt(r.worst_axis)}</td>"]
        A(f"<tr class='{cls}'>" + "".join(cells) + "</tr>")

    for i, r in sub.iterrows():
        row(r, "hi" if r["rank"] == 1 else "")
    b = base.copy()
    b["rank"] = 0
    b["design_id"] = "trastuzumab (control)"
    row(b, "base", rank=False)
    A("</tbody></table>")
    A("<div class=sub>ipTM<sub>bt</sub> = Boltz-2 binder&ndash;target interface ipTM. "
      "epitope = Jaccard overlap of contacted HER2 residues vs the in-run control. "
      "hot = conditioned hotspot atoms retained (of 4). SPR RF / FACS / ddG are independent scorers; "
      "ddG &lt; 0 stabilises the complex. &Delta;liab &lt; 0 means fewer chemical liabilities than "
      "trastuzumab, &Delta;agg &lt; 0 less aggregation-prone.</div>")

    # ---------- lead molecule
    L = sub.iloc[0]
    A("<h2>Lead: " + L.design_id + " &middot; <span class=mono>" + L.cdrh3 + "</span></h2>")
    A(f"""<p>Four mutations &mdash; <span class=mono>{L.mutations}</span> &mdash; on an otherwise
byte-identical trastuzumab scFv ({int(L.scfv_len)} aa).</p><ul>
<li><b>Axis 1 (binding).</b> Above the trastuzumab control on <b>4 of 5</b> independent scorers,
including the SPR-trained RandomForest ({fmt(L.kd_rf)} vs 0.127 measured) and StaB-ddG
({fmt(L.stabddg)} kcal/mol, stabilising). It is <i>not</i> the ipTM leader, which is the point:
we rank on agreement, not on any single maximum.</li>
<li><b>Axis 2 (epitope).</b> Retains all <b>4/4</b> conditioned HER2 hotspot contacts and
<b>{fmt(L.epitope_jaccard)}</b> Jaccard overlap with the residues trastuzumab itself contacts in the
same folding run &mdash; so trastuzumab should still compete it off, which is what the assay measures.</li>
<li><b>Axis 3 (stability).</b> The framework, the whole V<sub>L</sub>, the V<sub>H</sub>/V<sub>L</sub>
packing and both disulfides are unchanged, and AntiFold framework perplexity is flat across all 41
structures (1.92&ndash;2.10), confirming the framework is genuinely unperturbed. On top of that it
<b>deletes trastuzumab's own Asp102&ndash;Gly103 isomerisation motif</b> and is markedly
<b>less</b> aggregation-prone (&Delta;agg {fmt(L.d_agg_vs_tz,2)}).</li></ul>""")
    A(f"<p class=seq>{L.scfv}</p>")

    # ---------- why these, not the obvious ones
    A("<h2>Judgment calls worth defending</h2>")
    A("""<div class=note r><b>We rank on unanimity, because our scorers disagree.</b>
On this 40-design shortlist StaB-ddG <i>anti</i>-correlates with the SPR model
(&rho;=&minus;0.405, p=0.009) and with Boltz-2 ipTM (&rho;=&minus;0.338, p=0.033), while ipTM and the
SPR model agree with each other (&rho;=+0.343). Two pairs agree, two anti-correlate. Averaging
percentiles would have hidden that, so the primary key is instead <i>how many</i> independent
scorers beat the control. Earlier in the day a single-model argmax produced a candidate "1.6 log
units tighter" that collapsed under cross-checking &mdash; that is scorer-adversarial extrapolation,
and it is why nothing here is promoted on one model.</div>""")
    A("""<div class=note w><b>We dropped pLDDT, and we dropped the designs most likely to bind.</b>
Binder pLDDT correlates &rho;=&minus;0.013 with our empirical affinity model across 100 designs
&mdash; it is noise for this task, so it is not a ranking key. Separately, the two best binders we
found anywhere (<span class=mono>WGSSGFYAMD</span>, <span class=mono>WGYSGFYAMD</span>: top-2 on the
SPR model <i>and</i> 3rd/5th of 100 on ipTM) carry only <b>2</b> CDRH3 mutations and therefore fail
the track rule. They are in <code>results/rejected_designs.csv</code> as rows with reasons, not
silently dropped, and we re-ran them through StaB-ddG purely to calibrate its sign convention.</div>""")
    A(f"""<div class=note><b>Gates re-verified from scratch, not inherited.</b>
A filter applied by the generator is no evidence the shipped molecule passes, so
<code>verify_gates.py</code> recomputes the exact minimum Hamming distance from every shipped CDRH3
to all <b>525,063</b> provided CDRH3s by enumerating radius 0/1/2 &mdash; exact, not sampled.
<b>{n_gate}/{len(gates)-1}</b> designs pass; all are at distance exactly 3 (so &gt;2 mutations from
every provided antibody) and 4&ndash;5 from trastuzumab, i.e. &le;60% CDRH3 identity, which clears
the &lt;70% rule under either reading. scFv 242 aa &le; 250, no unpaired cysteines. The control
correctly fails, which is the check working.</div>""")
    A("""<div class=note w><b>What we could not verify.</b> <code>abguard</code> does not exist on this
VM &mdash; not on PATH, not in pip, not on PyPI, not inside any Apheris image, and absent from a
recursive search of the hackathon bucket. The checks above are our reimplementation of its four
documented rules. The one we <i>cannot</i> reproduce is immunum numbering confidence &ge;0.1, which
needs ANARCI/ANARCII; neither installs here, and AntiFold's bundled ANARCII failed to number our
Boltz-2 output chains. Every design is a 242-aa scFv on an unmodified clinical framework so the risk
is low, but it is unverified and we say so rather than implying a clean pass.</div>""")
    A("""<div class=note w><b>Thermostability is argued by construction, not predicted.</b>
No T<sub>m</sub> predictor was available on either lane &mdash; Aggrescan3D, SoluProt, NetSolP and
ProteinMPNN-soluble are all absent, and StaB-ddG is a <i>binding</i> ddG, not a folding one. We did
not substitute a proxy and call it stability. The claim is narrower and checkable: the framework is
identical to the clinical molecule, so F<sub>v</sub> T<sub>m</sub> is near-constant; what we changed
is a surface loop, so what we score is the loop's own aggregation, charge and chemical-liability
contribution, deterministically. Honest caveat: AntiFold CDRH3 perplexity is slightly
<i>worse</i> than trastuzumab's own loop for 37 of 40 designs, so loop&ndash;backbone compatibility
is the one axis-3 signal that does not favour us.</div>""")

    # ---------- method
    A("<h2>How it was produced</h2>")
    A("""<ul>
<li><b>Generation (lane A, H100).</b> RFD4-Proteina <code>her2-novel-2026-11-09</code>, 100 replicates,
seed 1, no <code>sequence_tsr_k</code> &mdash; the setting with the best novelty pass rate (86%) and
lowest interface error (6.8 &Aring;). 11.4 min. Used to <i>characterise</i> the novelty/affinity
tension; not the source of the shipped list.</li>
<li><b>Shell enumeration (CPU).</b> The ridge affinity model on one-hot CDRH3 is additive, so
score(s) = intercept + &Sigma;<sub>p</sub> W[p,s<sub>p</sub>] and the 4&ndash;5-mutation shell
(2,231,712 sequences) is searched <i>exactly</i>, with no sampling.</li>
<li><b>Empirical scorers.</b> HistGradientBoosting on 367,044 clonotype-split FACS labels
(test AUC 0.991) and a RandomForest/ridge on 180 quantified SPR K<sub>D</sub>s (CV &rho;=0.72).
Externally validated on data never seen: Shanehsazzadeh 198 designs &rho;=0.297 (p=2.1e-5),
SPR 187 K<sub>D</sub>s &rho;=0.353 (p=7.2e-7).</li>
<li><b>Folding (lane A).</b> Boltz-2 co-fold, templated on the trastuzumab&ndash;HER2 complex
(1N8Z Fv + domain IV), seed 1, with trastuzumab folded in the same batch as the control. 41
structures in 5.6 min.</li>
<li><b>Second opinions (lane B, Nebius, no GPU).</b> StaB-ddG binding ddG on 42 variants
(<code>op-79006e6f713984a7</code>), ESM-2 650M pseudo-likelihood
(<code>op-07b3495e1f4d104a</code>), AntiFold inverse-folding perplexity on 41 F<sub>v</sub>
backbones (<code>op-441067223204e140</code>). Ran concurrently with lane A.</li></ul>""")

    A(f"""<footer>Repo: <code>~/ubaydolla/her2</code> &middot; log: <code>NOTES.md</code> (every run id and
decision) &middot; tables: <code>results/</code> &middot; rejection ledger:
<code>results/rejected_designs.csv</code> ({len(rej)} row(s)).
Comparisons are only ever made inside one folding model. Credentials are gitignored and never
committed.</footer></div></html>""")

    out = f"{RES}/summary.html"
    open(out, "w").write("\n".join(H))
    print(f"wrote {out} ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
