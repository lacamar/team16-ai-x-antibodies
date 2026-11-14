import json,re
h=open('dashboard.html').read()
d=json.load(open('designs.json'))
a=h.index('const CIF = `'); b=h.index('\n`;',a)+3
data='const TRAST = '+json.dumps(d['trastuzumab'])+';\nconst DESIGNS = '+json.dumps(d['designs']).replace('</','<\\/')+';'
h=h[:a]+data+h[b:]
i=h.index('(function(){\n var v=$3Dmol'); j=h.index('})();',i)+5
js='''(function(){
 var v=$3Dmol.createViewer("viewer",{backgroundColor:"#0f172a"}),sel=document.getElementById("dsel"),sc=document.getElementById("scores"),cap=document.getElementById("capt");
 DESIGNS.forEach(function(d,i){var o=document.createElement("option");o.value=i;o.textContent=d.name+" \\u2014 "+d.cdrh3;sel.appendChild(o)});
 function cls(x,g,m){return x>=g?"good":x>=m?"mid":"bad"}
 function showDesign(i){
  var d=DESIGNS[i];
  v.removeAllModels();v.removeAllLabels();
  v.addModel(d.cif,"mmcif");
  v.setStyle({},{cartoon:{color:"#888"}});
  v.setStyle({chain:"A"},{cartoon:{color:"#60a5fa"}});
  v.setStyle({chain:"B"},{cartoon:{color:"#a78bfa"}});
  v.setStyle({chain:["C","D"]},{cartoon:{color:"#e07a1f",opacity:.9}});
  [["C",572],["C",573],["D",591],["D",593]].forEach(function(h){
   v.addStyle({chain:h[0],resi:h[1]},{stick:{color:"#ef4444",radius:.25},sphere:{color:"#ef4444",scale:.35}});
   v.addLabel(h[0]+h[1],{fontSize:11,backgroundOpacity:.5},{chain:h[0],resi:h[1],atom:"CA"});
  });
  v.zoomTo();v.render();
  cap.textContent="Selected: "+d.name+" ("+d.track+") \\u00b7 CDRH3 "+d.cdrh3+" docked to HER2";
  sc.innerHTML='<span><small>CDRH3</small><code>'+d.cdrh3+'</code></span>'+
   '<span><small>surrogate</small><b class="'+cls(d.surrogate,0.5,0)+'">'+d.surrogate.toFixed(2)+'</b></span>'+
   '<span><small>p_high</small><b class="'+cls(d.p_high,0.9,0.7)+'">'+d.p_high.toFixed(2)+'</b></span>'+
   '<span><small>stability</small><b>'+d.stability.toFixed(2)+'</b></span>'+
   '<span><small>muts vs trastuzumab ('+TRAST+')</small><b>'+d.muts+'</b></span>'+
   '<span class="chip '+(d.novel?"nov":"known")+'">'+(d.novel?"novel":"known")+'</span>';
 }
 sel.onchange=function(){showDesign(+sel.value)};
 showDesign(0);
 window.addEventListener("resize",function(){v.resize();v.render()});
})();'''
h=h[:i]+js+h[j:]
h=h.replace('<div class="vwrap"><div id="viewer"></div>','<div class="vwrap"><select id="dsel" aria-label="Design"></select><div id="viewer"></div>',1)
h=h.replace('<span><b>Designed binder docked to HER2 · ipTM ~0.95</b></span>','<span><b id="capt">Designed binder docked to HER2</b></span>',1)
h=h.replace('Drag to rotate · scroll to zoom</span></div></div>','Drag to rotate · scroll to zoom</span></div><div id="scores"></div></div>',1)
css='''#dsel{margin:10px 14px;padding:8px 10px;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;font:inherit;font-size:14px;max-width:calc(100% - 28px)}
#scores{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:flex-end;padding:10px 14px 14px;border-top:1px solid var(--line);color:#cbd5e1;font-size:13px}
#scores span{display:flex;flex-direction:column}#scores small{color:#94a3b8;font-size:11px}#scores b,#scores code{font-size:16px;color:#e2e8f0}
#scores .good{color:#4ade80}#scores .mid{color:#fbbf24}#scores .bad{color:#f87171}
#scores .chip{flex-direction:row;padding:3px 10px;border-radius:999px;font-weight:600}.chip.nov{background:#166534;color:#dcfce7}.chip.known{background:#475569;color:#e2e8f0}
</style>'''
h=h.replace('</style>',css,1)
open('dashboard.html','w').write(h)
s=h[h.index('<script>\n')+9:h.index('</script>',h.index('<script>\n'))]
open('chk.js','w').write(s)
