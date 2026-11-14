import json
h=open('dashboard.html').read()
d=json.load(open('designs.json'))
a=h.index('const DESIGNS = '); b=h.index('\n',a)
h=h[:a]+'const DESIGNS = '+json.dumps(d['designs']).replace('</','<\\/')+';'+h[b:]
def r(o,n):
    global h
    assert h.count(o)==1,o
    h=h.replace(o,n)
r('v.addModel(d.cif,"mmcif");','v.addModel(d.cif,d.format||"mmcif");')
r('''  v.setStyle({},{cartoon:{color:"#888"}});
  v.setStyle({chain:"A"}''','''  v.setStyle({},{cartoon:{color:"#888"}});
  if(d.vh_only)v.setStyle({},{cartoon:{color:"#a78bfa"}});
  else{
  v.setStyle({chain:"A"}''')
r('''atom:"CA"});
  });
  v.zoomTo()''','''atom:"CA"});
  });
  }
  v.zoomTo()''')
r("""'<span><small>stability</small>""","""(d.ptm!=null?'<span><small>pTM</small><b>'+d.ptm.toFixed(2)+'</b></span>':'')+
   '<span><small>stability</small>""")
r('" docked to HER2"','(d.vh_only?" (VH only, Boltz-2 fold)":" docked to HER2")')
open('dashboard.html','w').write(h)
s=h[h.index('<script>\n')+9:h.index('</script>',h.index('<script>\n'))]
open('chk.js','w').write(s)
