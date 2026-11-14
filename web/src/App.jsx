import { useEffect, useState, useCallback } from 'react'

const POLL = 15000

function useApi(url, poll) {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)
  const [at, setAt] = useState(0)
  const load = useCallback(() => {
    if (!url) return
    fetch(url).then(r => r.json().then(j => { if (!r.ok) throw new Error(j.error); return j }))
      .then(j => { setData(j); setErr(null); setAt(Date.now()) })
      .catch(e => setErr(String(e.message || e)))
  }, [url])
  useEffect(() => {
    setData(null); load()
    if (!poll) return
    const t = setInterval(load, POLL)
    return () => clearInterval(t)
  }, [load, poll])
  return { data, err, at }
}

function useAgo(at) {
  const [, tick] = useState(0)
  useEffect(() => { const t = setInterval(() => tick(x => x + 1), 1000); return () => clearInterval(t) }, [])
  return at ? Math.round((Date.now() - at) / 1000) : null
}

function Live({ at, err }) {
  const s = useAgo(at)
  return <span className="mut"><span className={'dot' + (err ? ' off' : '')} />{err ? 'offline' : s == null ? 'loading' : `updated ${s}s ago`}</span>
}

function Summary() {
  const { data: d, err } = useApi('/api/summary')
  if (err) return <p className="err">{err}</p>
  if (!d) return null
  return (
    <div className="grid" style={{ marginBottom: 16 }}>
      <div className="tile lead">
        <b>{d.iptm_vs_spr_affinity} vs {d.surrogate.spearman}</b>
        <span>ipTM-vs-SPR affinity correlation (structure confidence barely predicts affinity) vs our sequence surrogate Spearman. Surrogate AUC {d.surrogate.auc}.</span>
      </div>
      <div className="tile"><b>{d.epitope.residues} / {d.epitope.hotspots}</b><span>epitope residues / hotspots</span></div>
      <div className="tile"><b>{d.modules.length}</b><span>{d.modules.join(', ')}</span></div>
      <div className="tile"><b>2 tracks</b><span>{d.tracks.design}; {d.tracks.scoring}</span></div>
    </div>
  )
}

function VMStatus() {
  const { data: d, err, at } = useApi('/api/status', true)
  return (
    <div className="card">
      <h2>VM status <Live at={at} err={err} /></h2>
      {err && <p className="err">{err}</p>}
      {d && <>
        {d.gpus.map((g, i) => <div key={i}>
          <div>{g.name}: {g.util_pct}% util</div>
          <div className="bar"><i style={{ width: g.util_pct + '%' }} /></div>
          <div>VRAM {(g.mem_used_mib / 1024).toFixed(1)} / {(g.mem_total_mib / 1024).toFixed(1)} GiB</div>
          <div className="bar"><i style={{ width: 100 * g.mem_used_mib / g.mem_total_mib + '%' }} /></div>
        </div>)}
        <div>Disk free {d.disk_free_gb} / {d.disk_total_gb} GB</div>
        <h2 style={{ marginTop: 10 }}>GPU jobs ({d.procs.length})</h2>
        {d.procs.map(p => <div key={p.pid} className="mut">{p.pid} {p.name} {p.mem_mib} MiB</div>)}
      </>}
    </div>
  )
}

function RunsPanel({ sel, setSel }) {
  const { data: d, err, at } = useApi('/api/runs', true)
  return (
    <div className="card">
      <h2>Runs <Live at={at} err={err} /></h2>
      {err && <p className="err">{err}</p>}
      {d && d.runs.map(r => (
        <div key={r.name} onClick={() => setSel(r.name)} style={{ cursor: 'pointer', fontWeight: sel === r.name ? 700 : 400 }}>
          {sel === r.name ? '> ' : ''}{r.name}
          <div className="mut">designed {r.designs} | folded {r.folded} | scored {r.scored ? 'yes' : 'no'}</div>
          <div className="bar"><i style={{ width: r.designs ? 100 * r.folded / r.designs + '%' : 0 }} /></div>
        </div>
      ))}
    </div>
  )
}

const tone = (v, g, a, inv) => v == null ? '' : (inv ? v <= g : v >= g) ? 'good' : (inv ? v <= a : v >= a) ? 'mid' : 'bad'
const COLS = [
  ['cdrh3', 'CDRH3'], ['novel', 'Novel'], ['surrogate', 'Surrogate'], ['p_high', 'p_high'],
  ['iptm', 'ipTM'], ['min_ipae', 'min iPAE'], ['stability', 'Stability'],
]

function DesignsTable({ run }) {
  const { data: d, err, at } = useApi(run ? `/api/designs?run=${run}` : null, true)
  const [key, setKey] = useState('surrogate')
  const [asc, setAsc] = useState(false)
  if (!run) return <div className="card">Select a run.</div>
  const rows = d ? [...d.designs].sort((a, b) => {
    const x = a[key], y = b[key]
    if (x == null) return 1
    if (y == null) return -1
    return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1)
  }) : []
  const click = k => { k === key ? setAsc(!asc) : (setKey(k), setAsc(false)) }
  const chip = (v, c) => v == null ? '-' : <span className={'chip ' + c}>{typeof v === 'number' ? v.toFixed(3) : v}</span>
  return (
    <div className="card">
      <h2>Designs: {run} ({rows.length}) <Live at={at} err={err} /></h2>
      {err && <p className="err">{err}</p>}
      {!d && !err && <p className="mut">Loading (first load pulls CIFs from the VM)...</p>}
      <div className="tw"><table>
        <thead><tr>{COLS.map(([k, l]) => <th key={k} onClick={() => click(k)}>{l}{k === key ? (asc ? ' ▲' : ' ▼') : ''}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={r.id + r.cdrh3 + i} className={r.baseline ? 'base' : ''}>
            <td className="mono">{r.cdrh3}{r.baseline && ' (trastuzumab)'}{r.ood && ' *'}</td>
            <td title={r.novel_note}>{chip(r.novel ? 'novel' : 'known', r.novel ? 'good' : 'bad')}</td>
            <td>{chip(r.surrogate, tone(r.surrogate, 3, 2))}</td>
            <td>{chip(r.p_high, tone(r.p_high, 0.5, 0.2))}</td>
            <td>{chip(r.iptm, tone(r.iptm, 0.9, 0.7))}</td>
            <td>{chip(r.min_ipae, tone(r.min_ipae, 0.5, 1, true))}</td>
            <td>{chip(r.stability, tone(r.stability, 0.8, 0.5))}</td>
          </tr>))}</tbody>
      </table></div>
      <p className="mut">* out-of-distribution length (surrogate trained on 9-10-mers). Hover novelty chip for reason.</p>
    </div>
  )
}

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'))
  const [sel, setSel] = useState('opt-k1')
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('theme', theme) }, [theme])
  return (
    <div className="wrap">
      <header>
        <h1>Team 16: AI x Antibodies (HER2 CDRH3)</h1>
        <button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? 'Light' : 'Dark'}</button>
      </header>
      <Summary />
      <div className="two"><VMStatus /><RunsPanel sel={sel} setSel={setSel} /></div>
      <DesignsTable run={sel} />
    </div>
  )
}
