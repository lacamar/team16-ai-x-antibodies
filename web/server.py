import csv, io, json, re, subprocess, sys, tarfile, tempfile, time, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import novelty_filter, affinity_surrogate, thermostability  # noqa: E402
import gemmi  # noqa: E402

SSH = ["ssh", "-i", str(Path.home() / ".ssh/team-16"), "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=20", "hackathon@195.242.31.250"]
RUNS = "~/lachlan/rfd4-docs/runs"
TTL = 15
DATA = ROOT / "data" / "her2"
TZ = novelty_filter.TRASTUZUMAB_CDRH3

SUMMARY = {
    "surrogate": {"spearman": 0.72, "auc": 0.90},
    "iptm_vs_spr_affinity": 0.285,
    "epitope": {"residues": 19, "hotspots": 4},
    "modules": ["novelty_filter", "affinity_surrogate", "thermostability", "epitope_check", "reward"],
    "tracks": {"design": "RFdiffusion/RFD4 CDRH3 generation + Boltz2 refold", "scoring": "affinity surrogate + novelty + stability"},
}

_cache, _lock = {}, threading.Lock()


def cached(key, fn, ttl=TTL):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (time.time(), val)
    return val


def ssh(cmd, binary=False, timeout=120):
    r = subprocess.run(SSH + [cmd], capture_output=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(r.stderr.decode()[-300:])
    return r.stdout if binary else r.stdout.decode()


def status():
    out = ssh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits;"
              "echo ---;nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits;"
              "echo ---;df -B1 --output=avail,size ~ | tail -1")
    g, p, d = out.split("---\n")
    gpus = []
    for l in g.strip().splitlines():
        n, u, mu, mt = [x.strip() for x in l.split(",")]
        gpus.append({"name": n, "util_pct": float(u), "mem_used_mib": float(mu), "mem_total_mib": float(mt)})
    procs = []
    for l in p.strip().splitlines():
        pid, name, mem = [x.strip() for x in l.split(",")]
        procs.append({"pid": int(pid), "name": name, "mem_mib": float(mem)})
    free, size = map(int, d.split())
    return {"gpus": gpus, "procs": procs, "disk_free_gb": round(free / 1e9, 1),
            "disk_total_gb": round(size / 1e9, 1), "ts": time.time()}


def runs():
    out = ssh(f"cd {RUNS}; for r in *; do echo $r $(ls $r/design/generation/inference_samples 2>/dev/null | wc -l)"
              f" $(test -f $r/scored-boltz2/refold_metrics.csv && echo 1 || echo 0)"
              f" $(test -f $r/scored-boltz2/refold_metrics.csv && tail -n +2 $r/scored-boltz2/refold_metrics.csv | wc -l || echo 0); done")
    res = []
    for l in out.split("\n"):
        if l.strip():
            n, d, s, f = l.split()
            res.append({"name": n, "designs": int(d), "scored": s == "1", "folded": int(f)})
    return {"runs": res, "ts": time.time()}


CDR = re.compile(r"C.[RK](.+?)WG.G")


def cdrh3_from_seq(vh):
    m = re.search(r"C.[RK](.+?)YWG.G", vh) or CDR.search(vh)
    return m.group(1) if m else None


def cif_cdrh3s(run):
    # ponytail: re-pulls all CIFs per run on cache miss (30 min TTL)
    raw = ssh(f"cd {RUNS}/{run}/design/generation/inference_samples && tar c */*.cif 2>/dev/null", binary=True, timeout=300)
    rows = []
    with tempfile.TemporaryDirectory() as td, tarfile.open(fileobj=io.BytesIO(raw)) as t:
        t.extractall(td, filter="data")
        for f in sorted(Path(td).glob("*/*.cif")):
            try:
                ch = gemmi.read_structure(str(f))[0]["B"]
                seq = gemmi.one_letter_code([r.name for r in ch])
            except Exception:
                continue
            c = cdrh3_from_seq(seq)
            if c:
                rows.append({"cdrh3": c, "id": f.parent.name})
    return rows


def design_rows(run):
    if not re.fullmatch(r"[\w.-]+", run):
        raise ValueError("bad run")
    try:
        txt = ssh(f"cat {RUNS}/{run}/scored-boltz2/refold_metrics.csv")
        rows = []
        for r in csv.DictReader(io.StringIO(txt)):
            c = cdrh3_from_seq(r["binder_sequence"])
            if c:
                rows.append({"cdrh3": c, "id": r["query_id"], "iptm": float(r["iptm"] or "nan"),
                             "min_ipae": float(r["min_ipae"] or "nan")})
        src = "refold_metrics.csv"
    except RuntimeError:
        rows, src = cached(("cif", run), lambda: cif_cdrh3s(run), 1800), "cif"
    out = []
    for r in rows:
        c = r["cdrh3"]
        nov, why = novelty_filter.is_novel(c, BY_LEN)
        s = SURR.score(c)
        out.append({**r, "novel": nov, "novel_note": why, "surrogate": s["surrogate"], "p_high": s["p_high"],
                    "ood": s["ood"], "stability": round(thermostability.stability_score(c), 3),
                    "baseline": c == TZ,
                    **{k: (None if v != v else v) for k, v in r.items() if k in ("iptm", "min_ipae")}})
    out.sort(key=lambda x: -x["surrogate"])
    return {"run": run, "source": src, "designs": out, "ts": time.time()}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/summary":
                body = SUMMARY
            elif u.path == "/api/status":
                body = cached("status", status)
            elif u.path == "/api/runs":
                body = cached("runs", runs)
            elif u.path == "/api/designs":
                run = parse_qs(u.query).get("run", [""])[0]
                body = cached(("designs", run), lambda: design_rows(run))
            else:
                return self.reply(404, {"error": "not found"})
            self.reply(200, body)
        except Exception as e:
            self.reply(502, {"error": str(e)})

    def reply(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print("fitting surrogate...", flush=True)
    SURR = affinity_surrogate.AffinitySurrogate().fit()
    BY_LEN = novelty_filter.load_provided_cdrh3(DATA / "tz_her2_facs_3classes_524346.csv.zip",
                                                DATA / "trastuzumab_her2_spr_all_768.csv")
    print(f"serving on :{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
