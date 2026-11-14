#!/usr/bin/env python3
"""Client for the Nebius Scientific AI Platform. Python standard library only.

Reads your team credentials file. Secret values are never printed: every
output is masked before it reaches the screen or a file.

  forge.py models                      list the models your key may use
  forge.py schema MODEL                read one model schema (short by default)
  forge.py run MODEL --args JSON       run an inline model
  forge.py submit MODEL ...            run a file-based model
  forge.py status OP                   state of one run
  forge.py wait OP                     poll until the run is done
  forge.py fetch OP --dest DIR         download the results
  forge.py resume OP                   restart a run that hit a platform error
  forge.py runs                        list your runs recorded on this machine

Credentials are found in this order: --creds, $NEBIUS_CREDS,
./credentials.private.json, ~/credentials.private.json
"""
import argparse
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

TERMINAL = {"succeeded", "failed", "cancelled", "preempted", "expired"}
MANIFEST_SCHEMA = "fs2-serve.nebius.ai/scientific-artifact-manifest/v1"
REQUEST_SCHEMA = "fs2-serve.nebius.ai/scientific-run-request/v1"
JSON_MEDIA = {"application/json", "application/vnd.fs2.scientific-manifest+json"}
RUNS_DIR = os.environ.get("NEBIUS_RUNS_DIR", "nebius-runs")

# Models confirmed working on 12 November 2026. lane, MCP tool, operation,
# typical seconds for a first call, note. Times were measured with about 25 jobs
# open on one account, so they are closer to a busy hackathon than to an idle
# platform. The catalog holds more models; these are the tested ones.
CATALOG = {
    # inline: one call with arguments
    "boltz2": ("inline", "boltz2_predict_native", None, 80,
               "structure of one or more chains, needs an A3M alignment per chain"),
    "openfold3": ("inline", "infer_openfold3_native", None, 165,
                  "structure or complex from sequences"),
    "openfold2": ("inline", "infer_openfold2_native", None, 295,
                  "single chain structure, UNRELIABLE 12 Nov: endpoint fails to start"),
    "proteinmpnn": ("inline", "infer_proteinmpnn_native", None, 100,
                    "sequences for a given backbone (inverse folding)"),
    "msa-search-pdb70": ("inline", "msa_search_native", None, 40,
                         "alignment search against PDB70"),
    "diffdock": ("inline", "infer_diffdock_native", None, 320,
                 "dock a small molecule into a protein"),
    "mdprecheck": ("inline", "run_mdprecheck", None, 10,
                   "check molecular dynamics inputs before a run"),
    # antibody models, published 12 November
    "nanobodybuilder2": ("file", "submit_nanobodybuilder2", "predict", 130,
                         "nanobody structure from one VHH in a FASTA with header H"),
    "abodybuilder3": ("file", "submit_abodybuilder3", "predict", 120,
                      "paired antibody structure from a FASTA with headers H and L"),
    "antifold": ("file", "submit_antifold", "inverse-fold", 110,
                 "antibody CDR redesign, tar holding the PDB"),
    "esm-if1": ("file", "submit_esm_if1", "inverse-fold", 270,
                "inverse folding, same tar, custom_chain_mode"),
    "igbert": ("file", "submit_igbert", "embed", 260,
               "antibody heavy and light embeddings from a CSV"),
    "chai1": ("file", "submit_chai1", "predict", 270,
              "complex structure from a Chai FASTA in a tar"),
    # protein language models, published 12 November
    "esmc-300m": ("file", "submit_esmc_300m", "embed", 190,
                  "embeddings or variant scores from a FASTA in a tar"),
    "esmc-600m": ("file", "submit_esmc_600m", "embed", 230,
                  "embeddings or variant scores from a FASTA in a tar"),
    "esmc-6b": ("file", "submit_esmc_6b", "embed", 920,
                "embeddings from a FASTA in a tar, the big one"),
    "esm2-650m": ("file", "submit_esm2_650m", "score-sequences", 230,
                  "sequence scores from a FASTA in a tar"),
    "esm2-3b": ("file", "submit_esm2_3b", "score-sequences", 450,
                "sequence scores from a FASTA in a tar"),
    "saprot-650m": ("file", "submit_saprot_650m", "embed", 270,
                    "structure aware embeddings, tar with the PDB"),
    "saprot-1-3b": ("file", "submit_saprot_1_3b", "embed", 620,
                    "structure aware embeddings, bigger and slower"),
    "saprot-inverse-folding": ("file", "submit_saprot_inverse_folding", "inverse-fold", 560,
                               "structure aware sequence design, mask_positions picks the sites"),
    "stab-ddg": ("file", "submit_stab_ddg", "predict-binding-ddg", 730,
                 "binding ddG for named point mutations across two partner chains"),
    # file based: upload, manifest, submit, poll, download
    "rfd4-proteina": ("file", "submit_rfd4_proteina", "generate", 290,
                      "binder design guided to a site, input annotated-cif"),
    "rfdiffusion": ("file", "submit_rfdiffusion", "design-backbone", 330,
                    "backbone design, input design_constraint text"),
    "mosaic": ("file", "submit_mosaic", "design-binder", 1310,
               "binder design from a target FASTA, slow"),
    "esmfold2-fast": ("file", "submit_esmfold2_fast", "predict-protein-structure", 670,
                      "structure from sequence, long queue"),
    "esmfold2": ("file", "submit_esmfold2", "predict-structure", 1330,
                 "structure from sequence, long queue"),
    "protenix-v2": ("file", "submit_protenix_v2", "predict-complex-structure", 1220,
                    "complex structure, long queue"),
    "bioemu2": ("file", "submit_bioemu2", "sample-equilibrium", 225,
                "conformations of one protein"),
    "gromacs": ("file", "submit_gromacs_workflow", "run-workflow", 280,
                "molecular dynamics, request wrapper, send the bundle still gzipped"),
    "amber": ("file", "submit_amber_workflow", "run-workflow", 170,
              "molecular dynamics, request wrapper, send the bundle still gzipped"),
    "namd": ("file", "submit_namd_workflow", "run-workflow", 160,
             "molecular dynamics, request wrapper, send the bundle still gzipped"),
    "lammps": ("file", "submit_lammps_workflow", "run-workflow", 300,
               "molecular dynamics, request wrapper, send the bundle still gzipped"),
    "openmm": ("file", "submit_openmm_workflow", "run-workflow", 140,
               "molecular dynamics, run fields go inside request"),
}

# These answer in the same call and refuse an idempotency key or a wait handle.
SYNC_ONLY = {"mdprecheck"}

MEDIA_BY_EXT = {
    ".cif": "chemical/x-cif", ".mmcif": "chemical/x-cif", ".pdb": "chemical/x-pdb",
    ".fasta": "text/x-fasta", ".fa": "text/x-fasta", ".txt": "text/plain",
    ".a3m": "text/plain", ".csv": "text/csv", ".yaml": "application/yaml",
    ".json": "application/json", ".tar": "application/x-tar",
    ".gz": "application/gzip", ".zst": "application/zstd",
}

_CA = "/etc/ssl/cert.pem"
CTX = ssl.create_default_context(cafile=_CA if os.path.exists(_CA) else None)


# ---------------------------------------------------------------- credentials

def find_creds(explicit):
    names = [explicit, os.environ.get("NEBIUS_CREDS"), "credentials.private.json",
             os.path.expanduser("~/credentials.private.json")]
    for n in names:
        if n and os.path.exists(n):
            return n
    die("No credentials file. Pass --creds PATH or set NEBIUS_CREDS.")


def die(msg, code=2):
    print("error: " + msg, file=sys.stderr)
    raise SystemExit(code)


class Creds:
    def __init__(self, path):
        with open(path) as fh:
            doc = json.load(fh)
        api = doc.get("scientific_api") or {}
        self.base = (api.get("base_url") or "").rstrip("/")
        self.mcp = (api.get("mcp_url") or "").rstrip("/")
        self.key = api.get("api_key") or ""
        if not (self.base and self.mcp and self.key):
            die("credentials file has no scientific_api base_url, mcp_url and api_key")
        self.secrets = [v for v in (
            self.key,
            (doc.get("login") or {}).get("password"),
            (doc.get("private_storage") or {}).get("access_key_id"),
            (doc.get("private_storage") or {}).get("secret_access_key"),
        ) if isinstance(v, str) and len(v) > 7]

    def mask(self, text):
        for s in self.secrets:
            text = text.replace(s, "<hidden>")
        return text


# ------------------------------------------------------------------ transport

def http(url, method="GET", headers=None, body=None, timeout=180):
    req = urllib.request.Request(url, method=method, headers=headers or {}, data=body)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()
    except urllib.error.URLError as e:
        die("cannot reach %s (%s). Check your network and the URL." % (url, e.reason))


def parse_sse(text):
    out = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        data = [ln[5:].lstrip() for ln in block.split("\n") if ln.startswith("data:")]
        if data:
            try:
                out.append(json.loads("\n".join(data)))
            except json.JSONDecodeError:
                pass
    return out


class Api:
    """REST for uploads, operations and downloads. MCP for model calls."""

    def __init__(self, creds):
        self.c = creds
        self._mcp_session = None

    # ---- REST
    def rest(self, method, path, body=None, raw=None, headers=None, timeout=180):
        h = {"Authorization": "Bearer " + self.c.key, "Accept": "application/json"}
        data = raw
        if body is not None:
            h["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        h.update(headers or {})
        url = path if path.startswith("http") else self.c.base + path
        status, hdrs, out = http(url, method, h, data, timeout)
        return status, hdrs, out

    def rest_json(self, method, path, **kw):
        status, _, out = self.rest(method, path, **kw)
        text = out.decode("utf-8", "replace")
        try:
            doc = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            doc = {"_raw": text[:1000]}
        return status, doc

    # ---- MCP
    def _mcp_post(self, payload, extra=None, timeout=300):
        h = {"Authorization": "Bearer " + self.c.key, "Content-Type": "application/json",
             "Accept": "application/json, text/event-stream"}
        h.update(extra or {})
        status, hdrs, body = http(self.c.mcp, "POST", h, json.dumps(payload).encode(), timeout)
        text = body.decode("utf-8", "replace")
        if "text/event-stream" in hdrs.get("content-type", ""):
            msgs = parse_sse(text)
            doc = msgs[-1] if msgs else None
        else:
            try:
                doc = json.loads(text) if text.strip() else None
            except json.JSONDecodeError:
                doc = {"_raw": text[:800]}
        return status, hdrs, doc

    def _mcp_init(self):
        if self._mcp_session is not None:
            return self._mcp_session
        status, hdrs, init = self._mcp_post({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "oslo-participant", "version": "1.0"}}})
        if status == 403:
            die("MCP returned 403. Use the URL exactly, with no slash at the end.")
        extra = {}
        if hdrs.get("mcp-session-id"):
            extra["Mcp-Session-Id"] = hdrs["mcp-session-id"]
        proto = ((init or {}).get("result") or {}).get("protocolVersion")
        if proto:
            extra["MCP-Protocol-Version"] = proto
        self._mcp_post({"jsonrpc": "2.0", "method": "notifications/initialized"}, extra)
        self._mcp_session = extra
        return extra

    def call(self, tool, arguments, timeout=300):
        """One MCP tool call. Returns (ok, payload). Checks the inner error too."""
        extra = self._mcp_init()
        status, _, doc = self._mcp_post({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments}}, extra, timeout)
        if not isinstance(doc, dict):
            return False, {"message": "unreadable answer", "_raw": str(doc)[:400]}
        if "error" in doc:
            return False, doc["error"]
        res = doc.get("result") or {}
        payload = res.get("structuredContent")
        if payload is None:
            for part in res.get("content") or []:
                if part.get("type") == "text":
                    try:
                        payload = json.loads(part["text"])
                    except (json.JSONDecodeError, TypeError):
                        payload = {"_text": str(part.get("text"))[:1500]}
                    break
        payload = payload if payload is not None else {}
        # the gateway can answer HTTP 200, isError false, and still carry an
        # error object inside the payload
        inner = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(inner, dict):
            return False, inner
        return not res.get("isError"), payload

    # ---- uploads
    def upload(self, model, data, media_type, key, compression="none"):
        meta = {"model_id": model, "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data), "media_type": media_type,
                "compression": compression}
        status, begun = self.rest_json("POST", "/scientific-artifacts/uploads",
                                       body=meta, headers={"Idempotency-Key": key})
        if status >= 300:
            die("upload refused (HTTP %s): %s" % (status, self.c.mask(json.dumps(begun))[:400]))
        upload_id, op = begun["upload_id"], begun.get("operation_id")
        q = ("?operation_id=" + op) if op else ""
        status, _, out = self.rest("PUT", "/scientific-artifacts/uploads/%s/content%s" % (upload_id, q),
                                   raw=data, headers={"Content-Type": media_type}, timeout=600)
        if status >= 300 and status != 412:
            die("upload of the file failed (HTTP %s)" % status)
        status, art = self.rest_json("POST", "/scientific-artifacts/uploads/%s:finalize" % upload_id,
                                     body={"operation_id": op} if op else {})
        if status >= 300:
            die("upload could not be finished (HTTP %s): %s"
                % (status, self.c.mask(json.dumps(art))[:400]))
        art = art.get("artifact", art)
        ref = {k: art[k] for k in ("artifact_id", "sha256", "size_bytes", "media_type") if k in art}
        ref["compression"] = art.get("compression", compression)
        return ref


# --------------------------------------------------------------------- helpers

def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def idem_key(model, payload, label=None):
    """Same inputs give the same key, so a repeat resumes instead of starting again."""
    if label:
        return "oslo-%s-%s" % (model, label)
    h = hashlib.sha256(canonical(payload)).hexdigest()[:16]
    return "oslo-%s-%s" % (model, h)


def receipt_path(label):
    return os.path.join(RUNS_DIR, label + ".json")


def save_receipt(rec):
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = receipt_path(rec["label"])
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, path)
    return path


def load_receipt_by_op(op):
    if not os.path.isdir(RUNS_DIR):
        return None
    for name in os.listdir(RUNS_DIR):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(RUNS_DIR, name)) as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("operation_id") == op:
            return rec
    return None


def op_state(doc):
    o = doc.get("operation") if isinstance(doc.get("operation"), dict) else doc
    state = str(o.get("status") or o.get("state") or "unknown").lower()
    ready = bool(o.get("result_available") or doc.get("result_available"))
    return state, ready


def find_operation_id(payload):
    if not isinstance(payload, dict):
        return None
    o = payload.get("operation") if isinstance(payload.get("operation"), dict) else payload
    for k in ("id", "operation_id"):
        v = o.get(k) or payload.get(k)
        if isinstance(v, str) and v.startswith("op-"):
            return v
    return None


def media_for(path):
    return MEDIA_BY_EXT.get(os.path.splitext(path)[1].lower(), "application/octet-stream")


def wait_hint(model):
    secs = (CATALOG.get(model) or (None, None, None, 300, ""))[3]
    if secs <= 120:
        return "Usually about %d seconds on the first call." % secs
    return ("Usually about %d minutes on the first call, and it can be longer when "
            "many teams submit at once." % round(secs / 60.0))


def show(creds, obj, limit=4000):
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=1, ensure_ascii=False)
    text = creds.mask(text)
    if limit and len(text) > limit:
        text = text[:limit] + "\n...[cut, %d characters in total]" % len(text)
    print(text)


def explain(err):
    """Turn a platform error into one line a person can act on."""
    code = (err or {}).get("code")
    data = (err or {}).get("data") or {}
    code = data.get("code") or code
    msg = str((err or {}).get("message") or data.get("message") or "")
    if "not granted to tenant" in msg:
        return "This model's weights are not open for the event account. Pick another model."
    if code == "MODEL_MAINTENANCE":
        return "The model is paused by the platform. Try another one."
    if "missing required fields: request" in msg:
        return "Put the run fields inside a 'request' object. See RECIPES.md."
    if "private storage is temporarily unavailable" in msg:
        return "Storage hiccup. Run: forge.py resume <operation id>. Do not submit again."
    if code == "MODEL_INPUT_INVALID":
        return "The input does not match the model schema. Run: forge.py schema <model>."
    return msg[:300] or "no message"


# -------------------------------------------------------------------- commands

def cmd_models(api, a):
    status, doc = api.rest_json("GET", "/me")
    if status >= 300:
        die("cannot read /v1/me (HTTP %s). Is the API key right?" % status)
    allowed = []
    for key in ("models", "authorized_models", "allowed_models", "apps"):
        v = doc.get(key)
        if isinstance(v, list):
            allowed = v
            break
    ids = []
    for m in allowed:
        ids.append(m if isinstance(m, str) else (m.get("model_id") or m.get("id") or ""))
    ids = sorted(i for i in ids if i)
    if a.all:
        print("%d models on your key:" % len(ids))
        for i in ids:
            mark = "tested" if i in CATALOG else ""
            print("  %-34s %s" % (i, mark))
        return
    print("Models with a tested input format (%d of %d on your key).\n"
          "Run forge.py models --all to see every name.\n" % (
              len([i for i in ids if i in CATALOG]) or len(CATALOG), len(ids) or len(CATALOG)))
    for name, (lane, tool, op, secs, note) in sorted(CATALOG.items(), key=lambda kv: (kv[1][0], kv[0])):
        if ids and name not in ids:
            continue
        print("  %-16s %-6s ~%-5s %s" % (name, lane, _mins(secs), note))


def _mins(secs):
    return "%ds" % secs if secs < 120 else "%dm" % round(secs / 60.0)


# Blocks that repeat other blocks or only say where the schema came from.
SCHEMA_DROP = {"mcp_input_schema", "tool_name_unavailable", "tool_name_provenance",
               "tool_name_source_refs", "source_refs", "source_schema_sha256",
               "runtime_identity", "runtime_binding", "regions", "model_ref"}


def cmd_schema(api, a):
    # summary_only leaves out the input contract for file-based models, so ask
    # for the whole thing and drop the noisy parts here instead
    ok, payload = api.call("get_model_schema", {"model_id": a.model})
    if not ok:
        die(explain(payload))
    if a.full or not isinstance(payload, dict):
        show(api.c, payload, limit=None)
        return
    drop = set(SCHEMA_DROP)
    if "input_schema" in payload or "request_schema" in payload:
        drop.add("contracts")  # same schema again, one level deeper
    trimmed = {k: v for k, v in payload.items() if k not in drop}
    show(api.c, trimmed, limit=40000)
    left = sorted(set(payload) - set(trimmed))
    if left:
        print("\n[left out, use --full to see: %s]" % ", ".join(left))


def resolve_locals(api, model, value, key):
    """Replace {"$text": path} with the file text and {"$upload": path} with an
    artifact reference, so arguments can point at files on this machine."""
    if isinstance(value, list):
        return [resolve_locals(api, model, v, key) for v in value]
    if not isinstance(value, dict):
        return value
    if "$text" in value:
        with open(value["$text"]) as fh:
            return fh.read()
    if "$upload" in value:
        path = value["$upload"]
        with open(path, "rb") as fh:
            data = fh.read()
        media = value.get("media_type") or media_for(path)
        ref = api.upload(model, data, media, key + "-in-" + hashlib.sha256(data).hexdigest()[:12])
        print("  uploaded %s (%d bytes, %s)" % (os.path.basename(path), len(data), media))
        return ref
    return {k: resolve_locals(api, model, v, key) for k, v in value.items()}


def cmd_run(api, a):
    entry = CATALOG.get(a.model)
    if entry and entry[0] != "inline":
        die("%s is a file-based model. Use: forge.py submit %s ..." % (a.model, a.model))
    tool = a.tool or (entry[1] if entry else None)
    if not tool:
        die("unknown model %s. Run forge.py models." % a.model)
    if a.args_file:
        with open(a.args_file) as fh:
            args = json.load(fh)
    else:
        args = json.loads(a.args)
    label = a.label or idem_key(a.model, args)[5:]
    key = idem_key(a.model, args, a.label)
    args = resolve_locals(api, a.model, args, key)
    if a.model not in SYNC_ONLY:
        args.setdefault("idempotency_key", key)
        args.setdefault("wait_seconds", 0)
    rec = {"label": label, "model": a.model, "tool": tool, "key": key,
           "kind": "inline", "submitted_at": time.time()}
    print("Starting %s. %s" % (a.model, wait_hint(a.model)))
    ok, payload = api.call(tool, args)
    if not ok:
        rec["state"] = "rejected"
        rec["error"] = api.c.mask(json.dumps(payload))[:800]
        save_receipt(rec)
        die(explain(payload))
    op = find_operation_id(payload)
    if not op:
        rec["state"] = "done"
        save_receipt(rec)
        print("Finished in the same call. Answer:")
        show(api.c, payload)
        return
    rec["operation_id"] = op
    rec["state"] = "queued"
    path = save_receipt(rec)
    print("Run id: %s  (saved in %s)" % (op, path))
    if a.no_wait:
        print("Check it later with: forge.py wait %s" % op)
        return
    _poll(api, op, a.budget, rec)


def cmd_submit(api, a):
    entry = CATALOG.get(a.model)
    if entry and entry[0] != "file":
        die("%s is an inline model. Use: forge.py run %s --args ..." % (a.model, a.model))
    tool = a.tool or (entry[1] if entry else None)
    operation = a.operation or (entry[2] if entry else None)
    if not (tool and operation):
        die("need --tool and --operation for %s" % a.model)
    if not a.entry:
        die("need at least one --entry name|semantic_type|path")
    params = json.loads(a.params)
    label = a.label or idem_key(a.model, [a.entry, params])[5:]
    key = idem_key(a.model, [a.entry, params], a.label)

    existing = None
    if os.path.exists(receipt_path(label)):
        with open(receipt_path(label)) as fh:
            existing = json.load(fh)
    if existing and existing.get("operation_id") and not a.force:
        print("This exact run already exists: %s" % existing["operation_id"])
        print("Following it instead of starting a new one.")
        return _poll(api, existing["operation_id"], a.budget, existing)

    entries = []
    for spec in a.entry:
        parts = spec.split("|")
        if len(parts) < 3:
            die("--entry needs name|semantic_type|path[|media_type]")
        name, semantic, path = parts[0], parts[1], parts[2]
        media = parts[3] if len(parts) > 3 else media_for(path)
        if not os.path.exists(path):
            die("file not found: %s" % path)
        with open(path, "rb") as fh:
            data = fh.read()
        # A .tar.gz stays gzipped: the entry carries the inner media type and
        # says compression is gzip. Some engines, NAMD among them, refuse a
        # bundle that was unpacked first.
        compression = "none"
        if path.endswith(".gz") or data[:2] == b"\x1f\x8b":
            compression = "gzip"
            if media == "application/gzip":
                media = media_for(path[:-3]) if path.endswith(".gz") else "application/x-tar"
        ref = api.upload(a.model, data, media, key + "-in-" + hashlib.sha256(data).hexdigest()[:12],
                         compression)
        print("  uploaded %s (%d bytes, %s)" % (os.path.basename(path), len(data), media))
        entries.append({"name": name, "semantic_type": semantic, "artifact": ref})

    manifest = {"schema": MANIFEST_SCHEMA, "manifest_id": key, "entries": entries}
    mref = api.upload(a.model, canonical(manifest),
                      "application/vnd.fs2.scientific-manifest+json", key + "-manifest")
    request = {"schema": REQUEST_SCHEMA, "operation": operation,
               "service_class": a.service_class, "input_manifest": mref,
               "parameters": params}
    if a.display_name:
        request["client_context"] = {"display_name": a.display_name}
    if a.wrap_request:
        # molecular dynamics tools want the run fields inside "request"
        call_args = {"request": request, "idempotency_key": key}
    else:
        call_args = dict(request)
        call_args["idempotency_key"] = key

    rec = {"label": label, "model": a.model, "tool": tool, "key": key,
           "kind": "file", "operation": operation, "submitted_at": time.time()}
    print("Submitting %s. %s" % (a.model, wait_hint(a.model)))
    ok, payload = api.call(tool, call_args)
    if not ok:
        rec["state"] = "rejected"
        rec["error"] = api.c.mask(json.dumps(payload))[:800]
        save_receipt(rec)
        die(explain(payload))
    op = find_operation_id(payload)
    if not op:
        rec["state"] = "done"
        save_receipt(rec)
        show(api.c, payload)
        return
    rec["operation_id"] = op
    rec["state"] = "queued"
    path = save_receipt(rec)
    print("Run id: %s  (saved in %s)" % (op, path))
    if a.no_wait:
        print("Check it later with: forge.py wait %s" % op)
        return
    _poll(api, op, a.budget, rec)


def _poll(api, op, budget, rec=None):
    t0 = time.time()
    last = None
    while True:
        status, doc = api.rest_json("GET", "/operations/" + op)
        if status >= 300:
            print("cannot read the run yet (HTTP %s), trying again" % status)
            time.sleep(15)
            if time.time() - t0 > budget:
                return
            continue
        state, ready = op_state(doc)
        if state != last:
            print("  [%4ds] %s" % (time.time() - t0, state))
            last = state
            if rec is not None:
                rec["state"] = state
                save_receipt(rec)
        if state in TERMINAL and (ready or state != "succeeded"):
            break
        if time.time() - t0 > budget:
            print("Still %s after %d seconds. The run keeps going on the platform.\n"
                  "Come back with: forge.py wait %s" % (state, budget, op))
            return
        time.sleep(10)
    if rec is not None:
        rec["state"] = state
        rec["total_seconds"] = round(time.time() - t0)
        save_receipt(rec)
    if state == "succeeded":
        print("Done in %d seconds. Download with: forge.py fetch %s --dest results/%s"
              % (time.time() - t0, op, op))
        return
    print("Run ended as %s." % state)
    err = doc.get("error") or (doc.get("operation") or {}).get("error")
    if err:
        print("Reason: " + explain(err if isinstance(err, dict) else {"message": str(err)}))
        print("If this was a storage problem, run: forge.py resume %s" % op)


def cmd_status(api, a):
    status, doc = api.rest_json("GET", "/operations/" + a.operation_id)
    if status >= 300:
        die("cannot read that run (HTTP %s)" % status)
    state, ready = op_state(doc)
    print("%s: %s (result ready: %s)" % (a.operation_id, state, ready))
    if a.verbose:
        show(api.c, doc)


def cmd_wait(api, a):
    rec = load_receipt_by_op(a.operation_id)
    _poll(api, a.operation_id, a.budget, rec)


def walk_artifacts(o, acc):
    if isinstance(o, dict):
        if {"artifact_id", "sha256", "size_bytes", "media_type"} <= set(o.keys()):
            acc[o["artifact_id"]] = o
        for v in o.values():
            walk_artifacts(v, acc)
    elif isinstance(o, list):
        for v in o:
            walk_artifacts(v, acc)


def cmd_fetch(api, a):
    os.makedirs(a.dest, exist_ok=True)
    status, doc = api.rest_json("GET", "/operations/%s/result" % a.operation_id)
    with open(os.path.join(a.dest, "result.json"), "w") as fh:
        fh.write(api.c.mask(json.dumps(doc, indent=1)))
    if status >= 300:
        die("no result yet (HTTP %s). Check with: forge.py status %s" % (status, a.operation_id))
    todo, seen, total = {}, set(), 0
    walk_artifacts(doc, todo)
    while todo:
        aid, ref = todo.popitem()
        if aid in seen:
            continue
        seen.add(aid)
        if total + ref["size_bytes"] > a.max_bytes:
            print("  skipped %s (%d bytes, over the size limit)" % (aid, ref["size_bytes"]))
            continue
        st, _, data = api.rest("GET", "/artifacts/%s/content" % aid, timeout=600)
        if st != 200:
            print("  %s could not be downloaded (HTTP %s)" % (aid, st))
            continue
        good = hashlib.sha256(data).hexdigest() == ref["sha256"]
        name = aid + guess_ext(ref["media_type"])
        with open(os.path.join(a.dest, name), "wb") as fh:
            fh.write(data)
        total += len(data)
        print("  %s  %d bytes  checksum %s" % (name, len(data), "ok" if good else "WRONG"))
        if good and ref["media_type"] in JSON_MEDIA and ref.get("compression", "none") == "none":
            try:
                more = {}
                walk_artifacts(json.loads(data), more)
                for k, v in more.items():
                    if k not in seen:
                        todo[k] = v
            except json.JSONDecodeError:
                pass
    print("Saved in %s" % a.dest)


def guess_ext(media, data=b""):
    for ext, m in MEDIA_BY_EXT.items():
        if m == media:
            return ext
    head = data[:400]
    if head.startswith(b">"):
        return ".fasta"
    if head.startswith(b"HEADER") or head.startswith(b"ATOM") or head.startswith(b"REMARK"):
        return ".pdb"
    if head.startswith(b"data_"):
        return ".cif"
    if head.startswith(b"{") or head.startswith(b"["):
        return ".json"
    first = head.split(b"\n", 1)[0]
    if b"," in first and first.isascii() and b"\x00" not in head:
        return ".csv"
    if (media or "").startswith("text/") or (head.isascii() and b"\x00" not in head):
        return ".txt"
    return ".bin"


def cmd_resume(api, a):
    rec = load_receipt_by_op(a.operation_id)
    key = a.key or (rec or {}).get("key")
    if not key:
        die("no idempotency key for that run. Pass --key, or find it in %s" % RUNS_DIR)
    ok, payload = api.call("resume_scientific_run",
                           {"operation_id": a.operation_id, "idempotency_key": key})
    if not ok:
        die(explain(payload))
    print("Resumed. Following the same run.")
    _poll(api, a.operation_id, a.budget, rec)


def cmd_runs(api, a):
    if not os.path.isdir(RUNS_DIR):
        print("No runs recorded yet in %s" % RUNS_DIR)
        return
    rows = []
    for name in sorted(os.listdir(RUNS_DIR)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(RUNS_DIR, name)) as fh:
            rows.append(json.load(fh))
    for r in sorted(rows, key=lambda x: x.get("submitted_at") or 0):
        when = time.strftime("%H:%M", time.localtime(r.get("submitted_at") or 0))
        print("%-5s %-16s %-11s %-22s %s" % (
            when, r.get("model"), r.get("state"), r.get("operation_id") or "-", r.get("label")))


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Nebius Scientific AI Platform client")
    ap.add_argument("--creds", help="path to credentials.private.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("models", help="list models your key may use")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_models)

    p = sub.add_parser("schema", help="read one model schema")
    p.add_argument("model")
    p.add_argument("--full", action="store_true", help="whole schema, can be long")
    p.set_defaults(fn=cmd_schema)

    p = sub.add_parser("run", help="run an inline model")
    p.add_argument("model")
    p.add_argument("--args", default="{}", help="JSON arguments")
    p.add_argument("--args-file", help="read the JSON arguments from a file")
    p.add_argument("--tool", help="override the MCP tool name")
    p.add_argument("--label", help="short name for this run")
    p.add_argument("--budget", type=int, default=1800)
    p.add_argument("--no-wait", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("submit", help="run a file-based model")
    p.add_argument("model")
    p.add_argument("--entry", action="append",
                   help="name|semantic_type|path[|media_type], repeat for more files")
    p.add_argument("--params", default="{}", help="JSON parameters")
    p.add_argument("--operation", help="override the operation name")
    p.add_argument("--tool", help="override the MCP tool name")
    p.add_argument("--service-class", default="customer-batch")
    p.add_argument("--display-name")
    p.add_argument("--wrap-request", action="store_true",
                   help="put the run fields inside 'request' (molecular dynamics)")
    p.add_argument("--label", help="short name for this run")
    p.add_argument("--budget", type=int, default=1800)
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--force", action="store_true", help="submit again even if the run exists")
    p.set_defaults(fn=cmd_submit)

    p = sub.add_parser("status", help="state of one run")
    p.add_argument("operation_id")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("wait", help="poll until the run is done")
    p.add_argument("operation_id")
    p.add_argument("--budget", type=int, default=1800)
    p.set_defaults(fn=cmd_wait)

    p = sub.add_parser("fetch", help="download results")
    p.add_argument("operation_id")
    p.add_argument("--dest", required=True)
    p.add_argument("--max-bytes", type=int, default=50 * 1024 * 1024)
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("resume", help="restart a run after a platform error")
    p.add_argument("operation_id")
    p.add_argument("--key")
    p.add_argument("--budget", type=int, default=1800)
    p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("runs", help="list runs recorded on this machine")
    p.set_defaults(fn=cmd_runs)

    a = ap.parse_args()
    creds = Creds(find_creds(a.creds))
    a.fn(Api(creds), a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Your run keeps going. Use forge.py runs to find it.")
        sys.exit(130)
