"""HTTP wrapper around scoring.score_sequence — standard library only.

    ./.venv/bin/python server.py [port]        # default 8765

    http://localhost:8765/docs                 # Swagger UI
    curl 'localhost:8765/score?cdrh3=WASSSEYYFDY'
    curl -X POST localhost:8765/score -d '{"cdrh3":"WASSSEYYFDY","vh":"EVQ..."}'
    curl -X POST localhost:8765/score/batch -d '{"items":[{"cdrh3":"WASSSEYYFDY"}]}'

Why HTTP and not just the import: the scorers need numpy/scikit-learn/gemmi and the surrogate
costs ~10 s to fit. A long-lived process fits once and serves every teammate and the
dashboard, whatever environment they are in.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import scoring

MAX_BATCH = 500
MAX_BODY = 4 * 1024 * 1024
HERE = Path(__file__).resolve().parent
SPEC = HERE / "openapi.json"
SWAGGER = "https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.17.14"

DOCS_HTML = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HER2 Design Scoring API</title>
  <link rel="stylesheet" href="{SWAGGER}/swagger-ui.min.css">
  <style>body {{ margin: 0 }} .topbar {{ display: none }}</style>
</head>
<body>
  <div id="swagger"></div>
  <script src="{SWAGGER}/swagger-ui-bundle.min.js"></script>
  <script>
    window.ui = SwaggerUIBundle({{
      url: "/openapi.json",
      dom_id: "#swagger",
      deepLinking: true,
      defaultModelsExpandDepth: 0,
      tryItOutEnabled: true,
      presets: [SwaggerUIBundle.presets.apis]
    }});
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload):
        body = json.dumps(payload, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_raw(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError(f"body too large ({n} bytes)")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/health"):
                provided = sum(len(v) for v in scoring.load_provided().values())
                return self._send(200, {
                    "ok": True,
                    "scorer_version": scoring.SCORER_VERSION,
                    "provided_sequences": provided,
                    "routes": ["/health", "/rules", "/docs", "/openapi.json",
                               "/score?cdrh3=", "POST /score", "POST /score/batch"],
                    "docs": "/docs",
                })
            if u.path == "/docs":
                return self._send_raw(200, DOCS_HTML.encode(), "text/html; charset=utf-8")
            if u.path == "/openapi.json":
                if not SPEC.exists():
                    return self._send(500, {"error": f"spec not found at {SPEC}"})
                return self._send_raw(200, SPEC.read_bytes(), "application/json")
            if u.path == "/rules":
                return self._send(200, {
                    "max_trastuzumab_identity": scoring.MAX_TZ_IDENTITY,
                    "min_distance_from_provided": scoring.MIN_DIST_PROVIDED,
                    "de_novo_identity_threshold": scoring.DE_NOVO_IDENTITY,
                    "trastuzumab_cdrh3": {"10": scoring.TZ_10, "11": scoring.TZ_11},
                    "source": "task description; stricter than team16 novelty_filter.py",
                })
            if u.path == "/score":
                q = parse_qs(u.query)
                cdrh3 = (q.get("cdrh3") or [""])[0]
                if not cdrh3:
                    return self._send(400, {"error": "cdrh3 query parameter is required"})
                return self._send(200, scoring.score_sequence(
                    cdrh3, (q.get("vh") or [None])[0], (q.get("design_cif") or [None])[0]))
            self._send(404, {"error": f"no route {u.path}"})
        except Exception as e:                               # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            data = self._body()
            if u.path == "/score":
                if not data.get("cdrh3"):
                    return self._send(400, {"error": "cdrh3 is required"})
                return self._send(200, scoring.score_sequence(
                    data["cdrh3"], data.get("vh"), data.get("design_cif")))
            if u.path == "/score/batch":
                items = data.get("items") or []
                if not isinstance(items, list):
                    return self._send(400, {"error": "items must be a list"})
                if len(items) > MAX_BATCH:
                    return self._send(400, {"error": f"at most {MAX_BATCH} items per call"})
                results = scoring.score_batch(items)
                return self._send(200, {
                    "count": len(results),
                    "eligible": sum(1 for r in results if r.get("eligible")),
                    "results": results,
                })
            self._send(404, {"error": f"no route {u.path}"})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:                               # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    n = sum(len(v) for v in scoring.load_provided().values())
    print(f"loaded {n} provided CDRH3 sequences", flush=True)
    print(f"serving on http://127.0.0.1:{port}  (ctrl-c to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
