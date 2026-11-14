# Team 16 dashboard

Live view of the hackathon VM (SSH) plus local scoring (novelty, affinity surrogate, stability).

Backend (stdlib http.server, port 8000; fits surrogate at startup, ~15s):

    /home/lm/.local/src/ai-antibodies/scribe/.venv/bin/python web/server.py 8000

Frontend:

    cd web && npm install && npm run dev    # http://localhost:5173, /api proxied to :8000

Endpoints: /api/summary, /api/status, /api/runs, /api/designs?run=<name>. VM results cached 15s; CIF-derived CDRH3s cached 30min.
