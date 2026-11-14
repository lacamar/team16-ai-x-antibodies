# team16-antibodies — repo rules

One-day hackathon. Python + `uv`. No Rust, no RPM, no beads.

Two rules once we have results:
1. **Provenance** on every number: scorer, model_id, seed, run_id, git_sha, config_hash. Render tables from data, never hand-type.
2. **Decisions** logged with what was rejected and why; one human signs off each.

Architecture plan: `docs/BUILD_SPEC.md`. Models/metrics/liabilities: `docs/GUIDE_DISTILLED.md`.
