# team16-antibodies — repo rules

One-day hackathon. Python + `uv`. No Rust, no RPM, no beads.

Two rules once we have results:
1. **Provenance** on every number: scorer, model_id, seed, run_id, git_sha, config_hash. Render tables from data, never hand-type.
2. **Decisions** logged with what was rejected and why; one human signs off each.

Architecture plan: `docs/BUILD_SPEC.md`. Models/metrics/liabilities: `docs/GUIDE_DISTILLED.md`.

## IgLM weights — ALWAYS use the fine-tuned model
The team's fine-tuned IgLM weights live on the VM at `~/laulau/runs/finetune/best`
(HER2-biased: fine-tuned on the 524k + the reward loop by teammate laulau).
ALWAYS load IgLM from there, not base `IgLM()`, when generating sequences.
See `finetune_iglm.py` (`load_iglm(model_dir)`) and `docs/IGLM_LOOP.md`.
