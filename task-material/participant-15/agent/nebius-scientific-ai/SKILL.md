---
name: nebius-scientific-ai
description: Run models on the Nebius Scientific AI Platform at the AI x Antibodies Hackathon Oslo. Use when the user wants to fold a protein or nanobody, design a binder, design sequences for a backbone, search an MSA, dock a molecule, run molecular dynamics, or check a run that is already going. Handles model discovery, inline models, file-based models, waiting, resuming and downloading results.
---

# Nebius Scientific AI Platform

Everything goes through `forge.py`, next to this file. Let `F` be its path.

## You are the transport layer, not the only skill

The official Nebius `scientific-ai` bundle is installed alongside this one: 51
skills, one per model, with the scientific parameters, upstream provenance and
references. Use both.

- **Their skill, for the science.** Before running a model, read its skill if one
  exists: `boltz2`, `openfold2`, `openfold3`, `proteinmpnn`, `rfd4-proteina`,
  `rfdiffusion`, `diffdock`, `bioemu2`, `msa-search`, `gromacs`, `amber`,
  `namd`, `lammps`, `openmm`, `md-precheck`, plus `protein-binder-design`,
  `binder-design-campaigns`, `protein-scoring` and
  `msa-structure-prediction-pipeline`. They carry parameter meaning, upstream
  revisions and caveats this file does not repeat.
- **This skill, for getting the job to run.** Discovery, submission, waiting,
  resuming and downloading from a laptop, through `forge.py`.

Where they disagree on mechanics, prefer what is written here: it was measured
against the live platform on 12 November 2026. One of their instructions does not
hold for a laptop:

- `scientific-gateway` sends you to `/opt/bionemo/invoke-scientific-batch.py`
  for file-based work. That file is on the workbench image and does not ship in
  the bundle. `forge.py submit` is the replacement.

They are right about `input_artifact_contract`. The schema now carries it, and it
is how you call a model that has no recipe here. See "A model with no recipe".

The team credentials file is `credentials.private.json` from their team folder:

```bash
export NEBIUS_CREDS=/path/to/credentials.private.json
```

Ask where it is if they have not said. Never open it, print it or commit it.

## Hard rules

1. **Never pull the whole catalog.** The MCP tool list is about 310 KB and will
   fill your context. Use `forge.py models` and `forge.py schema <model>`. Do not
   call `tools/list` even if the MCP server is connected.
2. **Never guess an input format.** Either the model is in `forge.py models`, or
   its schema publishes an `input_artifact_contract` and you follow that exactly.
   If neither is true, say the model cannot be called here, offer a hosted
   alternative, and mention the team can run it on their own VM.
3. **Never submit the same job twice.** Repeating a command follows the existing
   run. After a storage failure use `resume`, not a new submit.
4. **Say the wait before it starts.** The script prints an estimate. Repeat it.

## Commands

```bash
python3 "$F" models                  # models with a tested input format
python3 "$F" models --all            # every name on the key
python3 "$F" schema boltz2           # one model, trimmed to what you need

python3 "$F" run proteinmpnn --args-file nebius-runs/design-1.json --label design-1
python3 "$F" submit rfd4-proteina \
  --entry 'annotated-cif|rfd4-proteina-annotated-cif/v1|target.cif' \
  --params '{...}' --label binder-1

python3 "$F" runs | status op-xxx | wait op-xxx | resume op-xxx
python3 "$F" fetch op-xxx --dest results/run1
```

Read the schema of the one model you chose, not of several in a loop.

**Picking a model for antibody work.** A VHH or nanobody alone goes to
`nanobodybuilder2`, a paired antibody to `abodybuilder3`: both are antibody
aware, both answer in about 2 minutes, and neither needs an alignment. Use
`boltz2` when the question involves the target, for example co-folding a
candidate against it. For new CDR loops use `antifold`, which keeps the
framework, rather than `proteinmpnn`, which has no antibody prior.

Keep the team folder root clean: it holds their credentials file. Inputs and
argument files go in `nebius-runs/`, downloads in `results/`. Do not create new
folders at the root. `mkdir -p nebius-runs` first if it is not there yet.

**Inline models** take arguments in one call. In the arguments,
`{"$upload": "file.pdb"}` uploads a local file and becomes an artifact
reference, and `{"$text": "file.txt"}` becomes the file text.

**File-based models** need `--entry name|semantic_type|path`, repeatable. The
name and semantic type come from the schema. Add `--wrap-request` for GROMACS,
AMBER, NAMD, LAMMPS and OpenMM, which want the run fields inside `request`.

## A model with no recipe

A model outside `forge.py models` can still be called when its schema publishes a
contract. One pass, no guessing:

1. `python3 "$F" schema <model> --full` and read `input_artifact_contract`
2. Per operation it gives the manifest entry `name`, the `semantic_type`, the
   `media_type` and the file layout. The top level gives `tool_name` and
   `request_schema.parameters`
3. Build exactly that file. For a tar, put the listed paths at the archive root
   with no leading `./` and no macOS extras: `COPYFILE_DISABLE=1 tar -cf out.tar -C dir .`
4. Submit with the tool and operation the schema named:

```bash
python3 "$F" submit <model> --tool <tool_name> --operation <operation> \
  --entry '<entry name>|<semantic type>|<path>' --params '{...}' --label <label>
```

If it returns 422, the message names the field. Fix that one field, try once
more, then stop and say what is missing.

`run` and `submit` wait and print state changes. `--no-wait` returns the shell.
`fetch` checks every checksum and follows nested manifests. Runs are recorded in
`nebius-runs/` in the current folder, which is how `resume` finds the key, so
keep working from one folder.

## First call times

Measured 12 November on one account with about 25 jobs open at once, which is
close to a busy hackathon.

| Model | First call |
| --- | --- |
| mdprecheck | seconds, it answers in the same call |
| msa-search-pdb70, boltz2, proteinmpnn, abodybuilder3 | 30 s to 100 s |
| openfold3, nanobodybuilder2, antifold | 1 to 2 min |
| esmc-300m, esmc-600m, esm2-650m, esm-if1, chai1, saprot-650m, igbert | 3 to 5 min |
| rfdiffusion, esm2-3b, rfd4-proteina, the MD engines | 6 to 8 min |
| saprot-inverse-folding, saprot-1-3b, stab-ddg, esmc-6b | 9 to 15 min |
| esmfold2-fast, protenix-v2, esmfold2, bioemu2, mosaic | 13 to 26 min |

Tell teams to send long jobs early and come back to the run id.

**Do not use OpenFold2.** On 12 November it failed twice with "the endpoint did
not become ready within 3298s (83 probes, last HTTP 000)", once under load and
once on an idle queue, so it is not a queue problem. Each attempt burns 55
minutes before it gives up. Use `openfold3` or `boltz2` for a single chain
instead, and say why if someone asks for OpenFold2 by name.

## Errors

- *weights not granted*: AlphaFold 3. Offer OpenFold3, Boltz2 or ESMFold2.
- *synchronous calls only*: MDprecheck answers in the same call and refuses an
  idempotency key. `forge.py` already handles it.
- *missing required fields: request*: add `--wrap-request`.
- *private storage is temporarily unavailable*: `resume`, do not resubmit.
- *input does not match the schema*: read the schema and fix it. Do not retry
  blind.

## The team also has an H100 VM

It has Nvidia drivers and nothing else: no CUDA toolkit, no Python packages, no
weights. Check `forge.py models` before you send anyone to install something.
NanoBodyBuilder2, ABodyBuilder3, AntiFold, ESM-IF1, IgBert, Chai-1, the ESM C,
ESM-2 and SaProt families are hosted since 12 November, so installing them is
wasted time.

What is still not hosted, and what it costs on the VM:

- **about an hour**: RFantibody, Germinal, IgDiff, PXDesign, RFdiffusion3.

Setup tip worth giving every time: they do not need the CUDA toolkit, because pip
PyTorch wheels carry their own CUDA libraries. `uv pip install torch` on the
driver is enough, and `torch.cuda.is_available()` must print `True` before
anything else goes on.

Split the work: design on their machine, score on the platform, which is where
parallelism comes from. Never suggest installing something already hosted and
working.

## After a run

Say what came out and where the files are, give the score, give the run id, and
suggest the next step. `RECIPES.md` has tested commands for
folding, co-folding a pair, sequence design, binders, MSA and molecular dynamics.
