# Recipes

Every recipe was run on 11 November 2026 on a real participant account. The times
are what those runs took.

```bash
export NEBIUS_CREDS=/path/to/credentials.private.json
F=/path/to/nebius-scientific-ai/forge.py
```

Examples use 1ZVY: a nanobody, chain A, bound to lysozyme, chain B.

```bash
curl -sL -o 1zvy.pdb https://files.rcsb.org/download/1ZVY.pdb
```

VHH sequence used below:

```
DVQLVESGGGSVQAGGSLRLSCAASGSTDSIEYMTWFRQAPGKAREGVAALYTHTGNTYYTDSVKGRFTISQDKAKNMAYLRMDSVKSEDTAIYTCGATRKYVPVRFALDQSSYDYWGQGTQVTVSSRGRHHHHHH
```

## 1. Fold a sequence (Boltz2, 80 s)

Boltz2 needs an alignment per chain. A one-line alignment works for a first look.

`boltz.json`:

```json
{"polymers": [{"id": "A", "molecule_type": "protein", "sequence": "<SEQ>",
  "msa": {"msa_search": {"a3m": {"alignment": ">query\n<SEQ>\n"}}}}]}
```

```bash
python3 "$F" run boltz2 --args-file boltz.json --label fold-1
python3 "$F" fetch op-xxxx --dest results/fold-1
```

Result holds `structures`, `confidence_scores`, `ptm_scores`. The VHH above
scored 0.93.

## 2. Co-fold two chains (Boltz2, 75 s)

Same shape, two polymers. This is how you check a binder against its target.

```json
{"polymers": [
  {"id": "A", "molecule_type": "protein", "sequence": "<VHH>",
   "msa": {"msa_search": {"a3m": {"alignment": ">query\n<VHH>\n"}}}},
  {"id": "B", "molecule_type": "protein", "sequence": "<TARGET>",
   "msa": {"msa_search": {"a3m": {"alignment": ">query\n<TARGET>\n"}}}}]}
```

The real 1ZVY pair gave confidence 0.77 and pTM 0.57, against 0.93 for the VHH
alone. The output has **no ipTM**, so do not report one. For an interface number,
count contacts across the chains in the returned mmCIF yourself.

## 2b. Build a nanobody model (NanoBodyBuilder2, 130 s)

Antibody-specific, much better than a general folder for a VHH. One plain FASTA,
header `H`.

```bash
printf '>H\n%s\n' "$VHH" > nebius-runs/inputs/vhh.fasta
python3 "$F" submit nanobodybuilder2 \
  --entry 'sequence|nanobody-fasta/v1|nebius-runs/inputs/vhh.fasta' \
  --params '{"numbering_scheme":"imgt"}' --label nbb2-1
```

Returns a refined PDB. ABodyBuilder3 is the same call with headers `H` and `L`
for a paired antibody, entry `sequences|antibody-fasta/v1`.

## 2c. Redesign the CDRs of a nanobody (AntiFold, 100 s)

Antibody inverse folding. Keeps the framework, proposes new CDR loops. This is
the antibody-aware version of recipe 3.

```bash
COPYFILE_DISABLE=1 tar -cf nebius-runs/inputs/antifold.tar -C nebius-runs/inputs 1zvy.pdb
python3 "$F" submit antifold \
  --entry 'workload|inverse-folding-tar/v1|nebius-runs/inputs/antifold.tar' \
  --params '{"pdb_file":"1zvy.pdb","nanobody_chain":"A","nanobody_mode":true,
             "regions":"CDR1 CDR2 CDR3","num_seq_per_target":5,"sampling_temp":"0.20"}' \
  --label antifold-1
```

Returns a FASTA of redesigned sequences with per-sequence scores, plus a CSV with
per-residue perplexity and the assumed region of every position. The tested run
scored 125 residues of 1ZVY chain A. ESM-IF1 uses the same entry and tar, with
`custom_chain_mode` instead of the antibody chain fields.

Feed the new sequences into recipe 2 to co-fold each one against the target.

## 3. Design sequences for a backbone (ProteinMPNN, 100 s)

Keeps the backbone, proposes new sequences for chain A only.

`mpnn.json`:

```json
{"input_pdb": {"$upload": "1zvy.pdb", "media_type": "chemical/x-pdb"},
 "input_pdb_chains": ["A"], "num_seq_per_target": 4,
 "sampling_temp": 0.1, "random_seed": 42}
```

```bash
python3 "$F" run proteinmpnn --args-file mpnn.json --label design-1
```

Returns `mfasta` with the sequences and `backbone_coverage`, which lists
positions it could not design because the structure does not resolve them. In
1ZVY that is residue 1 of chain A. Check that field before trusting a design.

Useful: `omit_AAs` to forbid residues, higher `sampling_temp` for more variety.

## 4. Search an alignment (MSA search, 40 s)

```bash
python3 "$F" run msa-search-pdb70 --label msa-1 --args '{
  "sequence": "<SEQ>", "databases": ["pdb70_220313"],
  "output_alignment_formats": ["a3m"]}'
```

Put the returned a3m into recipe 1 or 2 instead of the one-line version. For a
VHH, PDB70 often finds few relatives. That is normal.

## 5. Second opinion on a fold (OpenFold3, 3 min)

```bash
python3 "$F" run openfold3 --label of3-1 --args '{
  "request_id": "vhh-1",
  "inputs": [{"input_id": "VHH", "output_format": "cif",
    "molecules": [{"id": "A", "type": "protein", "sequence": "<SEQ>"}]}]}'
```

Two models agreeing is worth more than one high score.

## 6. Generate a binder (RFD4-Proteina, 6 min)

File-based. Takes one annotated mmCIF.

```bash
python3 "$F" submit rfd4-proteina \
  --entry 'annotated-cif|rfd4-proteina-annotated-cif/v1|target.cif' \
  --params '{"num_replicates":1,"batch_size":1,"seed":42,"nsteps":60,"recycle_strategy":"thinking_medium","compile":true,"save_trajectory_every":0,"save_chain_of_interest":false}' \
  --label binder-1
```

Test run took 385 s and returned a tar of designs. Raise `num_replicates` for
more, and expect the time to grow.

**The output is a general protein binder, not a VHH.** RFD4-Proteina has no
antibody prior. A generated backbone usually needs recipe 3 after it to get a
sequence.

Read `python3 "$F" schema rfd4-proteina` before changing the parameters.

## 7. Molecular dynamics

MDprecheck first, it catches broken topology in seconds. It needs the same
bundle you are about to simulate, and it answers in the same call:

```bash
python3 "$F" run mdprecheck --label md-check --args '{
  "input": {"$upload": "inputs.tar", "media_type": "application/x-tar"},
  "input_format": "tar"}'
```

It returns a verdict plus findings by severity, for example that `nstlist`
should be 20 or more on a GPU. A pass is not proof the science is right.

GROMACS works and needs `--wrap-request`.
Without it you get `missing required fields: request`.

```bash
python3 "$F" submit gromacs --wrap-request \
  --entry 'gromacs-inputs|gromacs-input-bundle/v1|inputs.tar' \
  --params '{}' --label md-1
```

The input is a tar bundle and it has to stay **gzipped**. Send `inputs.tar.gz`
as it is. If you unpack it first the engine stops with "not a gzip file".
`schema gromacs` says what goes inside the bundle. A short hackathon run is not
an equilibrated simulation.

## A small pipeline

1. `msa-search-pdb70` on the target, keep the a3m
2. `boltz2` or `openfold3` on the target alone, check it folds sensibly
3. Make candidates: recipe 2c on a known VHH framework, which is antibody aware,
   or recipe 6 for a general binder backbone
4. `nanobodybuilder2` on each new sequence for a quick antibody-aware model, or
   `proteinmpnn` on any backbone that still needs a sequence
5. Recipe 2 on each candidate with the target, 75 s each
6. Re-check the best with `openfold3`

Steps 1, 2 and 5 are quick. Start any `rfd4-proteina` or `mosaic` job early.

Everything above was run on 12 November on a real account, with about 25 jobs
open at once. That is close to what a busy hackathon feels like: small models
answered in 2 to 5 minutes, the big ones in 10 to 25. One model, OpenFold2,
never got an endpoint up and failed after 55 minutes, both under that load and
later on an idle queue. Do not plan around it.

## Models not on the platform

All open source, so install them on your team VM instead. Sources from the
event readiness sheet.

NanoBodyBuilder2, ABodyBuilder3, AntiFold, ESM-IF1, IgBert, Chai-1, the ESM C,
ESM-2 and SaProt families became hosted on 12 November. Run them here, do not
install them.

| Model | Source |
| --- | --- |
| RFantibody | github.com/RosettaCommons/RFantibody |
| Germinal | github.com/SantiagoMille/germinal |
| IgDiff | github.com/amelie-iska/IgDiff |
| PXDesign | github.com/bytedance/PXDesign |
| RFdiffusion3 | github.com/RosettaCommons/foundry |
| nanoBERT | github.com/abmap/nanoBERT |
| OpenDDE | aurekaresearch.github.io/OpenDDE-Website |
