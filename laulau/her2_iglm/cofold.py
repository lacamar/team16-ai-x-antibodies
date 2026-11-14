"""Co-fold VH:VL:HER2 complexes with the local Boltz-2 Foundry image and score them.

This is the expensive tier of the cascade. For each candidate CDRH3 we build the full
trastuzumab Fv (VL = chain A, VH = chain B with the candidate CDRH3 spliced into
positions 99-108) plus HER2 domain IV (chains C+D, from PDB 1N8Z, split at the unresolved
589-ish loop), and co-fold against the 1N8Z template -- exactly the inputs and command the
organisers' HER2 example uses (rfd4-proteina/docs/18-vm.md).

From each predicted complex we read, with no guessed metric:
  * iptm, ptm, plddt, model_ranking_score ...... from <q>_scores.json
  * ab:ag interface pTM ........................ chain-chain interface pTM, VH(B) vs HER2(C,D)
  * min_ipae (B vs HER2, and B vs all others) .. from the token-token PAE in the detailed cbor
  * binder_plddt ............................... mean pLDDT (B-factor) over chain B
  * epitope contacts ........................... HER2 residues within CONTACT_CUTOFF A of CDRH3

Optionally an apo fold (VL+VH alone, no antigen, no template) gives apo_plddt / apo_ptm --
an on-VM foldability / thermostability proxy (there is no dedicated ddG/Tm tool on this VM).
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ASSETS = os.path.join(ROOT, "her2_iglm/assets")
TEMPLATE = os.path.join(ASSETS, "her2_fv_1n8z_template.cif")
HER2_CHAINS = json.load(open(os.path.join(ASSETS, "her2_chains.json")))

BOLTZ_IMAGE = "quay.io/apheris/oslo-hackathon:apheris-boltz-v1.3.0-boltz2"

# CDRH3 occupies VH residues 99..108 (1-indexed) == vh[98:108]; see data.CDRH3_RANGE.
CDRH3_SEQID = list(range(99, 109))
CONTACT_CUTOFF = 4.5  # angstrom, heavy-atom contact definition for the epitope

# RFC 8746 typed-array CBOR tags -> numpy dtype (only those Boltz emits).
_CBOR_DTYPE = {85: "<f4", 86: "<f8", 81: ">f4", 82: ">f8"}


def build_vh(cdrh3: str, framework_prefix: str, framework_suffix: str) -> str:
    return framework_prefix + cdrh3 + framework_suffix


def build_queries(
    items: Sequence[tuple],  # (query_id, cdrh3)
    vl_sequence: str,
    framework_prefix: str,
    framework_suffix: str,
    holo: bool = True,
) -> dict:
    """Build a Boltz queries.json for a set of candidates. holo adds HER2 chains C,D."""
    queries = []
    for qid, cdrh3 in items:
        vh = build_vh(cdrh3, framework_prefix, framework_suffix)
        chains = [
            {"chain_id": "A", "sequence": vl_sequence, "msa_path": None},
            {"chain_id": "B", "sequence": vh, "msa_path": None},
        ]
        if holo:
            for c in HER2_CHAINS:
                chains.append({**c, "msa_path": None})
        queries.append({"query_id": qid, "chains": chains})
    return {"queries": queries}


def _decode_ndarray(tag) -> np.ndarray:
    dims, flat = tag.value
    dt = _CBOR_DTYPE.get(getattr(flat, "tag", None), "<f4")
    arr = np.frombuffer(flat.value, dtype=np.dtype(dt))
    return arr.reshape(dims)


def _docker_run(image: str, args: List[str], workdir: str) -> None:
    uid = os.getuid()
    gid = os.getgid()
    cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--shm-size", "16g",
        "--user", f"{uid}:{gid}", "-e", "HOME=/tmp",
        "-v", "/etc/passwd:/etc/passwd:ro",
        "-v", f"{ROOT}:/work", "-w", "/work",
        "--entrypoint", "python3", image,
        "-m", "apheris_boltz.cli", *args,
    ]
    subprocess.run(cmd, check=True, cwd=workdir)


def run_predict(queries_path: str, out_dir: str, template: Optional[str],
                seeds=(42,), num_diffusion_samples: int = 1) -> None:
    """Invoke `apheris_boltz.cli predict`. Paths are relative to the project root (/work)."""
    rel_q = os.path.relpath(queries_path, ROOT)
    rel_o = os.path.relpath(out_dir, ROOT)
    args = [
        "predict", "--input", rel_q, "--output", rel_o,
        "--write-detailed-scores", "true",
        "--num-diffusion-samples", str(num_diffusion_samples),
        "--query-failures", "skip",
    ]
    for s in seeds:
        args += ["--seeds", str(s)]
    if template is not None:
        args += ["--template-paths", os.path.relpath(template, ROOT)]
    _docker_run(BOLTZ_IMAGE, args, workdir=ROOT)


def _chain_token_blocks(st) -> Dict[str, range]:
    """Token index ranges per chain, in model order (one token per residue)."""
    blocks = {}
    start = 0
    for chain in st[0]:
        n = len(chain)
        blocks[chain.name] = range(start, start + n)
        start += n
    return blocks


def parse_complex(scores_path: str, cif_path: str, detailed_path: str) -> dict:
    import gemmi
    sc = json.load(open(scores_path))
    ccm = np.array(sc["chain_chain_interface_predicted_tm_score"])  # (n,n) chain order
    out = {
        "iptm": float(sc["interface_predicted_tm_score"]),
        "ptm": float(sc["predicted_tm_score"]),
        "plddt": float(sc["average_predicted_local_distance_difference_test"]),
        "pde": float(sc["global_predicted_distance_error"]),
        "ranking": float(sc["model_ranking_score"]),
    }
    # VH(B, idx1) vs HER2(C idx2, D idx3) interface pTM, symmetrised -- only for the holo
    # complex (>=4 chains: A,B,C,D). Apo (A,B only) has no antigen interface.
    if ccm.shape[0] >= 4:
        out["ab_ag_iptm"] = float(np.mean([ccm[1, 2], ccm[1, 3], ccm[2, 1], ccm[3, 1]]))
    st = gemmi.read_structure(cif_path)
    blocks = _chain_token_blocks(st)
    model = st[0]

    # binder pLDDT: mean B-factor over chain B atoms
    b_atoms = [a.b_iso for res in model["B"] for a in res]
    out["binder_plddt"] = float(np.mean(b_atoms)) if b_atoms else float("nan")

    # min interface PAE
    try:
        import cbor2
        det = cbor2.load(gzip.open(detailed_path))
        pae = _decode_ndarray(det["token_token_predicted_aligned_error"])
        B = list(blocks["B"])
        her2 = list(blocks.get("C", [])) + list(blocks.get("D", []))
        others = [i for name, rng in blocks.items() if name != "B" for i in rng]
        if her2:
            out["min_ipae_her2"] = float(min(pae[np.ix_(B, her2)].min(),
                                             pae[np.ix_(her2, B)].min()))
        if others:
            out["min_ipae_all"] = float(min(pae[np.ix_(B, others)].min(),
                                            pae[np.ix_(others, B)].min()))
    except Exception as e:  # detailed scores optional; never fail the whole scorer on this
        out["pae_error"] = str(e)

    # epitope contacts: HER2 residues within CONTACT_CUTOFF of any CDRH3 atom
    if "C" in blocks or "D" in blocks:
        cdr_atoms = []
        for res in model["B"]:
            if res.seqid.num in CDRH3_SEQID:
                cdr_atoms.extend([(a.pos.x, a.pos.y, a.pos.z) for a in res])
        cdr = np.array(cdr_atoms) if cdr_atoms else np.zeros((0, 3))
        contacts = set()
        for cname in ("C", "D"):
            if cname not in [c.name for c in model]:
                continue
            for res in model[cname]:
                coords = np.array([(a.pos.x, a.pos.y, a.pos.z) for a in res])
                if len(coords) and len(cdr):
                    dmin = np.sqrt(((coords[:, None, :] - cdr[None, :, :]) ** 2).sum(-1)).min()
                    if dmin <= CONTACT_CUTOFF:
                        contacts.add(f"{cname}{res.seqid.num}")
        out["epitope_contacts"] = sorted(contacts)
    return out


def collect_results(out_dir: str, query_ids: List[str], seed: int = 42) -> Dict[str, dict]:
    """Parse every predicted complex under out_dir, keyed by query_id."""
    results = {}
    for qid in query_ids:
        base = os.path.join(out_dir, qid, f"seed_{seed}", f"{qid}_seed_{seed}_sample_1")
        sp, cp, dp = base + "_scores.json", base + "_model.cif", base + "_detailed_scores.cbor.gz"
        if not (os.path.exists(sp) and os.path.exists(cp)):
            results[qid] = {"status": "missing"}
            continue
        try:
            results[qid] = parse_complex(sp, cp, dp)
            results[qid]["status"] = "ok"
        except Exception as e:
            results[qid] = {"status": "error", "error": str(e)}
    return results


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
