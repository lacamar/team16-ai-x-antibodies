"""Exact, local re-implementation of the abguard hard filters that actually bite a
CDRH3-only trastuzumab redesign.

abguard and immunum are NOT installed on this VM -- the organisers run them. But because
we only ever rewrite the 10-residue CDRH3 inside trastuzumab's own (human) VH/VL
framework, most of the hard checks are satisfied by construction and the rest are exact
string computations. We implement exactly those, so a "passing" design here is one that
will pass abguard's novelty/designability gate, with no guessed thresholds.

What abguard checks (hackathon task PDF, "Hard filters") and how it maps here:

  * Format VHH / scFv / Fab .................. satisfied by construction (paired VH/VL Fab).
  * Total length <= 250 aa ................... satisfied by construction (VH 120 + VL 107 = 227).
  * No unpaired cysteines ................... enforced here: reject any Cys in the CDRH3.
      The framework's only cysteines are the conserved intradomain pairs (unchanged); a
      Cys introduced into the CDRH3 would be the one way to create an unpaired cysteine,
      so forbidding Cys in the span is a sufficient, exact guard.
  * immunum numbering confidence >= 0.1 ...... satisfied by construction (real trastuzumab
      framework; immunum is not installed so this cannot be scored, only assumed -- noted).
  * Optimisation novelty:
      - CDRH3 identity to trastuzumab < 70% .. exact: >= 4 of 10 positions must differ.
      - strictly > 2 point mutations from any provided antibody .. exact: CDRH3 Hamming
          distance >= 3 to EVERY provided antibody. (Conservative reading: the design must
          be >2 mutations away from every provided data point, not just one. Passing this
          also passes the looser "from at least one" reading.) Provided antibodies with a
          different framework are automatically > 2 mutations away, so only same-framework
          CDRH3s constrain -- we compare against the union of all provided CDRH3s.

All checks operate on the 10-residue CDRH3 string (same length as trastuzumab's).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from her2_iglm.data import TRASTUZUMAB_CDRH3, CDRH3_LENGTH

#: < 70% identity to a 10-mer == at most 6 matches == at least 4 mismatches.
MAX_IDENTITY_FRACTION = 0.70
MIN_MISMATCHES_VS_TRASTUZUMAB = int(np.floor(MAX_IDENTITY_FRACTION * CDRH3_LENGTH)) + 1  # = 4

#: "strictly more than 2 point mutations" -> Hamming distance strictly > 2 -> >= 3.
MIN_HAMMING_TO_ANY_PROVIDED = 3


def _as_uint8_matrix(seqs: Iterable[str], length: int = CDRH3_LENGTH) -> np.ndarray:
    """Pack equal-length residue strings into an (n, length) uint8 matrix of byte codes."""
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    return arr.reshape(-1, length)


def identity_mismatches_vs_trastuzumab(cdrh3: np.ndarray) -> np.ndarray:
    """Per-candidate count of positions differing from trastuzumab's CDRH3."""
    ref = _as_uint8_matrix([TRASTUZUMAB_CDRH3])  # (1, 10)
    return (cdrh3 != ref).sum(axis=1)


def min_hamming_to_provided(
    candidates: np.ndarray,
    provided: np.ndarray,
    chunk: int = 4096,
) -> np.ndarray:
    """Minimum CDRH3 Hamming distance from each candidate to the provided-antibody set.

    candidates: (n, L) uint8, provided: (m, L) uint8. Returns (n,) int.
    Vectorised in chunks over candidates to bound memory (n*m*L can be large).
    """
    n = candidates.shape[0]
    out = np.empty(n, dtype=np.int16)
    for start in range(0, n, chunk):
        block = candidates[start : start + chunk]  # (b, L)
        # (b, 1, L) vs (1, m, L) -> (b, m) mismatch counts, then min over provided.
        d = (block[:, None, :] != provided[None, :, :]).sum(axis=2)
        out[start : start + chunk] = d.min(axis=1)
    return out


def _onehot_bytes(mat: np.ndarray, device):
    """(n,L) uint8 byte codes -> (n, L*256) float16 one-hot on device (positional)."""
    import torch
    n, L = mat.shape
    t = torch.from_numpy(mat.astype(np.int64)).to(device)  # (n,L) byte values 0..255
    oh = torch.zeros((n, L, 256), dtype=torch.float16, device=device)
    oh.scatter_(2, t.unsqueeze(2), 1.0)
    return oh.reshape(n, L * 256)


def min_hamming_to_provided_torch(candidates: np.ndarray, provided: np.ndarray,
                                   device, chunk: int = 2048) -> np.ndarray:
    """GPU min CDRH3 Hamming distance from each candidate to the provided set.

    Uses the one-hot matmul identity: matching positions = <onehot(a), onehot(b)>, so
    Hamming = L - matches and min-Hamming = L - max-matches. Memory is O(chunk * m), not
    O(chunk * m * L). Fast for n~1e5 vs m~5e5 on an H100.
    candidates (n,L) uint8, provided (m,L) uint8. Returns (n,) int16.
    """
    import torch
    L = candidates.shape[1]
    prov_oh = _onehot_bytes(provided, device).t().contiguous()  # (L*256, m) f16
    n = candidates.shape[0]
    out = np.empty(n, dtype=np.int16)
    for s in range(0, n, chunk):
        block_oh = _onehot_bytes(candidates[s : s + chunk], device)  # (b, L*256) f16
        matches = block_oh @ prov_oh  # (b, m) f16, count of matching positions
        best = matches.max(dim=1).values  # (b,) max matching positions
        out[s : s + chunk] = (L - best).round().to(torch.int16).cpu().numpy()
    return out


def eligibility_report(
    cdrh3_seqs,
    provided_cdrh3=None,
    provided_matrix: "np.ndarray" = None,
    chunk: int = 2048,
    device=None,
):
    """Return a dict of boolean masks and the overall pass mask for a list of CDRH3 strings.

    cdrh3_seqs: list[str] of length-10 candidate CDRH3s.
    provided_cdrh3 / provided_matrix: the provided antibodies' length-10 CDRH3s, either as an
        iterable of strings or a pre-packed (m,10) uint8 matrix (preferred at scale). Candidates
        with a different framework are auto-eligible on the Hamming rule, so only length-10
        provided CDRH3s constrain and are used here.
    device: if given (a torch device), the min-Hamming runs on GPU (fast for n~1e5).
    """
    cand = _as_uint8_matrix(cdrh3_seqs)

    has_cys = np.array(["C" in s for s in cdrh3_seqs], dtype=bool)
    mism_tz = identity_mismatches_vs_trastuzumab(cand)
    identity_ok = mism_tz >= MIN_MISMATCHES_VS_TRASTUZUMAB  # < 70% identity to trastuzumab

    if provided_matrix is None:
        provided_matrix = _as_uint8_matrix([s for s in provided_cdrh3 if len(s) == CDRH3_LENGTH])
    if device is not None:
        min_hd = min_hamming_to_provided_torch(cand, provided_matrix, device, chunk=chunk)
    else:
        min_hd = min_hamming_to_provided(cand, provided_matrix, chunk=chunk)
    novelty_ok = min_hd >= MIN_HAMMING_TO_ANY_PROVIDED  # > 2 mutations from any provided

    no_cys = ~has_cys
    passed = identity_ok & novelty_ok & no_cys

    return {
        "identity_ok": identity_ok,
        "novelty_ok": novelty_ok,
        "no_cys": no_cys,
        "pass": passed,
        "mismatches_vs_trastuzumab": mism_tz,
        "min_hamming_to_provided": min_hd,
    }
