"""One entry point that scores a HER2 CDRH3 design on every judged axis.

Why this exists: scoring is spread across six modules in team16-ai-x-antibodies, each with
its own CDRH3 length convention and its own idea of the novelty threshold. That is where the
known bugs came from. This wraps them in a single call with one convention, returns all three
judged axes plus the eligibility verdict, and never silently truncates a sequence.

    from scoring import score_sequence
    score_sequence("WASSSEYYFDY", vh=None, design_cif=None)

Core (novelty, track, eligibility) is standard library only and always works.
Affinity needs numpy+scikit-learn; epitope needs gemmi and a co-folded CIF. When a dependency
is missing the axis reports available=false instead of failing the call.

Thresholds follow the task description (see lngng/scoring-reference.md §1.1), which is
STRICTER than team16's novelty_filter.py. Both verdicts are returned so nothing is hidden.
"""
from __future__ import annotations

import csv
import functools
import io
import os
import zipfile
from itertools import combinations
from pathlib import Path

SCORER_VERSION = "1.0.0"

TZ_10 = "WGGDGFYAMD"
TZ_11 = "WGGDGFYAMDY"          # Kabat CDRH3, with the trailing Y the FACS table drops
TZ_VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTIS"
         "ADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS")
AA = "ACDEFGHIKLMNPQRSTVWY"

# Task rules. Optimisation track: identity to trastuzumab strictly under 70%, and strictly
# more than 2 point mutations from every provided antibody.
MAX_TZ_IDENTITY = 0.70
MIN_DIST_PROVIDED = 3          # strictly more than 2
DE_NOVO_IDENTITY = 0.70        # whole-chain identity below this routes to the de novo track

DATA_DIR = Path(os.environ.get("HER2_DATA", Path.home() / "data" / "her2-trastuzumab"))
TEAM16 = Path(os.environ.get("TEAM16_DIR", Path.home() / "team16-ai-x-antibodies"))


# --------------------------------------------------------------------------- helpers

def _clean(seq: str) -> str:
    return (seq or "").strip().upper().replace(" ", "")


def distance(a: str, b: str) -> int:
    """team16's metric: substitutions over the shared prefix + length difference."""
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def _ball(seq: str, radius: int):
    """Every sequence within `radius` substitutions of seq (radius 0-2)."""
    yield seq
    if radius < 1:
        return
    for i, c in enumerate(seq):
        for a in AA:
            if a != c:
                yield seq[:i] + a + seq[i + 1:]
    if radius < 2:
        return
    for i, j in combinations(range(len(seq)), 2):
        for a in AA:
            if a == seq[i]:
                continue
            for b in AA:
                if b != seq[j]:
                    yield seq[:i] + a + seq[i + 1:j] + b + seq[j + 1:]


# --------------------------------------------------------------------------- provided set

@functools.lru_cache(maxsize=1)
def load_provided() -> dict[int, frozenset]:
    """{length: frozenset(cdrh3)} over all 525,063 provided CDRH3 sequences."""
    by_len: dict[int, set] = {}

    def add(s):
        s = _clean(s)
        if s:
            by_len.setdefault(len(s), set()).add(s)

    facs_csv = DATA_DIR / "tz_her2_facs_3classes_524346.csv"
    facs_zip = DATA_DIR / "tz_her2_facs_3classes_524346.csv.zip"
    if facs_csv.exists():
        with open(facs_csv) as f:
            for row in csv.DictReader(f):
                add(row.get("cdrh3"))
    elif facs_zip.exists():
        with zipfile.ZipFile(facs_zip) as z, z.open(z.namelist()[0]) as f:
            for row in csv.DictReader(io.TextIOWrapper(f, "utf-8")):
                add(row.get("cdrh3"))
    else:
        raise FileNotFoundError(f"FACS table not found under {DATA_DIR}")

    spr = DATA_DIR / "trastuzumab_her2_spr_all_768.csv"
    if spr.exists():
        with open(spr) as f:
            for row in csv.DictReader(f):
                add(row.get("cdrh3"))
    return {k: frozenset(v) for k, v in by_len.items()}


@functools.lru_cache(maxsize=64)
def _prefixes(length: int, upto: int) -> frozenset:
    return frozenset(p[:upto] for p in load_provided().get(length, ()))


def min_distance_is_below(cdrh3: str, threshold: int) -> tuple[bool, str | None]:
    """Exact: is any provided CDRH3 within `threshold`-1 of this one? Returns (hit, which)."""
    by_len = load_provided()
    n = len(cdrh3)
    for m in sorted(by_len):
        budget = (threshold - 1) - abs(n - m)       # substitutions we can still afford
        if budget < 0:
            continue
        L = min(n, m)
        pool = _prefixes(m, L)
        for cand in _ball(cdrh3[:L], budget):
            if cand in pool:
                return True, f"within {abs(n - m) + distance(cdrh3[:L], cand)} of a provided CDRH3"
    return False, None


# --------------------------------------------------------------------------- axes

def novelty(cdrh3: str) -> dict:
    """Eligibility under the optimisation-track rules, strict and legacy side by side."""
    n = len(cdrh3)
    tz = TZ_11 if n == 11 else TZ_10
    d_tz = distance(cdrh3, tz)
    denom = max(len(tz), n)
    identity = (denom - d_tz) / denom

    reasons = []
    if identity >= MAX_TZ_IDENTITY:
        reasons.append(f"CDRH3 identity to trastuzumab {identity:.1%} is not under "
                       f"{MAX_TZ_IDENTITY:.0%} ({d_tz} mutations; needs "
                       f"{int(denom * (1 - MAX_TZ_IDENTITY)) + 1})")
    hit, why = min_distance_is_below(cdrh3, MIN_DIST_PROVIDED)
    if hit:
        reasons.append(f"{why}; needs {MIN_DIST_PROVIDED} or more from every provided sequence")

    legacy_hit, _ = min_distance_is_below(cdrh3, 2)
    legacy_ok = (d_tz >= 3) and not legacy_hit

    return {
        "passes": not reasons,
        "reasons": reasons,
        "mutations_from_trastuzumab": d_tz,
        "identity_to_trastuzumab": round(identity, 4),
        "compared_against": tz,
        "provided_sequences": sum(len(v) for v in load_provided().values()),
        "legacy_team16_passes": legacy_ok,
        "note": ("legacy and strict disagree: team16's novelty_filter.py uses thresholds one "
                 "lower than the task description" if legacy_ok != (not reasons) else None),
    }


def track(vh: str | None) -> dict:
    """Which track abguard would assign. Decided by whole-chain identity, not by us."""
    if not vh:
        return {"track": None, "available": False,
                "note": "pass vh= to classify; the track follows whole-chain identity"}
    vh = _clean(vh)
    d = distance(vh, TZ_VH)
    identity = (max(len(vh), len(TZ_VH)) - d) / max(len(vh), len(TZ_VH))
    is_de_novo = identity < DE_NOVO_IDENTITY
    return {
        "track": "de_novo" if is_de_novo else "optimisation",
        "available": True,
        "identity_to_trastuzumab_vh": round(identity, 4),
        "note": ("Compared against trastuzumab VH only. abguard compares against every known "
                 "antibody, so real identity can only be higher, never lower."),
    }


def thermostability(cdrh3: str) -> dict:
    try:
        import sys
        if str(TEAM16) not in sys.path:
            sys.path.insert(0, str(TEAM16))
        from thermostability import stability_flags, stability_score
    except Exception as e:                                   # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}"}
    score = stability_score(cdrh3)
    return {
        "available": True,
        "score": round(score, 4),
        "baseline_trastuzumab": round(stability_score(TZ_10), 4),
        "flags": stability_flags(cdrh3),
        "method": "team16 thermostability.py liability heuristic",
        "caveat": "Heuristic proxy, NOT a predicted Tm. Upgrade: StaB-ddG on the full VH.",
    }


_SURROGATE = None


def affinity(cdrh3: str) -> dict:
    global _SURROGATE
    try:
        import sys
        if str(TEAM16) not in sys.path:
            sys.path.insert(0, str(TEAM16))
        from affinity_surrogate import AffinitySurrogate
        if _SURROGATE is None:
            _SURROGATE = AffinitySurrogate().fit()
    except Exception as e:                                   # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}",
                "hint": "needs numpy + scikit-learn; see api/README.md"}
    s = _SURROGATE.score(cdrh3)
    base = _SURROGATE.score(TZ_10)
    return {
        "available": True,
        "surrogate": s["surrogate"],
        "pkd": s["pkd"],
        "pred_log_kd": s["pred_log_kd"],
        "pred_kd_nM": round(10 ** s["pred_log_kd"], 4),
        "p_high": s["p_high"],
        "out_of_distribution": s["ood"],
        "baseline_trastuzumab_surrogate": base["surrogate"],
        "method": "RidgeCV on 190 SPR log10 KD + LogisticRegression on 524k FACS",
    }


def epitope(design_cif: str | None) -> dict:
    if not design_cif:
        return {"available": False, "note": "pass design_cif= from a co-folded complex"}
    try:
        import sys
        if str(TEAM16) not in sys.path:
            sys.path.insert(0, str(TEAM16))
        from epitope_check import preservation
        p = preservation(design_cif)
    except Exception as e:                                   # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}"}
    p["available"] = True
    p["method"] = "heavy-atom contacts within 4.5 A, overlap with 1N8Z"
    return p


# --------------------------------------------------------------------------- entry point

def score_sequence(cdrh3: str, vh: str | None = None, design_cif: str | None = None) -> dict:
    """Score one design on every judged axis. Returns a JSON-serialisable dict."""
    cdrh3 = _clean(cdrh3)
    problems = []
    if not cdrh3:
        return {"error": "cdrh3 is required"}
    bad = sorted(set(cdrh3) - set(AA))
    if bad:
        problems.append(f"non-standard residues: {''.join(bad)}")
    if not 8 <= len(cdrh3) <= 20:
        problems.append(f"unusual CDRH3 length {len(cdrh3)}")

    nov = novelty(cdrh3)
    trk = track(vh)
    aff = affinity(cdrh3)
    thm = thermostability(cdrh3)
    epi = epitope(design_cif)

    # per-axis comparison against the trastuzumab baseline — the tiering is a conjunction,
    # so what matters is how many axes beat baseline, not a weighted sum.
    axes = {}
    if aff.get("available"):
        axes["affinity"] = (None if aff["out_of_distribution"]
                            else aff["surrogate"] > aff["baseline_trastuzumab_surrogate"])
    if thm.get("available"):
        axes["thermostability"] = thm["score"] > thm["baseline_trastuzumab"]
    if epi.get("available"):
        axes["epitope"] = epi["epitope_recall"] >= 0.30

    decided = [v for v in axes.values() if v is not None]
    beaten = sum(1 for v in decided if v)
    tier = {3: "S", 2: "A", 1: "B"}.get(beaten, "unsuccessful") if len(decided) == 3 else None

    eligible = nov["passes"] and not problems
    return {
        "cdrh3": cdrh3,
        "length": len(cdrh3),
        "convention": {10: "10-residue (FACS table)",
                       11: "11-residue (Kabat, trailing Y)"}.get(len(cdrh3), "non-standard"),
        "eligible": eligible,
        "verdict": "eligible" if eligible else "rejected",
        "reasons": problems + nov["reasons"],
        "track": trk,
        "novelty": nov,
        "axes": {"affinity": aff, "epitope": epi, "thermostability": thm},
        "predicted_tier": {
            "axes_beating_baseline": beaten,
            "axes_decided": len(decided),
            "per_axis": axes,
            "tier": tier,
            "caveat": ("Predicted from surrogates, not measured. Adaptyv decides the real "
                       "tier. A tie counts as a failure, not a pass."),
        },
        "provenance": {
            "scorer_version": SCORER_VERSION,
            "data_dir": str(DATA_DIR),
            "team16_dir": str(TEAM16),
            "provided_sequences": nov["provided_sequences"],
            "rules": "task PDF: identity < 70%, strictly more than 2 mutations from provided",
        },
    }


def score_batch(items: list[dict]) -> list[dict]:
    """items: [{"cdrh3": ..., "vh": ..., "design_cif": ...}, ...]"""
    return [score_sequence(i.get("cdrh3", ""), i.get("vh"), i.get("design_cif"))
            for i in items]


if __name__ == "__main__":
    import json
    import sys

    seqs = sys.argv[1:] or [TZ_10, "WGPPYFYAMDY", "WASSSEYYFDY", "WGGSGPYLFD", "KTRSYELDPM"]
    for s in seqs:
        r = score_sequence(s)
        print(json.dumps(r, indent=2) if len(seqs) == 1 else
              f"{r['cdrh3']:14} {r['verdict']:9} "
              f"d(TZ)={r['novelty']['mutations_from_trastuzumab']:<3} "
              f"id={r['novelty']['identity_to_trastuzumab']:.1%}  "
              f"{'; '.join(r['reasons']) or 'passes both rules'}")
