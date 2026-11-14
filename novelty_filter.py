"""Novelty + hard-filter gate (the missing `abguard`).

Challenge novelty filter (rfd4 docs 12-her2.md): a CDRH3 passes if it
  * differs at >= 2 of 10 positions from EVERY provided CDRH3 (the FACS 524k +
    SPR 768 sets), AND
  * is <= 70% identical to trastuzumab WGGDGFYAMD (>= 3 of 10 positions changed).
Known binders live in the provided set, so they fail. Deterministic, no deps.

"differs at >= 2 from every provided"  ==  "no provided seq within Hamming <= 1"
== candidate not in the set and none of its single-mutation neighbours in it.
That makes it O(190) per candidate instead of O(524k).

Plus cheap hard checks (length <= 250, no odd cysteine count). The VHH/scFv/Fab
class + numbering-confidence checks need ANARCI/abnumber — see `hard_checks`.
"""
import csv, io, zipfile
from pathlib import Path

TRASTUZUMAB_CDRH3 = "WGGDGFYAMD"
AA = "ACDEFGHIKLMNPQRSTVWY"
HERE = Path(__file__).resolve().parent


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def load_provided_cdrh3(facs_zip, spr_csv):
    """{length: set(cdrh3)} over all provided CDRH3 sequences."""
    by_len = {}
    def add(s):
        s = (s or "").strip().upper()
        if s:
            by_len.setdefault(len(s), set()).add(s)
    with zipfile.ZipFile(facs_zip) as z:
        with z.open(z.namelist()[0]) as f:
            for row in csv.DictReader(io.TextIOWrapper(f, "utf-8")):
                add(row.get("cdrh3", ""))
    with open(spr_csv) as f:
        for row in csv.DictReader(f):
            add(row.get("cdrh3", ""))
    return by_len


def _neighbours1(s):
    for i, c in enumerate(s):
        for a in AA:
            if a != c:
                yield s[:i] + a + s[i + 1:]


def is_novel(cdrh3, by_len, tz=TRASTUZUMAB_CDRH3):
    cdrh3 = cdrh3.strip().upper()
    if len(cdrh3) == len(tz) and hamming(cdrh3, tz) < 3:
        return False, f"too close to trastuzumab ({hamming(cdrh3, tz)} changes, need >=3)"
    pset = by_len.get(len(cdrh3), set())
    if cdrh3 in pset:
        return False, "identical to a provided CDRH3 (known sequence)"
    for nb in _neighbours1(cdrh3):
        if nb in pset:
            return False, "within 1 mutation of a provided CDRH3"
    return True, "novel"


def hard_checks(full_seq):
    """Cheap deterministic hard filters. NOTE: VHH/scFv/Fab class + numbering
    confidence still need ANARCI/abnumber; add that once installed."""
    full_seq = full_seq.strip().upper()
    issues = []
    if len(full_seq) > 250:
        issues.append(f"length {len(full_seq)} > 250")
    if full_seq.count("C") % 2 == 1:
        issues.append("odd cysteine count (likely unpaired Cys)")
    return (not issues), ("; ".join(issues) or "ok")


def demo():
    facs = HERE / "data/her2/tz_her2_facs_3classes_524346.csv.zip"
    spr = HERE / "data/her2/trastuzumab_her2_spr_all_768.csv"
    by_len = load_provided_cdrh3(facs, spr)
    total = sum(len(s) for s in by_len.values())
    print(f"loaded {total} provided CDRH3; by length: {{k: len for k...}} = " +
          str({k: len(v) for k, v in sorted(by_len.items())}))

    ok, why = is_novel(TRASTUZUMAB_CDRH3, by_len)
    assert not ok, "trastuzumab must fail novelty"
    print(f"  trastuzumab {TRASTUZUMAB_CDRH3}: novel={ok}  ({why})")

    ok, why = is_novel("WGGDGFYAMA", by_len)  # 1 change from trastuzumab
    assert not ok, "1-change variant must fail (too close to trastuzumab)"
    print(f"  1-change variant WGGDGFYAMA: novel={ok}  ({why})")

    known = next(iter(by_len[10]))            # a real provided 10-mer
    ok, why = is_novel(known, by_len)
    assert not ok, "a provided sequence must fail"
    print(f"  provided {known}: novel={ok}  ({why})")

    far = "KTRSYELDPM"                        # far from trastuzumab + the set
    ok, why = is_novel(far, by_len)
    print(f"  far design {far}: novel={ok}  ({why})")

    print("  hard_checks(200aa, even Cys):", hard_checks("QVQLVESGGG" * 20))
    print("  hard_checks(1 Cys):", hard_checks("QVQLVESGGGC"))
    print("self-check passed")


if __name__ == "__main__":
    demo()
