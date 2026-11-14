"""Heuristic CDRH3 developability/destabilization proxy (NOT a measured Tm).

Flags sequence liabilities of a CDRH3 loop grafted on trastuzumab's fixed VH
framework. Use as a soft penalty / tie-break only.
Upgrade path: ThermoMPNN or ESM-IF ddG, or Nebius `stab-ddg`, on the full VH.
"""
import re

# penalties subtracted from 1.0 (tunable)
W_HYDROPHOBIC_RUN = 0.15  # per [FILVWY] run >= 4 (aggregation)
W_CHARGE_RUN = 0.10       # per [KR] or [DE] run >= 3 (charge patch)
W_CYS = 0.30              # per cysteine (unpaired thiol)
W_PRO = 0.05              # per proline beyond the first
W_GLY = 0.10              # if glycine fraction > GLY_FRAC
W_NET_CHARGE = 0.10       # per |net charge| unit beyond NET_CHARGE_MAX
W_ANCHOR = 0.0            # disabled: "starts W/ends D" overfit to trastuzumab (would ding valid de-novo loops)
GLY_FRAC, NET_CHARGE_MAX = 0.4, 2
TRASTUZUMAB = "WGGDGFYAMD"


def stability_flags(cdrh3):
    s = cdrh3.upper()
    net = sum(s.count(c) for c in "KR") - sum(s.count(c) for c in "DE")
    return {
        "hydrophobic_runs": len(re.findall(r"[FILVWY]{4,}", s)),
        "charge_runs": len(re.findall(r"[KR]{3,}|[DE]{3,}", s)),
        "cys": s.count("C"),
        "extra_pro": max(0, s.count("P") - 1),
        "gly_rich": s.count("G") / max(1, len(s)) > GLY_FRAC,
        "net_charge": net,
        "lost_anchors": int(not s.startswith("W")) + int(not s.endswith("D")),
    }


def stability_score(cdrh3):
    f = stability_flags(cdrh3)
    p = (W_HYDROPHOBIC_RUN * f["hydrophobic_runs"] + W_CHARGE_RUN * f["charge_runs"]
         + W_CYS * f["cys"] + W_PRO * f["extra_pro"] + W_GLY * f["gly_rich"]
         + W_NET_CHARGE * max(0, abs(f["net_charge"]) - NET_CHARGE_MAX)
         + W_ANCHOR * f["lost_anchors"])
    return max(0.0, min(1.0, 1.0 - p))


def demo():
    cases = [TRASTUZUMAB, "CCCCWWWWKK", "GGGGGGSGGD", "WPPPYAMD", "WGKRKRKRMD", "WGGDGFYAMD"[::-1]]
    for c in cases:
        print(f"{c:12s} {stability_score(c):.2f} {stability_flags(c)}")
    assert stability_score(TRASTUZUMAB) >= 0.9
    assert stability_score("CCCCWWWWKK") <= 0.2
    assert stability_score("CCCCWWWWKK") < stability_score("WGGDGFYAMD")


if __name__ == "__main__":
    demo()
