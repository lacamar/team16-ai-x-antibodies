"""Fast sequence-level reward for the IgLM fine-tuning loop — NO folding.

`SeqReward().reward(cdrh3, vh=None)` returns a scalar (higher = better) in milliseconds,
so you can score thousands of IgLM generations per round and use it as the RL reward.

WHY sequence-level and not fold confidence: we measured ipTM vs real SPR affinity at
Spearman 0.285 (nearly blind), and folding is ~75 s/design on one GPU — far too slow and
too weak to be the reward. The affinity surrogate is trained on the SAME 524k FACS + 768 SPR
data you fine-tune IgLM on (Spearman 0.72), so it is the right reward. Fold + epitope_check
are the SLOW, high-precision gate you run on the final shortlist only (see rank.py).

Reward = surrogate (primary) - stability penalty, with novelty + developability as HARD gates
(reward = GATED_OUT if a design is non-novel or has a developability red flag).
"""
import novelty_filter as nf
from affinity_surrogate import AffinitySurrogate, DATA
from thermostability import stability_score

GATED_OUT = -10.0
W_STABILITY = 0.75


class SeqReward:
    def __init__(self):
        self.sur = AffinitySurrogate().fit()
        self.by_len = nf.load_provided_cdrh3(
            DATA / "tz_her2_facs_3classes_524346.csv.zip",
            DATA / "trastuzumab_her2_spr_all_768.csv")

    def reward(self, cdrh3: str, vh: str | None = None) -> dict:
        s = self.sur.score(cdrh3)
        stab = stability_score(cdrh3)
        novel, why = nf.is_novel(cdrh3, self.by_len)
        hard_ok, hard_why = (nf.hard_checks(vh) if vh else (True, "vh not given"))
        r = s["surrogate"] - W_STABILITY * (1.0 - stab)
        gated = (not novel) or (not hard_ok)
        return {
            "reward": GATED_OUT if gated else round(r, 3),
            "surrogate": s["surrogate"], "pkd": s["pkd"], "p_high": s["p_high"],
            "stability": stab, "novel": novel, "ood": s["ood"],
            "gate": "ok" if not gated else (why if not novel else hard_why),
        }

    def batch(self, cdrh3s, vhs=None):
        vhs = vhs or [None] * len(cdrh3s)
        return [self.reward(c, v) for c, v in zip(cdrh3s, vhs)]


def demo():
    rw = SeqReward()
    print("fit done; scoring examples (higher reward = better; gated = -10):")
    for name, c in [("trastuzumab (known: gated)", "WGGDGFYAMD"),
                    ("novel, best of random 200", "FLPQDAYHMG"),
                    ("novel, weaker", "NDWQEAGAYV")]:
        print(f"  {name:28} {rw.reward(c)}")
    # a known binder must gate out; a novel one must not
    assert rw.reward("WGGDGFYAMD")["reward"] == GATED_OUT
    assert rw.reward("KTRSYELDPM")["reward"] != GATED_OUT
    print("self-check passed")


if __name__ == "__main__":
    demo()
