"""One round of reward-guided IgLM fine-tuning (round N of the loop).

generate (IgLM infill CDRH3) -> score (reward.SeqReward) -> select top fraction, gated (-10) dropped
-> FINE-TUNE IgLM on the survivors -> next round generates from the fine-tuned model.

Stack rounds: pass the previous round's out_dir as --model, and keep the best CDRH3s of
all rounds as the shortlist for the slow fold/epitope gate (rank.py).

Training format = IgLM's own: [HEAVY] [HUMAN] <VH with CDRH3 -> [MASK]> [SEP] <CDRH3> [CLS]
Loss is only on the infilled CDRH3 + [CLS] (the part we steer).

  python finetune_iglm.py --scored /tmp/iglm_run/scored.json --out /tmp/iglm_ft1 --steps 50
"""
import argparse, json, random, sys
import torch
from iglm import IgLM
from iglm.model.utils import mask_span
from transformers import GPT2LMHeadModel

VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAV"
      "YYCSRWGGDGFYAMDYWGQGTLVTVSS")
SPAN = (98, 108)
GATED = -10.0


def make_vh(cdr):
    return VH[:SPAN[0]] + cdr + VH[SPAN[1]:]


def load_iglm(model_dir=None):
    m = IgLM()
    if model_dir:
        m.model = GPT2LMHeadModel.from_pretrained(model_dir).to(m.device)
    return m


def generate(m, n, temperature=1.0):
    return m.infill(VH, "[HEAVY]", "[HUMAN]", SPAN, num_to_generate=n, top_p=1, temperature=temperature)


def score_vhs(rw, vhs):
    tail = len(VH) - SPAN[1]
    cdrs = [v[SPAN[0]:len(v) - tail] for v in vhs]
    return [dict(cdr=c, vh=v, **r) for c, v, r in zip(cdrs, vhs, rw.batch(cdrs, vhs))]


def select(scored, frac=0.25):
    ok = sorted((s for s in scored if s["reward"] != GATED), key=lambda s: -s["reward"])
    return [make_vh(s["cdr"]) for s in ok[:max(1, int(len(ok) * frac))]]


def encode(tok, vh, device):
    ids = tok.convert_tokens_to_ids(["[HEAVY]", "[HUMAN]"] + mask_span(list(vh), *SPAN_FOR(vh), append_span=True))
    ids = ids + [tok.cls_token_id]
    sep = ids.index(tok.sep_token_id)
    labels = [-100] * (sep + 1) + ids[sep + 1:]
    return torch.tensor([ids], device=device), torch.tensor([labels], device=device)


def SPAN_FOR(vh):
    return SPAN[0], len(vh) - (len(VH) - SPAN[1])


def finetune(sequences, steps, lr, out_dir, m=None):
    m = m or load_iglm()
    model, tok = m.model, m.tokenizer
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    data = [encode(tok, v, m.device) for v in sequences]
    for step in range(steps):
        x, y = random.choice(data)  # ponytail: batch size 1, pad+batch if throughput matters
        loss = model(input_ids=x, labels=y).loss
        loss.backward()
        opt.step(); opt.zero_grad()
        print(f"step {step + 1}/{steps} loss {loss.item():.4f}", flush=True)
    model.eval()
    model.save_pretrained(out_dir)
    return out_dir


def summary(scored):
    ok = [s for s in scored if s["reward"] != GATED]
    mean = sum(s["reward"] for s in ok) / max(1, len(ok))
    return dict(n=len(scored), pct_novel=round(100 * sum(s["novel"] for s in scored) / len(scored), 1),
                n_pass=len(ok), mean_reward_pass=round(mean, 3),
                best=round(max((s["reward"] for s in ok), default=GATED), 3))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--scored", default="/tmp/iglm_run/scored.json")
    a.add_argument("--model", help="previous round's out_dir (default: base IgLM)")
    a.add_argument("--out", default="/tmp/iglm_ft1")
    a.add_argument("--steps", type=int, default=50)
    a.add_argument("--lr", type=float, default=1e-5)
    a.add_argument("--frac", type=float, default=0.25)
    a.add_argument("--n", type=int, default=40, help="seqs to generate for before/after")
    a.add_argument("--max-train", type=int, default=0, help="cap fine-tune set (smoke)")
    a = a.parse_args()
    sys.path.insert(0, ".")
    from reward import SeqReward
    rw = SeqReward()
    try:
        before = json.load(open(a.scored))
        for s in before:
            s.setdefault("vh", make_vh(s["cdr"]))
    except FileNotFoundError:
        before = score_vhs(rw, generate(load_iglm(a.model), a.n))
    train = select(before, a.frac)
    if a.max_train:
        train = train[:a.max_train]
    print("before", summary(before), "train set", len(train))
    finetune(train, a.steps, a.lr, a.out, load_iglm(a.model))
    after = score_vhs(rw, generate(load_iglm(a.out), a.n))
    print("after ", summary(after))


if __name__ == "__main__":
    main()
