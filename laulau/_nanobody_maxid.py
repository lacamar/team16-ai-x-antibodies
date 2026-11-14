"""Throwaway: max global-alignment identity of our designs vs 125 known VHH refs (RCSB).

Vectorised NW reproducing denovo_rank.identity() exactly (match+1/mismatch-1/gap-0.5,
matches-along-best-path / len(query)); validated against the reference python DP below.
"""
import json, random
import numpy as np
from her2_iglm import denovo_div as dv
from her2_iglm import data as d
from her2_iglm.tokenizer import IgLMTokenizer

OFF, MM = 4000, 4096  # key = (2*score + OFF)*MM + match  (exact ordering: score then match)


def ident_ref(q, r):  # reference (denovo_rank.identity), for validation
    n, m = len(q), len(r)
    S = [[(0.0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): S[i][0] = (-0.5 * i, 0)
    for j in range(1, m + 1): S[0][j] = (-0.5 * j, 0)
    for i in range(1, n + 1):
        Si, Sm = S[i], S[i - 1]; qi = q[i - 1]
        for j in range(1, m + 1):
            eq = qi == r[j - 1]; dg = Sm[j - 1]
            Si[j] = max((dg[0] + (1 if eq else -1), dg[1] + eq),
                        (Sm[j][0] - 0.5, Sm[j][1]), (Si[j - 1][0] - 0.5, Si[j - 1][1]))
    return S[n][m][1] / n


def batch_maxident(queries, refs):
    """Return (len(queries),) max identity to any ref, vectorised over all pairs."""
    A = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", np.uint8)
    code = {c: k + 1 for k, c in enumerate(A.tobytes().decode())}  # 0 reserved = never-equal pad
    def enc(s, L):
        a = np.zeros(L, np.int16);
        for k, c in enumerate(s): a[k] = code.get(c, 0)
        return a
    Nmax = max(len(q) for q in queries); Mmax = max(len(r) for r in refs)
    R = len(refs)
    rmat = np.stack([enc(r, Mmax) for r in refs])            # (R, Mmax)
    rlen = np.array([len(r) for r in refs])
    out = np.empty(len(queries))
    for qi, q in enumerate(queries):
        n = len(q); qarr = enc(q, n)
        # score2 (=2*score, int), match (int): previous & current rows, shape (R, Mmax+1)
        js = np.arange(Mmax + 1)
        ps2 = (-js).astype(np.int64)[None, :].repeat(R, 0)   # row 0: score=-0.5*j -> score2=-j
        pm = np.zeros((R, Mmax + 1), np.int64)
        res = np.zeros(R, np.int64); rown = np.zeros(R, np.int64)  # match at (n, rlen)
        for i in range(1, n + 1):
            eqrow = (rmat == qarr[i - 1])                    # (R, Mmax) bool
            # diag = S[i-1][j-1] + (+1/-1 ; +eq); up = S[i-1][j] - 0.5
            dg_s2 = ps2[:, :-1] + np.where(eqrow, 2, -2)
            dg_m = pm[:, :-1] + eqrow
            up_s2 = ps2[:, 1:] - 1
            up_m = pm[:, 1:]
            dk = (dg_s2 + OFF) * MM + dg_m
            uk = (up_s2 + OFF) * MM + up_m
            Ak = np.maximum(dk, uk)                          # A[j] for j=1..Mmax
            A_s2 = Ak // MM - OFF; A_m = Ak % MM
            # include col 0 (k=0): score2=-i, match 0
            col0_s2 = np.full((R, 1), -i, np.int64); col0_m = np.zeros((R, 1), np.int64)
            s2_all = np.concatenate([col0_s2, A_s2], 1)      # (R, Mmax+1), index j
            m_all = np.concatenate([col0_m, A_m], 1)
            t_key = ((s2_all + np.arange(Mmax + 1)) + OFF) * MM + m_all   # t = score2 + j
            run = np.maximum.accumulate(t_key, axis=1)       # running max over k<=j
            cur_s2 = (run // MM - OFF) - np.arange(Mmax + 1)  # subtract j
            cur_m = run % MM
            ps2, pm = cur_s2, cur_m
            hit = rown == 0  # not yet recorded; record pairs whose query length == i
            if i == n:
                res = cur_m[np.arange(R), rlen]
        out[qi] = (res / n).max()
    return out


def main():
    rng = random.Random(0)
    tok = IgLMTokenizer()
    refs = [l.strip() for l in open("_nanobody_refs.fasta") if l.strip() and not l.startswith(">")]
    # validate vectorised vs reference on 40 random pairs
    gen_all = [json.loads(l)["vh"] for l in open("runs/denovo_gen_div/generated.jsonl")]
    vq = rng.sample(gen_all, 8)
    mx = 0
    for q in vq:
        ref_vals = []
        # per-ref exact via batch of single query
        bv = batch_maxident([q], refs)[0]
        rv = max(ident_ref(q, r) for r in refs)
        mx = max(mx, abs(bv - rv))
    print(f"VALIDATION max|vectorised-reference| over 8 queries x125 refs = {mx:.2e}")

    gen = gen_all[:2000]
    g = batch_maxident(gen, refs)
    sp = dv.load_diverse_splits("data/processed/diverse_splits.npz")
    idx = rng.sample(range(sp.train_tokens.shape[0]), 2000)
    hosts = [tok.decode_residues(sp.train_tokens[i][2:int(sp.train_length[i]) - 1].tolist()) for i in idx]
    h = batch_maxident(hosts, refs)
    for name, a in (("GENERATED", g), ("TRAINING HOSTS", h)):
        print(f"{name} (n={len(a)}): max-identity to any of 125 known VHH  "
              f"mean {a.mean():.4f}  median {np.median(a):.4f}  overall-max {a.max():.4f}  "
              f"frac<0.70 {(a < 0.70).mean():.4f}  (#>=0.70: {(a>=0.70).sum()})")


if __name__ == "__main__":
    main()
