"""T352 (DRI): tie analysis for the discrete-input cells. Label convention (training and evaluation alike): 2nd-largest
position of the DISCRETISED input, ties broken by the lower index. A label is tie-affected if the 2nd-largest value
equals the largest or the 3rd-largest (v1 == v2 or v2 == v3). Reports the tie rate, accuracy on all / tie-free / tied
inputs (N float64 inputs on CPU, seed 3522000 + index; same inputs for all models of the same grid).
usage: python qties.py CKPT.pt [N] -> ties.json"""
import json, math, os, sys
import torch as th
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from qeval import q_inf, disc, label  # noqa: E402
f = sys.argv[1]; N = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10 ** 7
d = th.load(f, weights_only=False); out = {}
for k, sp in enumerate(d["meta"]["spec"]):
    if sp["grid"] == 0 and not sp["name"].startswith("int4_s0"):
        continue
    W = [d[a][k].double() for a in ("Wi", "Wh", "Wo")]; s = float(d["s"][k]); g = th.Generator(); g.manual_seed(3522000 + sp["grid"])
    n = ok = nt = okt = 0
    while n < N:
        m = min(100000, N - n); x = disc(th.randn(m, 10, generator=g, dtype=th.float64), sp["grid"]); v = x.sort(1, descending=True).values
        tie = (v[:, 0] == v[:, 1]) | (v[:, 1] == v[:, 2]); h = th.zeros(m, W[1].shape[0], dtype=th.float64)
        for t in range(10):
            h = q_inf(h @ W[1].T + x[:, t:t + 1] * W[0][None], sp["act"], s)
        c = (h @ W[2].T).argmax(1) == label(x)
        n += m; ok += int(c.sum()); nt += int(tie.sum()); okt += int((c & tie).sum())
    out[sp["name"]] = dict(grid=sp["grid"], act=sp["act"], n=N, acc=ok / N, tie_rate=nt / N, acc_tie_free=(ok - okt) / (N - nt),
                           acc_tied=okt / max(nt, 1))
    print(sp["name"], {a: round(b, 5) for a, b in out[sp["name"]].items() if isinstance(b, float)}, flush=True)
json.dump(out, open(os.path.join(HERE, "ties.json"), "w"), indent=1)
