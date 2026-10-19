"""T411 diagnostic: paired plain conditional-MC FD of P_W(2.0) along the pure-W direction (wonly.json) and along the
round-1 direction, vs the v1-line gradient's prediction. -> results/check_dW.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import design2 as D2  # noqa: E402
import truthc_copy as TCC  # noqa: E402
GW = D2.J("gW_r1.json"); S = np.array(D2.J("dir_r1.json")["S"]); g = np.array(GW["grad"])[S]
dirs = dict(pureW=np.array(D2.J("wonly.json")["d"]), round1=np.array(D2.J("dir_r1.json")["d"]))
draw = TCC.sampler(2.0); pg = TCC.p_gap(2.0); out = {}
for nm, d in dirs.items():
    for h in (2.5e-4, 1e-3):
        mp = [D2.W_ORIG.copy(), D2.W_ORIG.copy()]; mp[0][S] += h * d; mp[1][S] -= h * d; mp = [D2.rnn.unflat(v, D2.SH) for v in mp]
        rng = np.random.default_rng(411900); s1 = s2 = 0.0; N = 10_000_000; k0 = 0; m0 = D2.rnn.unflat(D2.W_ORIG, D2.SH)
        for _ in range(N // 250_000):
            x, p3 = draw(rng, 250_000); p2 = np.argsort(-x, 1)[:, 1]
            dd = (D2.rnn.forward(mp[0], x).argmax(1) != p2).astype(float) - (D2.rnn.forward(mp[1], x).argmax(1) != p2)
            s1 += dd.sum(); s2 += (dd ** 2).sum(); k0 += (D2.rnn.forward(m0, x).argmax(1) != p2).sum()
        mu = s1 / N; fd = pg * mu / (2 * h); se = pg * np.sqrt(s2 / N - mu ** 2) / np.sqrt(N) / (2 * h); P = pg * k0 / N
        out[f"{nm}@{h}"] = dict(fd_dlogP=fd / P, se=se / P, line_pred=float(g @ d / GW["P"]), P_plain=P)
        print(nm, h, out[f"{nm}@{h}"], flush=True)
        D2.J("check_dW.json", out)
