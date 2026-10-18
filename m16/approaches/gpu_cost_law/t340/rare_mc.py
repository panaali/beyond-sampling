"""T340 GPU MC truths for the rare-event test (request of session_01HPuYWfHz2C9jVqL15Q2rZ1, pre-registered at
claude/t286-cheap-frontier @716ea8b5 m16/approaches/rare_events/README.md). float64 forward, x ~ N(0, I_10),
h_t = relu(Wh h_{t-1} + Wi x_t), pick = argmax(Wo h_10); v1 > v2 > v3 top 3 of x at p1, p2, p3; g12 = v1 - v2,
g23 = v2 - v3. Counts: E1(t0) = pick == p3 & g23 > t0 (t0 = 0, 1, 1.25, 1.5); E2(t) = pick != p2 & g12 >= t & g23 >= t
(t = .5, .75); wrong = pick != p2. One CUDA Philox generator per model, seed 3400000 + index (orig 0, seed0 1, seed3 2).
usage: python rare_mc.py weights.npz NAME:N ... -> rare_mc.json"""
import json, sys, time
import numpy as np, torch as th
z = np.load(sys.argv[1]); out = {}
for k, arg in enumerate(sys.argv[2:]):
    nm, n = arg.split(":"); n = int(float(n)); i = list(z["names"]).index(nm)
    Wi, Wh, Wo = (th.as_tensor(z[a][i], dtype=th.float64, device="cuda") for a in ("Wi", "Wh", "Wo"))
    seed = 3400000 + i; g = th.Generator(device="cuda"); g.manual_seed(seed); t0 = time.time()
    c = dict(E1_0=0, E1_1=0, E1_1p25=0, E1_1p5=0, E2_0p5=0, E2_0p75=0, wrong=0); done = 0
    while done < n:
        m = min(1 << 24, n - done); x = th.randn(m, 10, generator=g, device="cuda", dtype=th.float64)
        h = th.zeros(m, 16, dtype=th.float64, device="cuda")
        for t in range(10):
            h = th.relu(h @ Wh.T + x[:, t:t + 1] * Wi[None])
        pick = (h @ Wo.T).argmax(1); v, p = x.topk(3, 1); g12 = v[:, 0] - v[:, 1]; g23 = v[:, 1] - v[:, 2]
        wrong = pick != p[:, 1]; at3 = pick == p[:, 2]
        for key, t0_ in (("E1_0", 0.0), ("E1_1", 1.0), ("E1_1p25", 1.25), ("E1_1p5", 1.5)):
            c[key] += int((at3 & (g23 > t0_)).sum())
        for key, t_ in (("E2_0p5", 0.5), ("E2_0p75", 0.75)):
            c[key] += int((wrong & (g12 >= t_) & (g23 >= t_)).sum())
        c["wrong"] += int(wrong.sum()); done += m
    out[nm] = dict(n=n, seed=seed, generator="torch CUDA Philox, randn float64, chunks of 2^24", precision="float64",
                   counts=c, sec=time.time() - t0)
    print(nm, json.dumps(out[nm]), flush=True)
    json.dump(out, open("rare_mc.json", "w"), indent=1)
