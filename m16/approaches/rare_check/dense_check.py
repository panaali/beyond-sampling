"""T350 check (DRI): does my T312 walk handle simultaneous zero crossings? Dense forward check of the set lengths along
sampled E1/B lines and E2/A lines: max |walk length - grid length| over rows."""
import json
import numpy as np
import rare

NG = 50_000


def grid_len(model, X0, D, label, tend=rare.TEND):
    Wi, Wh, Wo = model; t = (np.arange(NG) + 0.5) * tend / NG; out = []
    for x0, d, lab in zip(X0, D, label):
        X = x0[None] + t[:, None] * d[None]; h = np.zeros((NG, 16))
        for s in range(10):
            h = np.maximum(h @ Wh.T + X[:, s:s + 1] * Wi[None], 0)
        out.append((((h @ Wo.T).argmax(1)) == lab).mean() * tend)
    return np.array(out)


def check(model, X0, D, label, rows):
    ints, _ = rare.walk(model, X0[rows], D[rows], label[rows])
    wl = np.array([sum(b - a for a, b in iv) for iv in ints])
    gl = grid_len(model, X0[rows], D[rows], label[rows])
    return float(np.abs(wl - gl).max()), float(np.abs(wl - gl).mean())


res = {}
rng = np.random.default_rng(350)
for path, nm in (("/home/user/dl/dev3/t349_weights.npz", "dev3_s505"), ("/home/user/dl/t340/weights.npz", "seed0")):
    z = np.load(path, allow_pickle=True); i = list(z["names"]).index(nm)
    m = tuple(np.asarray(z[k][i], float) for k in ("Wi", "Wh", "Wo"))
    r = {}
    for reg in ("nb", "b"):                                # E1/B lines at GL nodes nearest v2 = 0 and two others
        nodes = rare.NODES[reg][0]
        for v2 in sorted(nodes, key=abs)[:2] + [nodes[len(nodes) // 2]]:
            X0, D = rare.line23(v2, *rare.node_point(v2))
            rows = rng.choice(rare.NT, 60, replace=False)
            r[f"E1B_{reg}_v2={v2:.3f}"] = check(m, X0, D, rare.TRI[:, 2], rows)
    for tab in rare.CE.TABLES:                             # E2/A lines (label p2)
        v2s, g12s, g23s = tab["point"]
        X0, D = rare.line23(v2s, g12s, v2s - g23s, tab["v4"], tab["m4"], tab["dm4"])
        rows = rng.choice(rare.NT, 60, replace=False)
        r[f"E2A_{tab['name']}"] = check(m, X0, D, rare.TRI[:, 1], rows)
    res[nm] = r
    print(nm, json.dumps(r, indent=1), flush=True)
json.dump(res, open("results/dense_check.json", "w"), indent=1)
