"""T340 bug check: compare a walk's correct set along each line with a dense forward evaluation of the network.
usage: python check_walk.py WALK tmax  (WALK = t317f | walk3) -> mismatch statistics per (model, region, line)"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "cheap_frontier")); sys.path.insert(0, os.path.join(HERE, "..", "closure_floor"))
import cf, rare
import cont_f_t317f as F
G = rare.G


def dense_correct(model, x0, d, k, tmax, n=20001):
    Wi, Wh, Wo = model; ts = np.linspace(0, tmax, n); out = np.zeros(len(x0))
    for i in range(len(x0)):
        X = x0[i][None] + ts[:, None] * d[i][None]; h = np.zeros((n, 16))
        for t in range(10):
            h = np.maximum(h @ Wh.T + np.outer(X[:, t], Wi), 0)
        ok = (h @ Wo.T).argmax(1) == k[i]
        out[i] = ok.mean() * tmax
    return out


def walk_len(pieces, n):
    L = np.zeros(n)
    for rows, lo, hi in pieces:
        np.add.at(L, rows, np.maximum(hi - lo, 0))
    return L


which, tmax = sys.argv[1], float(sys.argv[2])
rows = np.arange(0, 720, 24)                                  # 30 rows per line
for name in ("orig", "seed0"):
    m = tuple(np.asarray(w, float) for w in cf.lb3_models()[name])
    for tab in F.TABLES:
        for kind in ("23", "12"):
            if f"line{kind}" not in tab:
                continue
            x, dx = tab[f"line{kind}"]; k = rare.TRI[:, 1]
            if which == "t317f":
                p, _ = F.walk_c1(m, x[rows], dx[rows], k[rows], tmax)     # includes screen C1 (declared)
                p2, _ = F.walk_cont(m, x[rows], dx[rows], k[rows], tmax)
                Lw, Lw2 = walk_len(p, len(rows)), walk_len(p2, len(rows))
            else:
                p, pc, *_ = rare.walk3(m, x[rows], dx[rows], k[rows], rare.TRI[rows, 2], tmax)
                Lw = walk_len(pc, len(rows)); Lw2 = Lw
            Ld = dense_correct(m, x[rows], dx[rows], k[rows], tmax)
            e1, e2 = np.abs(Lw - Ld), np.abs(Lw2 - Ld)
            print(f"{which} {name} {tab['name']}-{kind}: |walk - dense| correct length mean {e2.mean():.4f} max {e2.max():.4f}"
                  + (f"  (with C1: mean {e1.mean():.4f} max {e1.max():.4f})" if which == "t317f" else "") + f"  [grid step {tmax/20000:.1e}]", flush=True)
