"""T101 diagnosis: which frozen step-3 neuron makes T75 err on zoo seed s (exact truth used only for attribution).
Variant 'T75 + always resolve step-3 neuron j': the frozen recipe (eps 0.1 at both steps) except that at step 3 neuron
j is resolved whenever it cuts a cell. A large drop in error for some j says that neuron's frozen gate carries the
error. Implemented on a private source-patched copy of the pinned est75_e37.py (one selection line changed).
DEV noise seed (default 0), never the 20-23 test panel.
usage: python diag_neuron.py k s [n=64] [noise_seed=0]"""
import os
import sys
import types

import numpy as np

import lt
from m43 import Ops
from zoo43 import noisy_copies

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "est75_e37.py")).read()
line = "            chosen = [k for d, k in cut if -d >= eps * scale][:K]; ops.add(2 * len(cut))\n"
assert src.count(line) == 1
src = src.replace(line, "            chosen = [k for d, k in cut if -d >= eps * scale or (eps == SENT and k in ALWAYS)][:K]; ops.add(2 * len(cut))\n"
                  "            eps = 0.1 if eps == SENT else eps\n")
D = types.ModuleType("est75_diag"); D.__file__ = "est75_e37.py [diag patch]"
exec(compile(src, D.__file__, "exec"), D.__dict__)
SENT = 0.1000000000001          # step-3 eps sentinel; numerically eps 0.1 for the threshold
D.SENT = SENT; D.ALWAYS = set()


def est(m, eps):
    o = Ops(); D.STATS.update(cut=0, resolved=0, cells=0)
    return D.est75(m, o, K=(4, 4), rule="adaptive", eps=eps), o.n


def main(k, s, n=64, ns=0):
    base = lt.zoo(k, s); ms = noisy_copies(base, n, lt.SIGMA, ns)
    tr = np.array([lt.truth(m)[0] for m in ms]); pq = np.mean(tr * (1 - tr))
    Wi, Wh, Wo = base
    print(f"k={k} s={s}: Wi {np.round(Wi, 2).tolist()}  Wo cols {np.round(Wo.T, 1).tolist()}")
    for alw in ([], [0], [1], [2], [3], [0, 1, 2, 3]):
        D.ALWAYS = set(alw)
        r = [est(m, (0.1, SENT)) for m in ms]; e = np.array([a for a, _ in r]) - tr; om = np.mean([b for _, b in r])
        print(f"   T75 + always resolve step-3 neurons {alw}: ratio {np.mean(e**2)/(pq/(om/72)):.4f} rms {100*np.sqrt(np.mean(e**2)):.2f}% "
              f"bias {100*e.mean():+.2f}% ops {om:.0f}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), int(a[2]) if len(a) > 2 else 64, int(a[3]) if len(a) > 3 else 0)
