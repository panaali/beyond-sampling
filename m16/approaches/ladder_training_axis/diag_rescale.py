"""T101 hypothesis test (dev noise seed 0): is seed 4's T75 frozen-gate error caused by neuron 0's large scale
inflating the 'surely-on' threshold? Function-preserving rescaling of neuron j by c (incoming Wi[j], Wh[j,:] x c;
outgoing Wh[:,j], Wo[:,j] / c; ReLU is positively homogeneous) leaves the network and truth unchanged; only T75's
impact bookkeeping changes. usage: python diag_rescale.py k s j c1,c2,...  [n=64]"""
import sys
import numpy as np
import lt
from zoo43 import noisy_copies


def rescale(m, j, c):
    Wi, Wh, Wo = (w.copy() for w in m)
    Wi[j] *= c; Wh[j, :] *= c; Wh[:, j] /= c; Wo[:, j] /= c
    return Wi, Wh, Wo


k, s, j = map(int, sys.argv[1:4]); cs = [float(x) for x in sys.argv[4].split(",")]; n = int(sys.argv[5]) if len(sys.argv) > 5 else 64
ms = noisy_copies(lt.zoo(k, s), n, lt.SIGMA, 0)
tr = np.array([lt.truth(m)[0] for m in ms]); pq = np.mean(tr * (1 - tr))
for c in cs:
    r = [lt.t75(rescale(m, j, c)) for m in ms]
    tr2 = lt.truth(rescale(ms[0], j, c))[0]
    e = np.array([a for a, _, _ in r]) - tr; om = np.mean([b for _, b, _ in r])
    print(f"k={k} s={s} neuron {j} x{c}: truth unchanged {abs(tr2 - tr[0]) < 1e-12}; T75 ratio {np.mean(e**2)/(pq/(om/72)):.4f} "
          f"rms {100*np.sqrt(np.mean(e**2)):.2f}% ops {om:.0f}", flush=True)
