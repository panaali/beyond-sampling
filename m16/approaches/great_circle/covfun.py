"""Covariance of the correct-indicator along a random great circle, C(D), and the efficiency of exact
integration over an arc of length alpha (cost = 1 + rate*alpha pieces), vs plain sampling."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/great_circle")
from common import Wi, Wh, Wo
from arc_eval import forward
d = np.load("pilot_0.npz"); f = d["f"]
print("circle fraction quantiles", np.quantile(f, [0, .01, .05, .25, .5, .75, .95]).round(4))
print("share of var from circles with f<0.85:", ((f - f.mean())**2 * (f < .85)).sum() / ((f - f.mean())**2).sum())
M = (Wi, Wh, Wo); G = 2048; nc = 300
th = np.arange(G) / G * 2 * np.pi
I = np.empty((nc, G))
for i in range(nc):
    x = np.cos(th)[:, None] * d["U"][i] + np.sin(th)[:, None] * d["V"][i]
    I[i] = forward(M, x).argmax(1) == np.argsort(x, 1)[:, -2]
p = I.mean(); Ic = I - p
F = np.fft.rfft(Ic, axis=1); C = np.fft.irfft((F * F.conj()).mean(0), n=G) / G   # circular autocov
np.save("cov.npy", C)
rate = d["p"].mean() / (2 * np.pi)
print(f"p={p:.4f} C0={C[0]:.4f} rate={rate:.1f} pieces/rad")
for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    a = k * 2 * np.pi / G
    # discrete arc average variance with k grid points (approximates exact arc integral)
    lag = np.arange(-k + 1, k)
    v = ((k - np.abs(lag)) * C[np.mod(lag, G)]).sum() / k**2
    cost = 1 + rate * a
    print(f"arc {np.degrees(a):7.1f} deg  var {v:.3e}  cost {cost:6.1f}  gain {C[0]/(v*cost):.3f}")
