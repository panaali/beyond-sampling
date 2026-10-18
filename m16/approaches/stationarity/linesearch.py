"""T310 check: is the readout gradient real? Paired expected loss/accuracy along W_o - eta G (G from stat.json), 2^20 fresh
inputs (rng 3101), and an autograd cross-check of G on one batch. Diagnostic. usage: python linesearch.py DEV2_WEIGHTS.npz name"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, torch as th
z = np.load(sys.argv[1], allow_pickle=False); names = list(map(str, z["names"])); nm = sys.argv[2]; i = names.index(nm)
Wi, Wh, Wo = z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float)
X = np.random.default_rng(3101).standard_normal((2 ** 20, 10)); y = np.argsort(X, 1)[:, -2]; h = np.zeros((len(X), 16))
for t in range(10):
    h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
def ev(W):
    L = h @ W.T; L = L - L.max(1, keepdims=True); lp = L - np.log(np.exp(L).sum(1, keepdims=True))
    return -lp[np.arange(len(y)), y].mean(), (L.argmax(1) == y).mean()
p = np.exp(ev(Wo) and (h @ Wo.T - (h @ Wo.T).max(1, keepdims=True))); p /= p.sum(1, keepdims=True); E = p.copy(); E[np.arange(len(y)), y] -= 1
G = E.T @ h / len(y)
# autograd check on first 1024
Wt = th.tensor(Wo, requires_grad=True); loss = th.nn.functional.cross_entropy(th.tensor(h[:1024]) @ Wt.T, th.tensor(y[:1024])); loss.backward()
Gb = (E[:1024].T @ h[:1024]) / 1024; print("autograd max diff", float(np.abs(Wt.grad.numpy() - Gb).max()), "||G||", np.linalg.norm(G))
for eta in (0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
    l, a = ev(Wo - eta * G); print(f"eta {eta:g}: loss {l:.5f} acc {a:.5f}")
