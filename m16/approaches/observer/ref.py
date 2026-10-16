"""DIAGNOSTIC ONLY (sampling): per-triple error rates on the b>0 region and the blind-region accuracy."""
import sys
import numpy as np
sys.path.insert(0, '/home/user/clean-neurons/m16')
from arc_eval import forward
from core import T
from common import Wi, Wh, Wo


def reference(model, n=4_000_000, seed=7, chunk=500_000):
    rng = np.random.default_rng(seed)
    cnt = np.zeros((T, T, T)); err = np.zeros((T, T, T)); e1 = np.zeros((T, T, T)); e3 = np.zeros((T, T, T)); nb = 0; eb = 0
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, T)); o = np.argsort(x, 1)
        out = forward(model, x).argmax(1); ok = out == o[:, -2]
        b = x[np.arange(chunk), o[:, -2]]
        pos = b > 0
        np.add.at(cnt, (o[pos, -1], o[pos, -2], o[pos, -3]), 1)
        np.add.at(err, (o[pos, -1], o[pos, -2], o[pos, -3]), ~ok[pos])
        idx = (o[pos, -1], o[pos, -2], o[pos, -3])
        np.add.at(e1, idx, out[pos] == o[pos, -1]); np.add.at(e3, idx, out[pos] == o[pos, -3])
        nb += (~pos).sum(); eb += (~ok[~pos]).sum()
    return dict(cnt=cnt, err=err, e1=e1, e3=e3, nb=nb, eb=eb, n=n)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.path.insert(0, '.'); from arc_eval import noisy_models
        sig, k = float(sys.argv[1]), int(sys.argv[2])
        m = noisy_models(6, sig)[k]; r = reference(m, n=2_000_000); np.savez(f'ref_s{sig}_{k}.npz', **r)
    else:
        r = reference((Wi, Wh, Wo)); np.savez('ref_orig.npz', **r)
    print('P(b<=0)', r['nb'] / r['n'], 'blind err', r['eb'] / r['nb'], 'err b>0 share', r['err'].sum() / r['n'])
