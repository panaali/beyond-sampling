"""Optimised samplers (float64, one thread): compiled (numba) forward + 2nd-argmax check, and plain numpy/BLAS.
truth: python sampler.py truth  -> truths.json (1e8 iid N(0,1) inputs per model, compiled sampler)."""
import json
import sys
import zlib

import numpy as np
from numba import njit


@njit(cache=True, error_model="numpy")
def count_correct(Wi, Wh, Wo, x):
    n, T = x.shape
    H = Wh.shape[0]; O = Wo.shape[0]
    h = np.zeros(H); z = np.zeros(H)
    ok = 0
    for q in range(n):
        for i in range(H):
            h[i] = 0.0
        for s in range(T):
            xs = x[q, s]
            for i in range(H):
                acc = 0.0
                for j in range(H):
                    acc += h[j] * Wh[i, j]
                z[i] = acc + xs * Wi[i]
            for i in range(H):
                h[i] = z[i] if z[i] > 0 else 0.0
        best = -np.inf; kb = 0
        for o in range(O):
            acc = 0.0
            for j in range(H):
                acc += h[j] * Wo[o, j]
            if acc > best:              # first maximiser, as numpy argmax
                best = acc; kb = o
        # index of the 2nd largest input
        i1 = 0
        for s in range(1, T):
            if x[q, s] > x[q, i1]:
                i1 = s
        i2 = 1 if i1 == 0 else 0
        for s in range(T):
            if s != i1 and x[q, s] > x[q, i2]:
                i2 = s
        if kb == i2:
            ok += 1
    return ok


@njit(cache=True, error_model="numpy", fastmath=False)
def count_correct_blk(Wi, Wh, Wo, xT, B=64):
    """Same result as count_correct, SIMD across a block of B sequences (inputs given transposed, (T, n))."""
    T, n = xT.shape
    H = Wh.shape[0]; O = Wo.shape[0]
    h = np.zeros((H, B)); z = np.zeros((H, B)); L = np.zeros((O, B))
    ok = 0
    for q0 in range(0, n, B):
        m = min(B, n - q0)
        h[:, :] = 0.0
        for s in range(T):
            for i in range(H):
                xi = Wi[i]
                for q in range(m):
                    z[i, q] = 0.0
                for j in range(H):
                    w = Wh[i, j]
                    for q in range(m):
                        z[i, q] += h[j, q] * w
                for q in range(m):
                    z[i, q] += xT[s, q0 + q] * xi
            for i in range(H):
                for q in range(m):
                    h[i, q] = z[i, q] if z[i, q] > 0 else 0.0
        for o in range(O):
            for q in range(m):
                L[o, q] = 0.0
            for j in range(H):
                w = Wo[o, j]
                for q in range(m):
                    L[o, q] += h[j, q] * w
        for q in range(m):
            best = L[0, q]; kb = 0
            for o in range(1, O):
                if L[o, q] > best:
                    best = L[o, q]; kb = o
            i1 = 0
            for s in range(1, T):
                if xT[s, q0 + q] > xT[i1, q0 + q]:
                    i1 = s
            i2 = 1 if i1 == 0 else 0
            for s in range(T):
                if s != i1 and xT[s, q0 + q] > xT[i2, q0 + q]:
                    i2 = s
            if kb == i2:
                ok += 1
    return ok


# exploratory: same kernel with reassociation/FMA allowed (symmetric to walk_nb's fastmath variant)
count_correct_fm = njit(error_model="numpy", fastmath={"reassoc", "contract", "nsz", "arcp"})(count_correct_blk.py_func)


def est_compiled(model, n, rng, chunk=65536, x=None):
    """Sampling estimate with n inputs; normals drawn by numpy's Generator (ziggurat) unless x (T, n) is given."""
    Wi, Wh, Wo = model; T = Wo.shape[0]; ok = 0
    if x is not None:
        return count_correct_blk(Wi, Wh, Wo, x) / x.shape[1]
    for i in range(0, n, chunk):
        ok += count_correct_blk(Wi, Wh, Wo, rng.standard_normal((T, min(chunk, n - i))))
    return ok / n


def est_numpy(model, n, rng, chunk=16384):
    Wi, Wh, Wo = model; T = Wo.shape[0]; ok = 0
    for i in range(0, n, chunk):
        x = rng.standard_normal((min(chunk, n - i), T))
        h = np.zeros((len(x), Wh.shape[0]))
        for s in range(T):
            h = np.maximum(h @ Wh.T + x[:, s:s + 1] * Wi[None], 0)
        ok += int(((h @ Wo.T).argmax(1) == np.argsort(x, 1)[:, -2]).sum())
    return ok / n


if __name__ == "__main__" and sys.argv[1] == "truth":
    import models as Mo
    out = Mo.truths(); N = 10 ** 8
    for T in (5, 10):
        for name, m in Mo.groups(T).items():
            key = f"{name}:T{T}"
            if key in out:
                continue
            rng = np.random.default_rng(zlib.crc32(key.encode()))
            p = est_compiled(m, N, rng, chunk=1 << 20)
            out[key] = {"acc": p, "se": float(np.sqrt(p * (1 - p) / N)), "n": N}
            print(key, out[key], flush=True)
            json.dump(out, open(f"{Mo.HERE}/truths.json", "w"), indent=1)
