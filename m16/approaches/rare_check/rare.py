"""T349: own implementation of T340's rare-event estimators (variant B for E1, variant A for E2) using my T312 walk.
Deductive: no random inputs. Cost charged as executed (walk cells per T312 + Phi at weight-dependent endpoints)."""
import sys
import numpy as np
from itertools import permutations
from scipy.integrate import quad
from scipy.special import ndtr

sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/r1w_repro")
import r1w_indep as W  # noqa: E402  (my T312 exact segment walk)
import cont_e as CE    # noqa: E402  (T317e's model-free region constants, for variant A)

T, TEND, G = 10, 6.0, 20
TRI = np.array(list(permutations(range(T), 3))); NT = len(TRI)
FREE = np.array([[p for p in range(T) if p not in t] for t in TRI]); P4, LOWS = FREE[:, -1], FREE[:, :-1]
phi = lambda v: np.exp(-v * v / 2) / np.sqrt(2 * np.pi)
mu = lambda v: -phi(v) / ndtr(v)
dmu = lambda v: -v * mu(v) + mu(v) ** 2
FPE, F_PHI = 5948.0, 20.0
_x, _w = np.polynomial.legendre.leggauss(G)
NODES = {"nb": (3.25 + 3.25 * _x, 3.25 * _w), "b": (-3.25 + 3.25 * _x, 3.25 * _w)}


def line23(v2, g12, v3, v4, m4, dm4):
    """Rows for all triples: x(t) = X0 + t D with v3 = v3 - t... built at t measured from v3 = v2 (as T317)."""
    delta = v3 - v4
    X0 = np.zeros((NT, T)); D = np.zeros((NT, T)); r = np.arange(NT)
    X0[r, TRI[:, 0]] = v2 + g12; X0[r, TRI[:, 1]] = v2; X0[r, TRI[:, 2]] = v2; D[r, TRI[:, 2]] = -1
    X0[r, P4] = v2 - delta; D[r, P4] = -1
    low0 = m4 + dm4 * (v2 - delta - v4)
    for j in range(6):
        X0[r, LOWS[:, j]] = low0; D[r, LOWS[:, j]] = -dm4
    return X0, D


def node_point(v2):
    g12 = (phi(v2) - v2 * (1 - ndtr(v2))) / (1 - ndtr(v2))
    g23 = quad(lambda v: (ndtr(v) / ndtr(v2)) ** 8, v2 - 14, v2, epsabs=1e-13)[0]
    v3 = v2 - g23
    v4 = v3 - quad(lambda u: (ndtr(u) / ndtr(v3)) ** 7, v3 - 14, v3, epsabs=1e-13)[0]
    return g12, v3, v4, mu(v4), dmu(v4)


def mass23(v2, ivs):
    return sum((ndtr(v2 - a) ** 8 - ndtr(v2 - b) ** 8) / 8 for a, b in ivs)


def clip(ivs, lo, hi=TEND):
    return [(max(a, lo), min(b, hi)) for a, b in ivs if min(b, hi) > max(a, lo)]


def complement(ivs, lo, hi=TEND):
    out, cur = [], lo
    for a, b in sorted(clip(ivs, lo, hi)):
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        out.append((cur, hi))
    return out


def walk(model, X0, D, label):
    W.MAX_CELLS = 100_000
    ints, cost, _ = W.walk(model, X0, D, label, tmax=TEND)
    return ints, cost


def est_E1(model, t0s=(1.0, 1.25, 1.5, 0.0)):
    """Variant B: one walk per triple per GL node; argmax = p3 intervals; full g12 mass."""
    out = {t0: 0.0 for t0 in t0s}; fpe = 0.0
    for r in ("nb", "b"):
        for v2, w in zip(*NODES[r]):
            g12, v3, v4, m4, dm4 = node_point(v2)
            X0, D = line23(v2, g12, v3, v4, m4, dm4)
            ints, c = walk(model, X0, D, TRI[:, 2]); fpe += c
            nend = sum(2 * len(iv) for iv in ints); fpe += nend * F_PHI / FPE
            for t0 in t0s:
                out[t0] += w * phi(v2) * (1 - ndtr(v2)) * sum(mass23(v2, clip(iv, t0)) for iv in ints)
    return out, fpe


def est_E2(model, ts=(0.5, 0.75)):
    """Variant A: one walk per triple per region at T317e's region point; wrong = complement of argmax = p2."""
    out = {t: 0.0 for t in ts}; fpe = 0.0
    for tab in CE.TABLES:
        r = "nb" if tab["name"].startswith("nb") else "b"
        v2s, g12s, g23s = tab["point"]
        X0, D = line23(v2s, g12s, v2s - g23s, tab["v4"], tab["m4"], tab["dm4"])
        ints, c = walk(model, X0, D, TRI[:, 1]); fpe += c
        for t in ts:
            wr = [complement(iv, t) for iv in ints]
            fpe += G * sum(2 * len(x) for x in wr) * F_PHI / FPE
            for v2, w in zip(*NODES[r]):
                out[t] += w * phi(v2) * (1 - ndtr(v2 + t)) * sum(mass23(v2, x) for x in wr)
    return out, fpe
