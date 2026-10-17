"""T113 review helpers, written for this review (generic H, T = 3): weights straight from alg_zoo (asserting no
biases), ARC-protocol noise (absolute N(0, sigma^2) on every weight; draw order Wi, Wh, Wo per copy), the
leaderboard's relative noise (sigma x RMS of each matrix), own forward pass, own MC truth, and an own exact
enumeration (`exact_mine`) that is built differently from arc_m43_repro/exact.py:
  * start cones are the 8 octants (not the 12 ordering x chart cones),
  * pieces are split by the step-2 and step-3 gate rows, then each final linear region is split by the label
    (median) chambers and the logit-difference planes,
  * solid angles by L'Huilier's spherical-excess formula (not Van Oosterom-Strackee).
It also counts linear regions (distinct gate patterns) and the pieces of the regions inside the 24 half-chambers
(the partition the recipe works in), which is the natural floor for any method that resolves every gate."""
import itertools
import math

import numpy as np

E = np.eye(3)


def zoo_w(H, seed, T=3):
    import alg_zoo
    m = alg_zoo.zoo_2nd_argmax(H, T, seed=seed)
    sd = {k: v.detach().numpy().astype(float) for k, v in m.state_dict().items()}
    assert set(sd) == {"rnn.weight_ih_l0", "rnn.weight_hh_l0", "linear.weight"}, sd.keys()   # no biases
    return sd["rnn.weight_ih_l0"][:, 0].copy(), sd["rnn.weight_hh_l0"].copy(), sd["linear.weight"].copy()


def noisy_abs(base, k, sigma, seed):
    g = np.random.default_rng(seed); Wi, Wh, Wo = base; out = []
    for _ in range(k):
        a = Wi + sigma * g.standard_normal(Wi.shape)
        b = Wh + sigma * g.standard_normal(Wh.shape)
        c = Wo + sigma * g.standard_normal(Wo.shape)
        out.append((a, b, c))
    return out


def noisy_rel(base, k, sigma, seed):
    """m16/leaderboard.py convention: each matrix gets N(0, (sigma * RMS(matrix))^2) per entry."""
    g = np.random.default_rng(seed); out = []
    for _ in range(k):
        out.append(tuple(W + sigma * math.sqrt(float((W ** 2).mean())) * g.standard_normal(W.shape) for W in base))
    return out


def fwd(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((x.shape[0], len(Wi)))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, [t]] * Wi, 0.0)
    return h @ Wo.T


def mc(model, n, seed, chunk=500_000):
    g = np.random.default_rng(seed); ok = 0; done = 0
    while done < n:
        m = min(chunk, n - done); x = g.standard_normal((m, 3))
        ok += int((fwd(model, x).argmax(1) == np.argsort(x, 1)[:, 1]).sum()); done += m
    return ok / n


def rescale_perm(model, rng, span=4.0):
    """Function-preserving: neuron k scaled by c_k = exp(U(-span, span)) (input weight and incoming recurrent row
    x c_k, outgoing recurrent column and readout column / c_k), then a random permutation."""
    Wi, Wh, Wo = model; H = len(Wi)
    c = np.exp(rng.uniform(-span, span, H)); P = rng.permutation(H)
    return (Wi * c)[P], (Wh * c[:, None] / c[None, :])[np.ix_(P, P)], (Wo / c[None, :])[:, P]


def split_neuron(model, k, n):
    """Function-preserving split of neuron k into n identical copies, each carrying 1/n of its outgoing weights
    (recurrent column and readout column). Incoming weights are copied. H grows by n - 1."""
    Wi, Wh, Wo = model; H = len(Wi)
    idx = list(range(H)) + [k] * (n - 1)
    Wi2 = Wi[idx].copy()
    Wh2 = Wh[np.ix_(idx, idx)].copy()
    cols = [k] + list(range(H, H + n - 1))
    Wh2[:, cols] /= n
    Wo2 = Wo[:, idx].copy(); Wo2[:, cols] /= n
    return Wi2, Wh2, Wo2


# ------------------------------------------------------------------ own spherical geometry (independent code)
def _clip(poly, n):
    s = [float(np.dot(n, v)) for v in poly]
    if min(s) >= 0:
        return poly, []
    if max(s) <= 0:
        return [], poly
    a, b = [], []
    m = len(poly)
    for i in range(m):
        u, w = poly[i], poly[(i + 1) % m]; su, sw = s[i], s[(i + 1) % m]
        if su >= 0:
            a.append(u)
        if su <= 0:
            b.append(u)
        if su * sw < 0:
            p = su * w - sw * u
            if su < 0:
                p = -p
            p = p / np.linalg.norm(p)
            a.append(p); b.append(p)
    return (a if len(a) >= 3 else []), (b if len(b) >= 3 else [])


def _side(u, w):
    return math.atan2(float(np.linalg.norm(np.cross(u, w))), float(np.dot(u, w)))


def girard(poly):
    """Area of a convex spherical polygon / (4 pi): fan triangulation, each triangle by L'Huilier's theorem from its
    side lengths (spherical excess; robust to duplicate vertices). Independent of exact.py's Van Oosterom-Strackee."""
    n = len(poly)
    if n < 3:
        return 0.0
    tot = 0.0; a0 = poly[0]
    for i in range(1, n - 1):
        b, c = poly[i], poly[i + 1]
        x, y, z = _side(b, c), _side(a0, c), _side(a0, b)
        s = 0.5 * (x + y + z)
        t = math.tan(s / 2) * math.tan(max(s - x, 0) / 2) * math.tan(max(s - y, 0) / 2) * math.tan(max(s - z, 0) / 2)
        tot += 4 * math.atan(math.sqrt(max(t, 0.0)))
    return tot / (4 * math.pi)


def _split(poly, rows):
    pieces = [(poly, ())]
    for r in rows:
        nxt = []
        for p, g in pieces:
            a, b = _clip(p, r)
            if a:
                nxt.append((a, g + (1,)))
            if b:
                nxt.append((b, g + (0,)))
        pieces = nxt
    return pieces


def _octants():
    out = []
    for s0, s1, s2 in itertools.product((1.0, -1.0), repeat=3):
        out.append((s0, [s0 * E[0], s1 * E[1], s2 * E[2]]))
    return out


LABEL_CHAMBERS = []      # (class c, normals of the ordering chamber in which x_c is the median)
for c in range(3):
    a, b = [o for o in range(3) if o != c]
    LABEL_CHAMBERS.append((c, [E[a] - E[c], E[c] - E[b]]))
    LABEL_CHAMBERS.append((c, [E[b] - E[c], E[c] - E[a]]))


def exact_mine(model, stats=None):
    Wi, Wh, Wo = model
    acc = 0.0; pats = set()
    for s0, poly in _octants():
        g1 = (s0 * Wi > 0).astype(float)
        A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1])
        for p2, g2 in _split(poly, A2):
            A3 = Wh @ (np.array(g2, float)[:, None] * A2) + np.outer(Wi, E[2])
            for p3, g3 in _split(p2, A3):
                pats.add((s0, g2, g3))
                g3a = np.array(g3, float)
                L = Wo @ (g3a[:, None] * A3)
                for c, nrm in LABEL_CHAMBERS:
                    q = p3
                    for nn in nrm:
                        q, _ = _clip(q, nn)
                        if not q:
                            break
                    if not q:
                        continue
                    if g3a.sum() == 0:
                        if c == 0:
                            acc += girard(q)
                        continue
                    for o in range(3):
                        if o != c and q:
                            q, _ = _clip(q, L[c] - L[o])
                    if q:
                        acc += girard(q)
    if stats is not None:
        stats["regions"] = len(pats)
    return acc


def halves_mine():
    """The 24 half-chambers (ordering chamber x chart sign(x0), split by x_max + x_min = 2 x_med)."""
    out = []
    for s0 in (1.0, -1.0):
        for mx, md, mn in itertools.permutations(range(3)):
            R = np.linalg.inv(np.array([E[mx] - E[md], E[md] - E[mn], s0 * E[0]]))
            poly = [R[:, i] / np.linalg.norm(R[:, i]) for i in range(3)]
            split = E[mx] + E[mn] - 2 * E[md]
            for nn in (-split, split):
                p, _ = _clip(poly, nn)
                out.append((s0, p))
    return out


HALVES_MINE = halves_mine()


def half_pieces(model, min_frac=0.0):
    """Number of nonempty (half-chamber, linear region) pieces = cells an exact resolve-everything pass over the
    24 half-chambers must visit (gate rows only, before readout clipping)."""
    Wi, Wh, Wo = model; n = 0
    for s0, poly in HALVES_MINE:
        g1 = (s0 * Wi > 0).astype(float)
        A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1])
        for p2, g2 in _split(poly, A2):
            A3 = Wh @ (np.array(g2, float)[:, None] * A2) + np.outer(Wi, E[2])
            for p3, g3 in _split(p2, A3):
                if min_frac <= 0 or girard(p3) > min_frac:
                    n += 1
    return n
