"""ARC's Hermite-corrected cumulant propagation (arXiv:2605.05179, code: vendor/mlp_kprop) applied to the
AlgZoo 2nd-argmax RNN M_H,T.

The RNN plus its 0/1 "correct" read-out is rewritten as a feed-forward network of linear maps and coordinatewise
activations (identity / ReLU / Heaviside), the form ARC's algorithm takes ("treat the loss function as an additional
activation function", paper Sec. 2.2). ARC's own linear and non-linear steps (`linear_kprop`, `nonlin_kprop` from the
vendored library, unmodified) are then applied layer by layer, and the final first cumulant is the estimate.

Two ways to express accuracy (see README, Method):
  mode "uncond": one network on state (h, x_0..x_{T-1}); x enters as pass-through (identity) coordinates.
                 Read-out computes 1[x_j is 2nd largest] AND 1[argmax logits = j] from pairwise Heavisides.
  mode "cell":   condition on (p1, p2) = positions of the largest and 2nd largest input. x | cell is non-Gaussian
                 but its cumulants are model-free (precomputed). Read-out is 1[l_p2 > l_i for all i != p2].
                 accuracy = sum over the T(T-1) cells of P(cell) * estimate(cell), P(cell) = 1/(T(T-1)).

FLOPs: counted with ARC's own counter (vendor/mlp_kprop/flop_utils.NamedFlopCounter), raw count (no symmetric-tensor
discount, i.e. as implemented). Nothing here draws random model inputs.
"""
import itertools
import logging
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
for p in (os.path.join(ROOT, "data/arc_kprop_deps"), os.path.join(HERE, "vendor")):
    if p not in sys.path:
        sys.path.insert(0, p)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from numpy.polynomial import Polynomial  # noqa: E402

from mlp_kprop import kprop_harmonic as KH  # noqa: E402
from mlp_kprop.wick import relu_wick_coef, heaviside_wick_coef, poly_wick_coef  # noqa: E402
from mlp_kprop.flop_utils import NamedFlopCounter, flop_name  # noqa: E402

logging.getLogger("mlp_kprop").setLevel(logging.ERROR)
for _n in list(logging.root.manager.loggerDict):
    if _n.startswith("mlp_kprop"):
        logging.getLogger(_n).setLevel(logging.ERROR)
torch.set_grad_enabled(False)
torch.set_num_threads(1)
DT = torch.float64

ID, RELU, HEAV = 0, 1, 2
_ident = Polynomial([0, 1])


def mixed_wick(codes):
    """1-D Hermite ("Wick") coefficients E[d^k phi_i(Z_i)^p], Z_i ~ N(mean_i, var_i), with phi_i chosen per unit."""
    codes = torch.as_tensor(codes)
    masks = {c: codes == c for c in (ID, RELU, HEAV) if bool((codes == c).any())}

    def f(mean, var, k, p=1):
        out = torch.zeros_like(mean)
        for c, m in masks.items():
            if c == ID:
                v = poly_wick_coef(_ident, mean, var, k, p)
            elif c == RELU:
                v = relu_wick_coef(mean, var, k, p)
            else:
                v = heaviside_wick_coef(mean, var, k, p)
            v = torch.as_tensor(v, dtype=mean.dtype).expand_as(mean)
            out = torch.where(m, v, out)
        return out
    return f


# ------------------------------------------------------------------ the network as linear + coordinatewise layers
def rnn_layers(model):
    """T layers on state (h, x): h <- ReLU(Wh h + Wi x_t), x <- x (identity)."""
    Wi, Wh, Wo = model
    H, T = Wh.shape[0], Wo.shape[0]
    L = []
    for t in range(T):
        W = np.zeros((H + T, H + T))
        W[:H, :H] = Wh
        W[:H, H + t] = Wi
        W[H:, H:] = np.eye(T)
        L.append((W, None, [RELU] * H + [ID] * T))
    return L


def readout_uncond(model):
    """Layers mapping (h_T, x) to 1[correct] using Heavisides only."""
    Wi, Wh, Wo = model
    H, T = Wh.shape[0], Wo.shape[0]
    pairs = [(i, j) for i in range(T) for j in range(i + 1, T)]
    P = len(pairs)
    # R1: U_l[ij] = 1[l_j > l_i], U_x[ij] = 1[x_i > x_j]   (i < j). Logit ties go to the smaller index, as np.argmax
    #     (T42 audit finding 1); input ties have probability 0.
    W1 = np.zeros((2 * P, H + T))
    for a, (i, j) in enumerate(pairs):
        W1[a, :H] = Wo[j] - Wo[i]
        W1[P + a, H + i], W1[P + a, H + j] = 1.0, -1.0
    # R2: a_j = #{i: l_j > l_i}, b_j = #{i: x_i > x_j};  A_j = 1[a_j = T-1], G1_j = 1[b_j >= 1], G2_j = 1[b_j >= 2]
    W2 = np.zeros((3 * T, 2 * P)); b2 = np.zeros(3 * T)
    for a, (i, j) in enumerate(pairs):
        # j beats i  <=> U_l[ij];  i beats j (incl. ties) <=> 1 - U_l[ij]
        W2[j, a] += 1                      # a_j += U_l
        W2[i, a] -= 1; b2[i] += 1          # a_i += 1 - U_l
        # x_i > x_j counts for b_j: +U_x;  x_j > x_i counts for b_i: 1 - U_x
        for off in (T, 2 * T):
            W2[off + j, P + a] += 1
            W2[off + i, P + a] -= 1; b2[off + i] += 1
    b2[:T] -= (T - 1) - 0.5
    b2[T:2 * T] -= 0.5
    b2[2 * T:] -= 1.5
    # R3: C_j = 1[A_j + G1_j - G2_j >= 2]
    W3 = np.zeros((T, 3 * T)); b3 = -1.5 * np.ones(T)
    for j in range(T):
        W3[j, j], W3[j, T + j], W3[j, 2 * T + j] = 1, 1, -1
    W4 = np.ones((1, T))
    return [(W1, None, [HEAV] * (2 * P)), (W2, b2, [HEAV] * (3 * T)), (W3, b3, [HEAV] * T), (W4, None, None)]


def readout_cell(model, p2):
    """Layers mapping (h_T, x) to 1[argmax l = p2] with np.argmax's tie rule: l_p2 > l_i for i < p2 and
    l_p2 >= l_i for i > p2 (the latter as 1 - 1[l_i > l_p2]; T42 audit finding 1)."""
    Wi, Wh, Wo = model
    H, T = Wh.shape[0], Wo.shape[0]
    others = [i for i in range(T) if i != p2]
    W1 = np.zeros((T - 1, H + T))
    W2 = np.ones((1, T - 1)); b2 = np.array([-(T - 1) + 0.5])
    for a, i in enumerate(others):
        if i < p2:
            W1[a, :H] = Wo[p2] - Wo[i]
        else:
            W1[a, :H] = Wo[i] - Wo[p2]
            W2[0, a] = -1.0; b2[0] += 1.0
    return [(W1, None, [HEAV] * (T - 1)), (W2, b2, [HEAV]), (np.ones((1, 1)), None, None)]


def forward_layers(layers, s):
    """Exact evaluation of the layer stack on states s (N, D); used only by the self-test."""
    for W, b, act in layers:
        z = s @ W.T + (0 if b is None else b)
        if act is None:
            return z
        act = np.array(act)
        s = np.where(act == RELU, np.maximum(z, 0), np.where(act == HEAV, (z > 0).astype(float), z))
    return s


# ------------------------------------------------------------------ ARC's propagation over a layer stack
def propagate(layers, K_in, k_max, kind="SIMPLE"):
    """ARC basic (SIMPLE) / augmented (AUGMENT) cumulant propagation; returns final mean vector (numpy)."""
    kind = getattr(KH, kind)
    skip_metric = not KH.metric_needed(k_max, kind)
    with flop_name("coerce"):
        K = KH.coerce_input({d: torch.as_tensor(v, dtype=DT) for d, v in K_in.items()}, k_max=k_max, kind=kind)
    for W, b, act in layers:
        Wt = torch.as_tensor(W, dtype=DT)
        bt = None if b is None else torch.as_tensor(b, dtype=DT)
        if act is None:
            K = KH.linear_kprop(K, Wt, k_max=k_max, d_max=1, set_metric=1.0 if skip_metric else None, bias=bt)
            return K[1].to_tensor().numpy()
        K = KH.linear_kprop(K, Wt, k_max=k_max, set_metric=1.0 if skip_metric else None, bias=bt)
        K = KH.nonlin_kprop(K, nonlin_wick_coef=mixed_wick(act), k_max=k_max, kind=kind)
    raise ValueError("layer stack must end with a linear output layer (act=None)")


def counted(fn, *a, **kw):
    """Run fn under ARC's FLOP counter. Returns (result, raw_flops, symmetric_adjusted_flops)."""
    with NamedFlopCounter() as c:
        out = fn(*a, **kw)
    return out, c.raw_total(), c.total()


# ------------------------------------------------------------------ input cumulants
def uncond_input(H, T, k_max):
    D = H + T
    K = {1: np.zeros(D), 2: np.zeros((D, D))}
    K[2][H:, H:] = np.eye(T)
    return K


_CELL_CACHE = {}


def cell_base_cumulants(T, order=4, n=4_000_000, seed=12345, full_sym=True):
    """Model-free cumulants (orders 1..order) of x ~ N(0, I_T) conditioned on x_0 > x_1 > max(x_2..x_{T-1}).

    Exact sampler (every draw used: sort, put the two largest at positions 0, 1, the rest in random order), followed by
    symmetrisation over the T-2 exchangeable coordinates. Cached on disk under data/. This is precomputation that does
    not depend on the model (leaderboard rule 4)."""
    key = (T, order, n, seed, full_sym)
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    assert n % 200_000 == 0, "n must be a multiple of the 200k chunk (T42 finding 3)"
    path = os.path.join(ROOT, f"data/arc_cumulants_cellK_T{T}_o{order}_n{n}_s{seed}{'_fullsym' if full_sym else ''}.npz")
    if os.path.exists(path):
        z = np.load(path)
        out = {d: z[f"k{d}"] for d in range(1, order + 1)}
        _CELL_CACHE[key] = out
        return out
    rng = np.random.default_rng(seed)
    acc = {d: 0.0 for d in range(1, order + 1)}
    chunk = 200_000
    for _ in range(n // chunk):
        x = np.sort(rng.standard_normal((chunk, T)), axis=1)[:, ::-1]
        rest = x[:, 2:].copy()
        rng.permuted(rest, axis=1, out=rest)
        y = np.concatenate([x[:, :2], rest], 1)
        acc[1] = acc[1] + y.sum(0)
        acc[2] = acc[2] + y.T @ y
        y2 = (y[:, :, None] * y[:, None, :]).reshape(chunk, T * T)
        if order >= 3:
            acc[3] = acc[3] + (y2.T @ y).reshape(T, T, T)
        if order >= 4:
            acc[4] = acc[4] + (y2.T @ y2).reshape(T, T, T, T)
    m = {d: acc[d] / n for d in acc}
    # raw moments -> cumulants (orders <= 4)
    m1, m2 = m[1], m[2]
    out = {1: m1, 2: m2 - np.einsum("i,j->ij", m1, m1)}
    if order >= 3:
        m3 = m[3]
        sym3 = lambda A: (A + A.transpose(1, 2, 0) + A.transpose(2, 0, 1))
        out[3] = m3 - sym3(np.einsum("ij,k->ijk", m2, m1)) + 2 * np.einsum("i,j,k->ijk", m1, m1, m1)
    if order >= 4:
        m4 = m[4]
        def symsum(A):          # sum over all distinct index permutations of a 4-tensor (then divided below)
            return sum(np.transpose(A, p) for p in itertools.permutations(range(4)))
        e31 = np.einsum("ijk,l->ijkl", m3, m1)
        e22 = np.einsum("ij,kl->ijkl", m2, m2)
        e211 = np.einsum("ij,k,l->ijkl", m2, m1, m1)
        e1111 = np.einsum("i,j,k,l->ijkl", m1, m1, m1, m1)
        out[4] = (m4 - symsum(e31) / 6 - symsum(e22) / 8 + 2 * symsum(e211) / 4 - 6 * e1111)
    # symmetrise over exchangeable coordinates 2..T-1
    if full_sym:   # all (T-2)! permutations of the exchangeable coordinates (T42 finding 3)
        perms = [[0, 1] + [2 + i for i in p] for p in itertools.permutations(range(T - 2))]
    else:          # dihedral subgroup only (the version used for the first leaderboard runs)
        perms = [list(range(2)) + list(np.roll(np.arange(2, T), s)) for s in range(T - 2)]
        perms += [list(range(2)) + list(np.roll(np.arange(2, T)[::-1], s)) for s in range(T - 2)]
    for d in out:
        A = out[d]
        out[d] = sum(A[np.ix_(*[pp] * d)] for pp in perms) / len(perms)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{f"k{d}": out[d] for d in out})
    _CELL_CACHE[key] = out
    return out


def cell_input(H, T, p1, p2, k_max, base, kind="SIMPLE"):
    """Joint cumulants of the state (h_0 = 0, x) for cell (p1, p2): base cumulants with indices permuted."""
    D = H + T
    perm = [p1, p2] + [i for i in range(T) if i not in (p1, p2)]   # base coordinate a lives at position perm[a]
    inv = np.empty(T, int); inv[perm] = np.arange(T)
    dmax = KH.get_d_max(k_max, getattr(KH, kind))     # T42 finding 2: was hard-coded SIMPLE
    K = {}
    for d in range(1, min(dmax, max(base)) + 1):
        A = np.zeros((D,) * d)
        A[(slice(H, None),) * d] = base[d][np.ix_(*[inv] * d)]
        K[d] = A
    return K


# ------------------------------------------------------------------ estimators
def estimate_uncond(model, k_max, kind="SIMPLE"):
    H, T = model[1].shape[0], model[2].shape[0]
    layers = rnn_layers(model) + readout_uncond(model)
    out, raw, adj = counted(propagate, layers, uncond_input(H, T, k_max), k_max, kind)
    return float(out[0]), raw, adj


def estimate_cell(model, k_max, kind="SIMPLE", base=None, per_cell=False):
    H, T = model[1].shape[0], model[2].shape[0]
    if base is None:
        base = cell_base_cumulants(T, order=min(4, KH.get_d_max(k_max, getattr(KH, kind))))
    rnn = rnn_layers(model)
    tot, raw_t, adj_t, cells = 0.0, 0, 0, {}
    for p1 in range(T):
        for p2 in range(T):
            if p1 == p2:
                continue
            layers = rnn + readout_cell(model, p2)
            out, raw, adj = counted(propagate, layers, cell_input(H, T, p1, p2, k_max, base, kind), k_max, kind)
            cells[(p1, p2)] = float(out[0])
            tot += float(out[0]) / (T * (T - 1)); raw_t += raw; adj_t += adj
    return (tot, raw_t, adj_t, cells) if per_cell else (tot, raw_t, adj_t)


def propagate_trace(layers, K_in, k_max, kind="SIMPLE"):
    """Like propagate, but returns the estimated (mean, variance-diagonal or None) after every activation layer."""
    kind = getattr(KH, kind)
    skip_metric = not KH.metric_needed(k_max, kind)
    K = KH.coerce_input({d: torch.as_tensor(v, dtype=DT) for d, v in K_in.items()}, k_max=k_max, kind=kind)
    out = []
    for W, b, act in layers:
        Wt = torch.as_tensor(W, dtype=DT)
        bt = None if b is None else torch.as_tensor(b, dtype=DT)
        if act is None:
            K = KH.linear_kprop(K, Wt, k_max=k_max, d_max=1, set_metric=1.0 if skip_metric else None, bias=bt)
            out.append((K[1].to_tensor().numpy(), None))
            return out
        K = KH.linear_kprop(K, Wt, k_max=k_max, set_metric=1.0 if skip_metric else None, bias=bt)
        K = KH.nonlin_kprop(K, nonlin_wick_coef=mixed_wick(act), k_max=k_max, kind=kind)
        var = K[2].to_tensor().diagonal().numpy().copy() if 2 in K else None
        out.append((K[1].to_tensor().numpy().copy(), var))
    return out
