"""T417: independent high-precision reference for the M5t per-simplex quantity bounded by m5t_fast.bounds_batch.

Written from the formula (rigour._bounds_tangent docstring and T412_README.md), not from m5t_fast's code.
All arithmetic is python-flint arb at PREC >= 256 bits, on the exact rational vertices.

ref_bounds(...) returns two rigorous intervals [HI_lo, HI_hi] and [LO_lo, LO_hi] (exact rationals):
  HI = the M5t upper-bound formula evaluated exactly (same tangent centre mc and float z-bin edges as fast),
  LO = the M5t lower-bound formula evaluated exactly.
By T400's derivation LO <= Q <= HI, Q = c0 int_simplex int_{z>=t0} z^(d-1) exp(-z^2 |m|^2 / 2) Phi(z y_p3)^mlow.
So fast hi < LO_lo or fast lo > HI_hi is a soundness violation; fast hi < HI_lo (1 - tol) is a 'shortfall' (fast's
float evaluation of the same formula came out below the exact value: a gap in its rounding analysis).

E_unif exp(-u) over a simplex, u affine with vertex values t: E = n! e^-T e[s_0..s_n], s_i = T - t_i, T = max t
(Hermite-Genocchi), and e[s] = exp(Z)_{0n} for the bidiagonal Z = diag(s) + superdiag(1) (Opitz). exp(Z) is
flint's rigorous arb_mat.exp (its own Taylor / scaling-squaring error bounds), which handles coincident knots."""
import math

from flint import arb, arb_mat, ctx, fmpq

PREC = 256


def Q(x):
    """Exact rational of a Python float."""
    return fmpq(*float(x).as_integer_ratio())


def aq(x):
    return arb(Q(x)) if isinstance(x, float) else arb(x)


def log_phi(x):
    """log Phi(x) for an arb x."""
    return ((-x) / arb(2).sqrt()).erfc().log() - arb(2).log()


def logE(t):
    """Enclosure of log E_unif exp(-u), u affine with vertex values t (arb balls; E decreasing in each knot, so a
    ball knot gives a ball result). Uses s_i = T - t_i >= 0 with T the max of the knot midpoints."""
    V = len(t)
    n = V - 1
    if n == 0:
        return -t[0]
    T = max(t, key=lambda v: v.mid()).mid()
    Z = arb_mat(V, V)
    for i in range(V):
        Z[i, i] = T - t[i]
        if i + 1 < V:
            Z[i, i + 1] = 1
    e = Z.exp()[0, n]
    return -T + (e * math.factorial(n)).log()


def G(d, s, a, b):
    """int_a^b z^(d-1) exp(-z^2 s / 2) dz, s > 0 exact (fmpq), a, b fmpq (b None = inf). Returns arb."""
    s = arb(s); h = arb(d) / 2
    ga = (arb(a) ** 2 * s / 2).gamma_upper(h)
    gb = arb(0) if b is None else (arb(b) ** 2 * s / 2).gamma_upper(h)
    return (2 / s) ** h / 2 * (ga - gb)


def _lo(x):
    return x.lower()


def _hi(x):
    return x.upper()


def ref_bounds(M, vol, mc, d, mlow, jp3, edges, prune=fmpq(1, 10 ** 30)):
    """M: V x d exact rationals (fmpq) on the slice; vol: fmpq; mc: d floats (the tangent centre given to fast);
    edges: the float z-bin edges of fast (Params.edges). Returns (HI_lo, HI_hi, LO_lo, LO_hi) as arbs (exact
    endpoints) and the number of bins evaluated exactly."""
    old = ctx.prec
    ctx.prec = PREC
    try:
        return _ref(M, vol, mc, d, mlow, jp3, edges, prune)
    finally:
        ctx.prec = old


def _ref(M, vol, mc, d, mlow, jp3, edges, prune):
    V = len(M)
    mq = [Q(x) for x in mc]
    sc = sum(x * x for x in mq)
    dots = [sum(a * b for a, b in zip(mq, Mi)) for Mi in M]
    tau = [2 * (dt - sc) for dt in dots]
    T0 = min(tau)
    tsh = [t - T0 for t in tau]
    R2 = max(sum((a - b) ** 2 for a, b in zip(Mi, mq)) for Mi in M)
    smax = max(sum(a * a for a in Mi) for Mi in M)
    L = min(dots)
    smin = fmpq(1, 2)
    if L > 0 and sc > 0 and L * L / sc > smin:
        smin = L * L / sc
    y = [Mi[jp3] for Mi in M]
    ylo, yhi = min(y), max(y)
    neg = max(fmpq(0), -ylo)
    ypc = float(mc[jp3])
    eq = [Q(e) for e in edges]
    nb = len(edges) - 1
    zero = arb(0)
    # per-bin set-up and trivial (pruning) bounds
    bins = []
    for b in range(nb):
        A, B = eq[b], eq[b + 1]
        Af, Bf = float(edges[b]), float(edges[b + 1])
        llo, lhi = A * A / 2, B * B / 2
        base_u = -(lhi if T0 < 0 else llo) * T0
        base_l = -(llo if T0 < 0 else lhi) * T0
        Gu = G(d, sc, A, B) if sc > 0 else arb("inf")
        Gl = G(d, sc + R2, A, B)
        xl, xh = min(A * ylo, B * ylo), max(A * yhi, B * yhi)
        lsup = mlow * log_phi(arb(xh)) if mlow else zero
        triv = Gu * (arb(base_u) + lsup).exp()          # const variant with E <= e^-(min knot) = 1
        bins.append((b, A, B, Af, Bf, llo, lhi, base_u, base_l, Gu, Gl, xl, xh, lsup, triv))
    order = sorted(range(nb), key=lambda i: -float(bins[i][-1].upper()) if bins[i][-1].is_finite() else -math.inf)
    HI = [arb(0), arb(0)]; LO = [arb(0), arb(0)]
    # last bin [A, inf)
    A = eq[-1]
    sup_phi = arb(1) if (mlow == 0 or yhi >= 0) else (mlow * log_phi(arb(A * yhi))).exp()
    inf_phi = arb(1) if mlow == 0 else (arb(0) if ylo < 0 else (mlow * log_phi(arb(A * ylo))).exp())
    gh = G(d, smin, A, None) * sup_phi
    gl = G(d, smax, A, None) * inf_phi
    HI[0] += _lo(gh); HI[1] += _hi(gh); LO[0] += _lo(gl); LO[1] += _hi(gl)
    nex = 0
    rest = sum((bins[i][-1] for i in range(nb)), arb(0))
    for i in order:
        b, A, B, Af, Bf, llo, lhi, base_u, base_l, Gu, Gl, xl, xh, lsup, triv = bins[i]
        rest = rest - triv
        if nex > 0 and HI[0] > 0 and triv.is_finite() and triv.upper() + max(rest.upper(), arb(0)) <= prune * HI[0]:
            HI[1] += _hi(triv); LO[1] += _hi(triv)      # pruned: contributes [0, triv] to both
            continue
        nex += 1
        # ---- upper formula: min over the constant-Phi^m and tangent variants
        kc = [arb(llo * t) for t in tsh]
        eu_c = arb(base_u) + lsup + logE(kc)
        cands_u = [eu_c]
        if mlow:
            x0 = Q((0.5 * (Af + Bf)) * ypc)
            f0 = log_phi(arb(x0))
            r0 = (-arb(x0) ** 2 / 2 - (2 * arb.pi()).sqrt().log() - f0).exp()
            cu = mlow * (f0 - r0 * arb(x0) + r0 * arb(B - A) * arb(neg))
            kt = [arb(llo * t) - mlow * r0 * arb(B) * arb(yi) for t, yi in zip(tsh, y)]
            cands_u.append(arb(base_u) + cu + logE(kt))
        eu_lo = min(_lo(c) for c in cands_u); eu_hi = min(_hi(c) for c in cands_u)
        HI[0] += _lo(Gu * eu_lo.exp()); HI[1] += _hi(Gu * eu_hi.exp())
        # ---- lower formula: max over the constant-Phi^m and chord variants
        kl = [arb(lhi * t) for t in tsh]
        linf = mlow * log_phi(arb(xl)) if mlow else zero
        cands_l = [arb(base_l) + linf + logE(kl)]
        if mlow:
            L1 = log_phi(arb(xl))
            bet = (log_phi(arb(xh)) - L1) / arb(xh - xl) if xh > xl else arb(0)
            cl = mlow * (L1 - bet * arb(xl) - bet * arb(B - A) * arb(neg))
            kch = [arb(lhi * t) - mlow * bet * arb(A) * arb(yi) for t, yi in zip(tsh, y)]
            cands_l.append(arb(base_l) + cl + logE(kch))
        el_lo = max(_lo(c) for c in cands_l); el_hi = max(_hi(c) for c in cands_l)
        LO[0] += _lo(Gl * el_lo.exp()); LO[1] += _hi(Gl * el_hi.exp())
    c0 = (2 * arb.pi()) ** (-arb(d) / 2)
    f = c0 * arb(vol)
    return _lo(f * HI[0]), _hi(f * HI[1]), _lo(f * LO[0]), _hi(f * LO[1]), nex
