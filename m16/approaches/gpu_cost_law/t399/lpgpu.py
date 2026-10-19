"""T399: batched GPU LP relaxation engine for the certified rare-event branch and bound (INTERFACE.md v1 node format).

The relaxation is relax2.py's (T398): x in [-1, 1]^10, G x >= 0, x_p2 - x_p3 >= DELTA, the triangle relaxation of every
unstable ReLU of the remaining steps (exact rows for stable and phase-fixed neurons), and, for the event-aware variants,
the event rows L_p3 - L_j >= -EPS. Instead of HiGHS on the CPU, every LP of a batch of nodes is solved approximately by
diagonally preconditioned PDHG (Pock-Chambolle) in float32 on the GPU, and its value is CERTIFIED by weak duality in
float64: for min c.v s.t. K v >= b, lb <= v <= ub and ANY y >= 0,
    c.v >= b.y + sum_i min((c - K^T y)_i lb_i, (c - K^T y)_i ub_i).
So every reported bound is a valid lower bound whatever the PDHG accuracy (the accuracy only costs tightness); rounding of
the float64 evaluation is T400's job (it re-checks y exactly).

One LP shape for everything: variables v = (x (10), h_s (16) for the nb = 10 - k remaining steps, t), rows
  G x >= 0 (m, zero-padded), gap >= DELTA, per neuron 3 rows (h - z >= 0, upper triangle / h = z, phase sign), and
  L_p3 - L_j - t >= 0 (9 rows). t in [-TBIG, TBIG] makes the last rows slack (no event); t in [-EPS, TBIG] is the event;
  objective -t gives the relaxation's max margin max (L_p3 - max_j L_j) (negated).
Prune iff the certified upper bound of the max margin is < -EPS (or the domain is certified empty)."""
import math
import torch as th

D, H, T = 10, 16, 10
DELTA, EPS, TBIG = 0.1, 1e-7, 1e3


class LPEngine:
    def __init__(self, Wi, Wh, Wo, device="cuda", iters=400, dtype=th.float32, restart=100, restart_avg=True, ruiz=10, omega=1.0):
        f = lambda a: th.as_tensor(a, dtype=th.float64, device=device)
        self.Wi, self.Wh, self.Wo = f(Wi), f(Wh), f(Wo); self.dev, self.iters, self.dt = device, iters, dtype
        self.restart, self.restart_avg, self.ruiz, self.omega = restart, restart_avg, ruiz, omega

    # ---------------------------------------------------------------- LP construction (float64)
    def zrows(self, A, k):
        """pre-activation rows Z[s] (N, 16, nv) of the nb remaining steps (relax2.Relax.Z)."""
        N = A.shape[0]; nb = T - k; nv = D + H * nb + 1; Z = []
        prev = th.zeros(N, H, nv, dtype=th.float64, device=self.dev); prev[:, :, :D] = A
        for s in range(nb):
            Zs = th.einsum("ij,njv->niv", self.Wh, prev); Zs[:, :, k + s] += self.Wi
            Z.append(Zs); prev = th.zeros_like(prev); prev[:, :, D + H * s:D + H * (s + 1)] = th.eye(H, dtype=th.float64, device=self.dev)
        L = th.einsum("ij,njv->niv", self.Wo, prev)                    # prev = h_10 (or A x at k = 10)
        return Z, L

    def build(self, G, Z, L, lo, hi, phase, upto, p3):
        """K (N, R, nv), b (N, R), lb/ub (N, nv) with the relaxation of steps < upto (later h fixed to 0)."""
        N, m, _ = G.shape; nb = len(Z); nv = D + H * nb + 1; R = m + 1 + 3 * H * nb + (D - 1)
        K = th.zeros(N, R, nv, dtype=th.float64, device=self.dev); b = th.zeros(N, R, dtype=th.float64, device=self.dev)
        K[:, :m, :D] = G; K[:, m, self.tri[1]] = 1; K[:, m, self.tri[2]] = -1; b[:, m] = DELTA
        lb = th.zeros(N, nv, dtype=th.float64, device=self.dev); ub = th.zeros_like(lb)
        lb[:, :D] = -1; ub[:, :D] = 1; lb[:, -1] = -TBIG; ub[:, -1] = TBIG
        r = m + 1
        for s in range(upto):
            z = Z[s]; l, u = lo[:, s], hi[:, s]; ph = phase[:, s]
            l = th.where(ph > 0, l.clamp_min(0), l); u = th.where(ph < 0, u.clamp_max(0), u)
            act = l >= 0; inact = u <= 0; unst = ~(act | inact)
            hcol = D + H * s + th.arange(H, device=self.dev)
            eh = th.zeros(N, H, nv, dtype=th.float64, device=self.dev); eh[:, th.arange(H), hcol] = 1
            on = (~inact)[..., None].double()
            K[:, r:r + H] = (eh - z) * on                                 # h - z >= 0
            sl = th.where(unst, u / (u - l).clamp_min(1e-300), act.double())
            K[:, r + H:r + 2 * H] = (-eh + sl[..., None] * z) * on        # h <= sl (z - l)  (h <= z when active)
            b[:, r + H:r + 2 * H] = th.where(unst, sl * l, th.zeros_like(l))
            sg = th.where(ph > 0, 1.0, th.where(ph < 0, -1.0, 0.0)).double()
            K[:, r + 2 * H:r + 3 * H] = sg[..., None] * z                 # phase: +-z >= 0
            ub[:, hcol] = u.clamp_min(0) * (~inact).double()
            r += 3 * H
        K[:, r:r + D - 1] = (L[:, p3:p3 + 1] - L[:, [j for j in range(D) if j != p3]])
        K[:, r:r + D - 1, -1] = -1
        return K, b, lb, ub

    # ---------------------------------------------------------------- batched PDHG + float64 certificate
    def solve(self, K, b, lb, ub, c):
        """min c.v over the LP, c (N, Q, nv). Returns (N, Q) certified float64 lower bounds and the duals y (N, Q, R)."""
        dt = self.dt; dr = th.ones_like(b); dc = th.ones_like(lb); Ks = K
        for _ in range(self.ruiz):                                       # Ruiz equilibration: Ks = diag(dr) K diag(dc)
            rr = Ks.abs().amax(2).clamp_min(1e-12).rsqrt(); cr = Ks.abs().amax(1).clamp_min(1e-12).rsqrt()
            dr, dc = dr * rr, dc * cr; Ks = K * dr[..., None] * dc[:, None, :]
        Kf, bf, cf = Ks.to(dt), (b * dr).to(dt), (c * dc[:, None]).to(dt)  # scaled: v = dc v', y = dr y'
        lbf = (lb / dc).to(dt); ubf = (ub / dc).to(dt)
        tau = self.omega / Kf.abs().sum(1).clamp_min(1e-12)[:, None, :]  # (N, 1, nv) column sums
        sig = 1 / self.omega / Kf.abs().sum(2).clamp_min(1e-12)[:, None, :]  # (N, 1, R) row sums
        N, Q, nv = c.shape; R = K.shape[1]
        v = th.zeros(N, Q, nv, dtype=dt, device=self.dev).clamp(lbf[:, None], ubf[:, None]); y = th.zeros(N, Q, R, dtype=dt, device=self.dev)
        Kt = Kf.transpose(1, 2).contiguous()
        best = th.full((N, Q), -math.inf, dtype=th.float64, device=self.dev); ybest = y.double()
        va, ya, na = th.zeros_like(v), th.zeros_like(y), 0
        for it in range(self.iters):
            vn = th.minimum(th.maximum(v - tau * (cf - th.bmm(y, Kf)), lbf[:, None]), ubf[:, None])
            y = (y + sig * (bf[:, None] - th.bmm(2 * vn - v, Kt))).clamp_min(0); v = vn
            va += v; ya += y; na += 1
            if (it + 1) % self.restart == 0 or it == self.iters - 1:
                for yy in (y, ya / na):
                    val = self.certify(K, b, lb, ub, c, yy.double() * dr[:, None]); better = val > best
                    best = th.where(better, val, best); ybest = th.where(better[..., None], yy.double() * dr[:, None], ybest)
                if self.restart_avg:                                     # fixed-frequency restart to the average
                    v, y = va / na, ya / na
                va.zero_(); ya.zero_(); na = 0
        return best, ybest.double()

    @staticmethod
    def certify(K, b, lb, ub, c, y):
        r = c - th.bmm(y, K)
        return (y * b[:, None]).sum(-1) + th.minimum(r * lb[:, None], r * ub[:, None]).sum(-1)

    # ---------------------------------------------------------------- node bounds, prune
    def bounds(self, G, A, k, phase, event_tighten=True):
        N = G.shape[0]; nb = T - k; Z, L = self.zrows(A, k)
        lo = th.full((N, max(nb, 1), H), -math.inf, dtype=th.float64, device=self.dev); hi = -lo.clone()
        for s in range(nb):                                              # forward pass, no event
            K, b, lb, ub = self.build(G, Z, L, lo, hi, phase, s, self.p3)
            c = th.cat([Z[s], -Z[s]], 1)
            val, _ = self.solve(K, b, lb, ub, c)
            lo[:, s] = th.maximum(lo[:, s], val[:, :H]); hi[:, s] = th.minimum(hi[:, s], -val[:, H:])
        if event_tighten and nb:                                         # one Jacobi pass with the event (relax2 stage 2)
            K, b, lb, ub = self.build(G, Z, L, lo, hi, phase, nb, self.p3); lb[:, -1] = -EPS
            c = th.cat([th.cat([Z[s], -Z[s]], 1) for s in range(nb)], 1)
            val, _ = self.solve(K, b, lb, ub, c); val = val.view(N, nb, 2 * H)
            lo2 = th.maximum(lo[:, :nb], val[..., :H]); hi2 = th.minimum(hi[:, :nb], -val[..., H:])
            lo[:, :nb], hi[:, :nb] = lo2, hi2
        return Z, L, lo, hi

    def margin(self, G, Z, L, lo, hi, phase):
        """certified upper bound on the relaxation's max (L_p3 - max_j L_j) (+inf-safe: empty -> very negative)."""
        K, b, lb, ub = self.build(G, Z, L, lo, hi, phase, len(Z), self.p3)
        c = th.zeros(G.shape[0], 1, K.shape[2], dtype=th.float64, device=self.dev); c[..., -1] = -1
        val, y = self.solve(K, b, lb, ub, c)
        return -val[:, 0], y[:, 0]

    def prune(self, G, A, k, tri, phase=None, event_tighten=True):
        """G (N, m, 10), A (N, 16, 10), common k and tri. Returns (margin_ub (N,), pruned (N,), lo, hi)."""
        self.tri, self.p3 = tri, tri[2]; N = G.shape[0]; nb = T - k
        G, A = (th.as_tensor(a, dtype=th.float64, device=self.dev) for a in (G, A))
        phase = th.zeros(N, max(nb, 1), H, dtype=th.int8, device=self.dev) if phase is None else th.as_tensor(phase, device=self.dev)
        Z, L, lo, hi = self.bounds(G, A, k, phase, event_tighten)
        empty = (lo > hi + 1e-9).flatten(1).any(1)
        mg, _ = self.margin(G, Z, L, lo, hi, phase)
        mg = th.where(empty, th.full_like(mg, -math.inf), mg)
        return mg, mg < -EPS, lo, hi
