"""T399: batched GPU bound propagation for the certified rare-event branch and bound (T390 / T398 node convention).

Node (T390): a cell {G x >= 0} of R^10 on which the first k steps have fixed gates, so h_k = A x exactly.
Domain (T390 relax.py): x in [-1, 1]^10, G x >= 0, x_p2 - x_p3 >= DELTA (homogeneous normalisation).
Event (far-gap pick of p3): L_p3 >= L_j - EPS for all j != p3, L = Wo h_10.
Prune a node iff  min over the domain of  max_j (L_j - L_p3)  > EPS  is certified. For any lambda in the simplex,
max_j (L_j - L_p3) >= sum_j lambda_j (L_j - L_p3); its minimum is lower-bounded by CROWN: backward linear relaxation
of the remaining ReLUs (h >= alpha z, h <= u (z - l) / (u - l) for unstable neurons; exact for stable ones) down to x,
then concretisation over the domain by Lagrangian duality:
  min_{x in D} c.x + c0 >= max_{mu, nu >= 0} [ -|| c - G^T mu - nu e ||_1 + nu DELTA + c0 ]   (box [-1, 1]^10),
valid for ANY mu, nu >= 0 (weak duality). Intermediate pre-activation bounds [l, u] by the same backward pass + dual.
lambda, alpha, mu, nu are optimised by projected gradient; every value of them gives a valid bound, so soundness
depends only on floating-point rounding (handled by a safety margin here; exact re-verification is T400's job).
All tensors float64, batched over nodes (and over neurons for intermediate bounds)."""
import math
import torch as th

D, H, T = 10, 16, 10
DELTA, EPS = 0.1, 1e-7


class Engine:
    def __init__(self, Wi, Wh, Wo, device="cuda", dual_iters=60, spec_iters=60, lr=0.05):
        f = lambda a: th.as_tensor(a, dtype=th.float64, device=device)
        self.Wi, self.Wh, self.Wo = f(Wi), f(Wh), f(Wo); self.dev = device
        self.dual_iters, self.spec_iters, self.lr = dual_iters, spec_iters, lr

    # ---- concretisation: lower bound of c.x + c0 over the node domain, batched (N, Q, D) -------------------------
    def concretise(self, c, c0, G, e, iters=None, mu=None):
        """c (N, Q, D), c0 (N, Q), G (N, R, D), e (N, D) gap row. Returns (N, Q) lower bounds (max over duals)."""
        iters = self.dual_iters if iters is None else iters
        N, Q, _ = c.shape; R = G.shape[1]
        mu = th.zeros(N, Q, R, dtype=c.dtype, device=c.device) if mu is None else mu
        nu = th.zeros(N, Q, dtype=c.dtype, device=c.device)
        best = th.full((N, Q), -math.inf, dtype=c.dtype, device=c.device)
        for it in range(iters + 1):
            r = c - th.einsum("nqr,nrd->nqd", mu, G) - nu[..., None] * e[:, None, :]
            val = -r.abs().sum(-1) + nu * DELTA + c0
            best = th.maximum(best, val)
            if it == iters:
                break
            s = th.sign(r)                                             # supergradient of -|r|_1 wrt r is -sign(r)
            gmu = th.einsum("nqd,nrd->nqr", s, G); gnu = (s * e[:, None, :]).sum(-1) + DELTA
            step = self.lr / math.sqrt(it + 1)
            mu = th.clamp(mu + step * gmu, min=0); nu = th.clamp(nu + step * gnu, min=0)
        return best

    # ---- backward pass: coefficients of a linear function of z_s (or h_10) in terms of x -----------------------------
    def backward(self, cz, s_top, k, A, lo, up, alpha):
        """cz (N, Q, H): coefficients on z_{s_top} (pre-activation at step s_top). Relaxation of h_s for s < s_top
        uses bounds lo/up[s] and lower slopes alpha[s]. Returns (cx (N, Q, D), c0 (N, Q)) with
        cz.z_{s_top} >= cx.x + c0 for all x in the node domain."""
        N, Q, _ = cz.shape
        cx = cz * 0 @ th.zeros(H, D, dtype=cz.dtype, device=cz.device)  # (N, Q, D) zeros
        cx = th.zeros(N, Q, D, dtype=cz.dtype, device=cz.device); c0 = th.zeros(N, Q, dtype=cz.dtype, device=cz.device)
        lam = cz                                                          # coefficients on z_s
        for s in range(s_top, k - 1, -1):
            cx[:, :, s] += (lam * self.Wi[None, None, :]).sum(-1)       # z_s = Wh h_s + Wi x_s
            ch = lam @ self.Wh                                            # coefficients on h_s (state before step s)
            if s == k:                                                    # h_k = A x exactly
                cx = cx + th.einsum("nqh,nhd->nqd", ch, A)
                break
            l, u, a = lo[s - 1], up[s - 1], alpha[s - 1]                   # bounds of z_{s-1}; h_s = relu(z_{s-1})
            act = (l >= 0); inact = (u <= 0); unst = ~(act | inact)
            slope_up = th.where(unst, u / (u - l).clamp_min(1e-300), act.to(u.dtype))
            icpt_up = th.where(unst, -u * l / (u - l).clamp_min(1e-300), th.zeros_like(u))
            slope_lo = th.where(unst, a, act.to(u.dtype))
            pos = ch >= 0
            slope = th.where(pos, slope_lo[:, None, :], slope_up[:, None, :])
            c0 = c0 + th.where(pos, th.zeros_like(ch), ch * icpt_up[:, None, :]).sum(-1)
            lam = ch * slope
        return cx, c0

    def bounds(self, G, A, k, tri):
        """intermediate bounds lo/up[s] for z_s, s = k..T-1 (CROWN-Ada slopes). G (N, R, D), A (N, H, D)."""
        N = G.shape[0]; e = th.zeros(N, D, dtype=G.dtype, device=G.device); e[:, tri[1]] = 1; e[:, tri[2]] = -1
        lo, up, alpha = {}, {}, {}
        eye = th.eye(H, dtype=G.dtype, device=G.device)
        for s in range(k, T):
            cz = th.cat([eye, -eye])[None].expand(N, 2 * H, H).contiguous()
            cx, c0 = self.backward(cz, s, k, A, lo, up, alpha)
            b = self.concretise(cx, c0, G, e)
            lo[s], up[s] = b[:, :H], -b[:, H:]
            alpha[s] = (up[s] >= -lo[s]).to(G.dtype)                    # CROWN-Ada lower slope
        return lo, up, alpha, e

    def empty(self, G, tri, iters=200):
        """certificate that the domain is empty: max over mu, nu >= 0 of -||G^T mu + nu e||_1 + nu DELTA > 0
        (weak duality with the zero objective); returns (N,) bool"""
        N = G.shape[0]; e = th.zeros(N, D, dtype=G.dtype, device=G.device); e[:, tri[1]] = 1; e[:, tri[2]] = -1
        z = th.zeros(N, 1, D, dtype=G.dtype, device=G.device)
        return self.concretise(z, th.zeros(N, 1, dtype=G.dtype, device=G.device), G, e, iters=iters)[:, 0] > 1e-12

    def prune(self, G, A, k, tri, margin=1e-9):
        """returns (lower bound on min max_j (L_j - L_p3), pruned bool) per node"""
        emp = self.empty(G, tri)
        lo, up, alpha, e = self.bounds(G, A, k, tri); N = G.shape[0]; p3 = tri[2]
        others = [j for j in range(D) if j != p3]
        Dm = self.Wo[others] - self.Wo[p3][None]                        # (9, H) on h_10 = relu(z_9)
        theta = th.zeros(N, len(others), dtype=G.dtype, device=G.device, requires_grad=True)
        best = th.full((N,), -math.inf, dtype=G.dtype, device=G.device)
        opt = th.optim.Adam([theta], lr=0.1)
        for it in range(self.spec_iters):
            lam = th.softmax(theta, -1)                                  # simplex
            ch = (lam @ Dm)[:, None, :]                                   # (N, 1, H) coefficients on h_10
            # h_10 = relu(z_9): relax with bounds of z_9
            l, u, a = lo[T - 1], up[T - 1], alpha[T - 1]
            act = (l >= 0); inact = (u <= 0); unst = ~(act | inact)
            su = th.where(unst, u / (u - l).clamp_min(1e-300), act.to(u.dtype)); iu = th.where(unst, -u * l / (u - l).clamp_min(1e-300), th.zeros_like(u))
            sl = th.where(unst, a, act.to(u.dtype)); pos = ch >= 0
            c0h = th.where(pos, th.zeros_like(ch), ch * iu[:, None, :]).sum(-1)
            cz = ch * th.where(pos, sl[:, None, :], su[:, None, :])
            cx, c0 = self.backward(cz, T - 1, k, A, lo, up, alpha)
            with th.no_grad():
                b = self.concretise(cx.detach(), (c0 + c0h).detach(), G, e)[:, 0]
                best = th.maximum(best, b)
            # differentiable surrogate (dual fixed at zero) to steer lambda
            sur = -(cx.abs().sum(-1))[:, 0] + (c0 + c0h)[:, 0]
            opt.zero_grad(); (-sur.sum()).backward(); opt.step()
        best = th.where(emp, th.full_like(best, math.inf), best)
        return best, (best > EPS + margin) | emp
