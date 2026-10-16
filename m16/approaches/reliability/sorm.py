"""Smooth SORM: curvature of the limit state at the design point coming from the (model-free) Rosenblatt map.

In x-space g is exactly linear within the gate pattern at u*, so its u-space Hessian there is
H = sum_i w_i d^2 x_i/du^2 with w = dg/dx (exact, one gradient pass). Gate kinks are ignored.
Breitung: Pf = Phi(-beta) prod (1 + beta kappa_i)^(-1/2); Hohenbichler variant with phi(b)/Phi(-b) in place of b.
"""
import numpy as np, torch
from scipy.stats import norm
from form import place, logits, C, T
from rosen import sorted_from_u


def x_grad(tm, u, perms, ks, positive):
    """w = dg/dv (sorted-value coordinates) at u, exact local network gradient."""
    v = sorted_from_u(u, positive).detach().requires_grad_(True)
    L = logits(tm, place(v, perms))
    p2 = perms[:, 1]; pk = perms.gather(1, ks[:, None])[:, 0]
    g = L.gather(1, p2[:, None])[:, 0] - L.gather(1, pk[:, None])[:, 0]
    (w,) = torch.autograd.grad(g.sum(), v)
    C.fwd += 3 * u.shape[0]
    return w, g.detach()


def u_hessian(u, w, positive, h=1e-4):
    """Hessian of w . v(u) by central differences of exact gradients (model-free)."""
    B = u.shape[0]
    def grad_at(uu):
        uu = uu.detach().requires_grad_(True)
        f = (w * sorted_from_u(uu, positive)).sum()
        return torch.autograd.grad(f, uu)[0]
    Hs = torch.zeros(B, T, T)
    for j in range(T):
        e = torch.zeros(T); e[j] = h
        Hs[:, :, j] = (grad_at(u + e) - grad_at(u - e)) / (2 * h)
    return 0.5 * (Hs + Hs.transpose(1, 2)), grad_at(u)


def curvatures(H, grad):
    """Principal curvatures kappa (B, T-1) of {g=0} with failure side g<0; positive = failure set shrinks."""
    n = grad / grad.norm(dim=1, keepdim=True)
    B = H.shape[0]; out = np.zeros((B, T - 1))
    for b in range(B):
        nb = n[b].numpy()
        Q, _ = np.linalg.qr(np.c_[nb, np.eye(T)])            # Q[:,0] = +-n
        Pt = Q[:, 1:T]
        Ht = Pt.T @ H[b].numpy() @ Pt
        out[b] = np.linalg.eigvalsh(Ht) / grad[b].norm().item()
    return out


def sorm_pf(beta, kappa, kind='hr'):
    """Failure probability; for beta<0 apply SORM to the safe set (beta -> -beta, kappa -> -kappa)."""
    beta = np.asarray(beta); pf = np.empty(len(beta))
    for i, (b, k) in enumerate(zip(beta, kappa)):
        sgn = 1 if b >= 0 else -1
        bb, kk = abs(b), sgn * k
        if bb >= 7.9:
            pf[i] = 0.0 if sgn > 0 else 1.0; continue
        psi = norm.pdf(bb) / norm.cdf(-bb) if kind == 'hr' else bb
        fac = np.clip(1 + psi * kk, 0.05, None)
        p = norm.cdf(-bb) * np.prod(fac ** -0.5)
        p = min(p, 1.0)
        pf[i] = p if sgn > 0 else 1 - p
    return pf
