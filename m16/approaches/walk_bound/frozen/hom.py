"""Radially integrated (homogeneous) tie-face integrand.

The model has no biases, so its error set is a cone. A face point y = r w (w = y/|y|; under the face law r ~ chi_{T-1}
independent of w, the face law being a Gaussian restricted to an order cone). Integrating r out of
    F(y) = int_{|t|<tau(y)} 1[err(y + t d)] phi(t) dt
gives  F_hom(w) = E[F(r w) | w] = int_{|s|<tau(w)} 1[err(w + s d)] k(s) ds,  with k(s) ds the Student-t_{T-1} mass of
s*sqrt(T-1) and tau(w) = tau(y)/|y| (window constraints are order constraints, hence scale-invariant).
The walk is unchanged (exact linear pieces along w + s d); only the 1-D weight changes. E[F_hom] = E[F] exactly.
"""
import numpy as np
from scipy.special import stdtr
import core as C


def tcdf(T):
    nu = T - 1; sq = np.sqrt(nu)
    return lambda z: stdtr(nu, np.where(np.isfinite(z), z, np.sign(z) * 1e300) * sq)


def faces_hom(T, kind, U):
    y, a, b, tau = C.face_points(T, kind, U, cap=np.inf)
    r = np.linalg.norm(y, axis=1)
    return y / r[:, None], a, b, np.minimum(tau / r, 1e8)


def walk_mass_hom(model, w, a, b, tw, kind):
    return C.walk_mass(model, w, a, b, tw, kind, cdf=tcdf(w.shape[1]))
