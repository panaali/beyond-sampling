"""T270 diagnosis: is the missing mass in a leaf cell's RAYS or its TRIANGULATION? For the worst regions of subtree POS
(sx6/mc6), re-walk the subtree keeping the leaf cones, then:
  rays_q: vertex enumeration of {N x >= 0} independently (scipy qhull HalfspaceIntersection on the slice x . c = 1,
          c = normalized sum of inward normals, interior point by Chebyshev LP);
  mass_tri_stored: mass of the stored rays c.R re-triangulated by qhull (Delaunay of the slice points);
  mass_tri_q: mass of rays_q triangulated the same way; vs estT's pulling triangulation mass and the region MC.
usage: python cx6.py POS [nreg]"""
import json, os, sys
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import Delaunay, HalfspaceIntersection
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import prefix6 as Q, prefix as P, gmass, estG  # noqa: E402
import estT_base as B  # noqa: E402
from m43 import Ops  # noqa: E402
pos = int(sys.argv[1]); nreg = int(sys.argv[2]) if len(sys.argv) > 2 else 3
w = json.load(open(os.path.join(Q.OUT, f"mc6_{pos}.json")))["worst"][:nreg]; want = {r["i"]: r for r in w}
model, front = Q.front_list(11); P._init(model, Q.K, True)
c0, g0, A0, t0, h0 = front[pos]; stack = [(c0, g0, A0, t0, h0)]; leaf = 0; cones = {}
while stack:                                            # same DFS order as prefix._dfs
    c, g, A, t, hist = stack.pop()
    if t == Q.K:
        if leaf in want:
            cones[leaf] = c
        leaf += 1; continue
    for c2, g2, A2 in P._children(c, g, A, t, Ops()):
        stack.append((c2, g2, A2, t + 1, hist + np.packbits(g2 > 0).tobytes()))


def slice_basis(N):
    cdir = N.sum(0); cdir /= np.linalg.norm(cdir); U = np.linalg.svd(np.eye(6) - np.outer(cdir, cdir))[0][:, :5]
    return cdir, U


def mass_of(Rays, cdir, U):
    Z = (Rays / (Rays @ cdir)[:, None] - cdir) @ U                       # slice coordinates
    tri = Delaunay(Z); S = Rays[tri.simplices]                          # (s, 6, 6) unit rays
    S = S / np.linalg.norm(S, axis=2, keepdims=True)
    return float(gmass.grouped(S, np.zeros(len(S), np.int64), 1, "cuda", chunk=1024, tb=512)[0]), len(S)


out = []
for i, c in cones.items():
    N = c.N / np.linalg.norm(c.N, axis=1, keepdims=True); cdir, U = slice_basis(N)
    A_ = -(N @ U); b_ = -(N @ cdir)                                      # A z + b <= 0
    nrm = np.linalg.norm(A_, axis=1)
    lp = linprog(np.r_[np.zeros(5), -1], A_ub=np.c_[A_, nrm], b_ub=-b_, bounds=[(None, None)] * 5 + [(0, None)])
    hs = HalfspaceIntersection(np.c_[A_, b_], lp.x[:5])
    Rq = cdir + hs.intersections @ U.T; Rq /= np.linalg.norm(Rq, axis=1, keepdims=True)
    Rq = np.unique(np.round(Rq, 12), axis=0)
    col = P._Col(); estG.BUF[0] = col; B.mass(c, Ops())
    m_est = float(gmass.grouped(np.array(col.R), np.zeros(len(col.R), np.int64), 1, "cuda", chunk=1024, tb=512)[0])
    ms, ns = mass_of(c.R, cdir, U); mq, nq = mass_of(Rq, cdir, U)
    # stored rays missing from qhull's list and vice versa (angle tolerance 1e-9)
    D = np.abs(c.R @ Rq.T); miss_q = int((D.max(0) < 1 - 1e-9).sum()); extra = int((D.max(1) < 1 - 1e-9).sum())
    out.append(dict(region=i, mc=want[i]["mc"], mass_estT=m_est, simplices_estT=len(col.R), mass_tri_stored=ms,
                    simplices_tri_stored=ns, mass_tri_q=mq, rays_stored=len(c.R), rays_q=len(Rq), normals=len(N),
                    q_rays_missing_from_stored=miss_q, stored_rays_not_vertices=extra, lp_radius=float(lp.x[5])))
    print(json.dumps(out[-1]), flush=True)
json.dump(out, open(os.path.join(Q.OUT, f"cx6_{pos}.json"), "w"), indent=1)
