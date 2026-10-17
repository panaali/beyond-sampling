"""Exact polyhedral cones {x in R^n : a_i . x >= 0} by the double-description method (Motzkin).

A cone is stored by generators: lineality basis L (k,n) and extreme rays R (r,n) with, for each ray, the set of
constraints tight at it (Python int bitmask). This is used only to decide, exactly, whether a new halfspace
contains, misses or splits the current cone, so that the harmonic density is only cut where geometry requires.
Every operation adds to Cone.flops (counted as model-dependent work).
"""
import numpy as np

EPS = 1e-11


class Cone:
    __slots__ = ("n", "L", "R", "Z", "nc", "flops")

    def __init__(self, n):
        self.n = n; self.L = np.eye(n); self.R = np.zeros((0, n)); self.Z = []; self.nc = 0; self.flops = 0.0

    def copy(self):
        c = Cone.__new__(Cone)
        c.n, c.L, c.R, c.Z, c.nc, c.flops = self.n, self.L, self.R, list(self.Z), self.nc, 0.0
        return c

    def classify(self, a):
        """+1: cone inside {a.x >= 0}; -1: inside {a.x <= 0} (no interior on the + side); 0: splits."""
        self.flops += 2 * self.n * (len(self.L) + len(self.R))
        if len(self.L) and np.abs(self.L @ a).max() > EPS:
            return 0
        s = self.R @ a
        pos, neg = (s > EPS).any(), (s < -EPS).any()
        return 0 if (pos and neg) else (1 if pos or not neg else -1)

    def add(self, a):
        """Intersect with {a.x >= 0} in place (caller has checked that it splits or is relevant)."""
        a = a / np.linalg.norm(a)
        bit = 1 << self.nc; self.nc += 1
        n = self.n
        if len(self.L):
            la = self.L @ a
            i0 = int(np.argmax(np.abs(la)))
            if abs(la[i0]) > EPS:
                l0 = self.L[i0] * np.sign(la[i0]); a0 = abs(la[i0])
                others = np.delete(self.L, i0, 0)
                others = others - np.outer(np.delete(la, i0) / a0, l0) if len(others) else others
                # (others . a now 0)
                ra = self.R @ a
                R = self.R - np.outer(ra / a0, l0) if len(self.R) else self.R
                Z = [z | bit for z in self.Z]
                full = (1 << (self.nc - 1)) - 1          # l0 was tight on every old constraint
                self.L = others
                self.R = np.vstack([R, l0[None]]) if len(R) else l0[None].copy()
                self.Z = Z + [full]
                self.R = self.R / np.linalg.norm(self.R, axis=1, keepdims=True)
                self.flops += 4 * n * (len(self.L) + len(self.R) + 1)
                return
        s = self.R @ a
        P = np.where(s > EPS)[0]; Nn = np.where(s < -EPS)[0]; Zr = np.where(np.abs(s) <= EPS)[0]
        self.flops += 2 * n * len(self.R)
        newR = [self.R[i] for i in P] + [self.R[i] for i in Zr]
        newZ = [self.Z[i] for i in P] + [self.Z[i] | bit for i in Zr]
        need = n - len(self.L) - 2                         # adjacency: common tight set of rank >= need
        allZ = self.Z
        for p in P:
            for q in Nn:
                z = allZ[p] & allZ[q]
                self.flops += len(allZ)
                if bin(z).count("1") < need:
                    continue
                ok = True
                for r, zr in enumerate(allZ):
                    if r != p and r != q and (zr & z) == z:
                        ok = False; break
                if ok:
                    v = s[p] * self.R[q] - s[q] * self.R[p]
                    newR.append(v / np.linalg.norm(v)); newZ.append(z | bit)
                    self.flops += 3 * n
        self.R = np.array(newR) if newR else np.zeros((0, n)); self.Z = newZ

    def interior_point(self):
        return self.R.sum(0) if len(self.R) else np.zeros(self.n)
