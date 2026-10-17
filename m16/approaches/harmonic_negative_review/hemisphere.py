"""Independent S2 projection by direct rotated-hemisphere polynomial cubature.

No imports from harmonic_sphere. Normalized sphere measure has total mass one.
"""
from collections import defaultdict
import math
import time
import numpy as np
from scipy.special import lpmv, roots_legendre


class HemisphereSpace:
    def __init__(self, n, D, deadline=math.inf):
        if n != 3:
            raise ValueError('Independent reference supports S2 only')
        self.n, self.D, self.m = n, D, (D+1)**2
        self.deadline = deadline
        self.counts = defaultdict(int)
        self.flops = 0.0  # only counted linear arithmetic; excludes special functions
        start = time.process_time()
        z, w = roots_legendre(D+1)
        z = (z+1)/2
        phi = 2*np.pi*np.arange(2*D+1)/(2*D+1)
        z, phi = np.broadcast_arrays(z[:,None], phi[None,:])
        radius = np.sqrt(1-z*z)
        self.local = np.stack([radius*np.cos(phi),radius*np.sin(phi),z],axis=-1).reshape(-1,3)
        self.weights = np.repeat(w/(4*(2*D+1)),2*D+1)
        self.N = len(self.local)
        self.terms = [(l,m) for l in range(D+1) for m in range(l+1)]
        self.norms = {(l,m): math.sqrt((2*l+1)*math.factorial(l-m)/math.factorial(l+m))
                      for l,m in self.terms}
        self.setup_cpu = time.process_time()-start

    def check(self):
        if time.process_time() > self.deadline:
            raise TimeoutError('independent review CPU deadline')

    def at(self, X):
        self.check()
        X = np.asarray(X)
        z = np.clip(X[:,2],-1,1)
        phi = np.arctan2(X[:,1],X[:,0])
        cols = []
        for l,m in self.terms:
            base = self.norms[l,m]*lpmv(m,l,z)
            if m:
                base = math.sqrt(2)*base
                cols.extend([base*np.cos(m*phi),base*np.sin(m*phi)])
            else:
                cols.append(base)
        self.counts['basis_scalar_values'] += len(X)*self.m
        self.counts['associated_legendre_scalar_evaluations'] += len(X)*len(self.terms)
        self.counts['atan2_scalar_evaluations'] += len(X)
        self.counts['sin_cos_scalar_evaluations'] += len(X)*self.D*(self.D+1)
        return np.stack(cols,axis=1)

    def hemisphere(self, v):
        self.check()
        norm = np.linalg.norm(v)
        if not norm > 0:
            raise ValueError('zero normal is a tie, not a hemisphere')
        v = v/norm
        axis = np.eye(3)[int(np.argmin(abs(v)))]
        e1 = np.cross(v,axis); e1 /= np.linalg.norm(e1)
        e2 = np.cross(v,e1)
        X = self.local @ np.stack([e1,e2,v])
        self.counts['rotated_nodes'] += self.N
        self.flops += 18*self.N
        return self.at(X)

    def cut(self,C,V):
        out = np.empty_like(C)
        for b,v in enumerate(V):
            P = self.hemisphere(v)
            out[:,b] = P.T @ (self.weights*(P @ C[:,b]))
            self.counts['cuts'] += 1
            self.counts['contraction_multiply_add_upper'] += 4*self.N*self.m+self.N
            self.flops += 4*self.N*self.m+self.N
        return out

    def mass_in(self,C,V):
        out = np.empty(C.shape[1])
        for b,v in enumerate(V):
            P = self.hemisphere(v)
            out[b] = self.weights @ (P @ C[:,b])
            self.counts['mass_queries'] += 1
            self.counts['contraction_multiply_add_upper'] += 2*self.N*self.m+2*self.N
            self.flops += 2*self.N*self.m+2*self.N
        return out

    def one(self):
        c = np.zeros(self.m); c[0] = 1
        return c

    def precompute_flops(self):
        raise NotImplementedError('Special-function setup has no audited FLOP count')
