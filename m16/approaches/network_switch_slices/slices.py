"""Bounded two-dimensional affine network cells on a fixed tie-face slice."""
from collections import Counter
import importlib.util
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri

ROOT = Path('/workspace/clean-neurons')
WALKER = ROOT/'m16/approaches/sci_tao_v2/core.py'
spec = importlib.util.spec_from_file_location('_slice_reference_walker', WALKER)
walker = importlib.util.module_from_spec(spec); spec.loader.exec_module(walker)


def area(p):
    if len(p) < 3: return 0.
    q = np.asarray(p)-p[0]
    return float(abs(np.sum(q[:-1,0]*q[1:,1]-q[:-1,1]*q[1:,0]))/2)


def clip(p, row, work):
    """Keep row[0]*z+row[1]*t+row[2]>=0. No epsilon deletion."""
    work['polygon_clips'] += 1; work['tested_vertices'] += len(p)
    if len(p) < 3: return np.empty((0,2))
    v = p@row[:2]+row[2]; out = []
    for i, a in enumerate(p):
        j = (i+1) % len(p); b = p[j]
        if v[i] >= 0: out.append(a)
        if (v[i] > 0 and v[j] < 0) or (v[i] < 0 and v[j] > 0):
            out.append(a+(v[i]/(v[i]-v[j]))*(b-a)); work['intersections'] += 1
    if not out: return np.empty((0,2))
    clean = [out[0]]
    for point in out[1:]:
        if not np.array_equal(point, clean[-1]): clean.append(point)
    if len(clean)>1 and np.array_equal(clean[0],clean[-1]): clean.pop()
    out = np.asarray(clean)
    return out if area(out)>0 else np.empty((0,2))


def vertical_mass(polygons, z, work):
    """Polygon interiors; half-open projected intervals avoid shared-edge doubles."""
    out = np.zeros(len(z))
    for p in polygons:
        loz, hiz = p[:,0].min(), p[:,0].max()
        ids = np.flatnonzero((z >= loz)&(z < hiz))
        if not len(ids): continue
        low = np.full(len(ids), -np.inf); high = np.full(len(ids), np.inf)
        for a, b in zip(p, np.roll(p,-1,axis=0)):
            # CCW polygon interior: cross(b-a, point-a)>=0.
            dz, dt = b-a
            if dz > 0: low = np.maximum(low, a[1]+dt/dz*(z[ids]-a[0]))
            elif dz < 0: high = np.minimum(high, a[1]+dt/dz*(z[ids]-a[0]))
            work['vertical_edge_queries'] += len(ids)
        if np.any(high < low-1e-11): raise ArithmeticError('Inverted vertical section')
        out[ids] += ndtr(high)-ndtr(low); work['cdf_calls'] += 2*len(ids)
    return out


class Slice:
    def __init__(self, model, kind, configuration, qlo=.05, qhi=.95, cap=5000, fixed_quantiles=None):
        self.model, self.kind = tuple(np.asarray(a,float) for a in model), kind
        wi, wh, wo = self.model; H, T = len(wi), len(wo)
        self.work = Counter(); self.qlo, self.qhi = qlo, qhi
        cfg = walker.geometry.configs(T,kind)
        U = np.full((3,T),.5)
        if fixed_quantiles is not None:
            fixed_quantiles=np.asarray(fixed_quantiles,float)
            if fixed_quantiles.shape!=(T,) or np.any((fixed_quantiles<=0)|(fixed_quantiles>=1)):
                raise ValueError('Interior fixed face quantiles required')
            U[:]=fixed_quantiles
        U[:,0]=(configuration+.5)/len(cfg); U[:,2]=[qlo,.5,qhi]
        y,a,b,_ = walker.face_points(T,kind,U)
        self.a, self.b = int(a[0]),int(b[0])
        changed = np.flatnonzero(y[0]!=y[2])
        if len(changed)!=1: raise ValueError('Expected one varying nontied input')
        self.input_index = int(changed[0]); k=self.input_index
        self.tie = float(y[1,self.a]); self.P = float(ndtr(self.tie))
        self.upper = kind=='23'  # coordinate2 is upper in23, first lower in12
        self.zlo, self.zhi = float(y[0,k]),float(y[2,k])
        X = np.zeros((T,3)); X[:,2]=y[1]; X[k]=[1,0,0]
        X[self.a,1]=1/np.sqrt(2); X[self.b,1]=-1/np.sqrt(2)
        self.X = X
        p = np.array([[self.zlo,-8.5],[self.zhi,-8.5],[self.zhi,8.5],[self.zlo,8.5]])
        limits=[]
        if kind=='23':
            top=int(cfg[configuration,2]); lower=[j for j in range(T) if j not in (self.a,self.b,top)]
            limits.append(np.sqrt(2)/3*(X[top]-np.array([0,0,self.tie])))
            limits += [np.sqrt(2)*(np.array([0,0,self.tie])-X[j]) for j in lower]
        else:
            limits += [np.sqrt(2)/3*(np.array([0,0,self.tie])-X[j]) for j in range(T) if j not in (self.a,self.b)]
        for row in limits:
            for sign in (-1,1):
                r=row.copy(); r[1]=sign; p=clip(p,r,self.work)
        self.domain=p; self.geometry_knots=self.toq(np.unique(p[:,0]))
        cells=[(p,np.zeros((H,3)))]; self.area_defects=[]
        for s in range(T):
            nextcells=[]
            for poly,B in cells:
                Z=wh@B+wi[:,None]*X[s]
                self.work['affine_coefficient_proxy'] += 3*H*(2*H+1)
                pieces=[(poly,Z.copy())]
                for neuron in range(H):
                    split=[]; row=Z[neuron]
                    for polygon,C in pieces:
                        if not np.any(row):
                            C=C.copy(); C[neuron]=0; split.append((polygon,C)); continue
                        before=area(polygon); after=0.
                        for positive in (False,True):
                            q=clip(polygon,row if positive else -row,self.work)
                            if len(q):
                                D=C.copy()
                                if not positive: D[neuron]=0
                                split.append((q,D)); after+=area(q)
                        self.area_defects.append(abs(after-before))
                    pieces=split
                    if len(pieces)+len(nextcells)>cap: raise RuntimeError('Explicit polygon cap exceeded')
                nextcells.extend(pieces)
            cells=nextcells
            if len(cells)>cap: raise RuntimeError('Explicit polygon cap exceeded')
            self.work['stage_cells'] += len(cells)
        self.cells=cells; self.correct=[]; self.state_defect=0.
        for polygon,B in cells:
            center=polygon.mean(0); v=np.r_[center,1.]; h=np.zeros(H)
            for row in X: h=np.maximum(wh@h+wi*np.dot(row,v),0)
            self.state_defect=max(self.state_defect,float(np.max(abs(h-B@v))))
            self.work['centroid_validation_forward_steps'] += T
            logits=wo@B; self.work['readout_coefficient_proxy'] += 3*T*(2*H-1)
            for side in (-1,1):
                target=(self.b if side<0 else self.a) if kind=='23' else (self.a if side<0 else self.b)
                q=clip(polygon,np.array([0,side,0]),self.work)
                for j in range(T):
                    if j==target or not len(q): continue
                    margin=logits[target]-logits[j]; self.work['readout_inequalities'] += 1
                    if not np.any(margin):
                        if j<target: q=np.empty((0,2))
                    else: q=clip(q,margin,self.work)
                if len(q): self.correct.append(q)
        vertices=[self.domain[:,0]]+[p[:,0] for p in self.correct]
        self.knots=np.unique(np.r_[qlo,self.toq(np.concatenate(vertices)),qhi])
        self.knots=self.knots[(self.knots>=qlo)&(self.knots<=qhi)]
        self.work['retained_gate_cells']=len(cells); self.work['correct_polygons']=len(self.correct)

    def toq(self,z):
        self.work['cdf_calls'] += np.size(z)
        return (ndtr(z)-self.P)/(1-self.P) if self.upper else ndtr(z)/self.P

    def toz(self,q):
        self.work['quantile_calls'] += np.size(q)
        return ndtri(self.P+(1-self.P)*np.asarray(q)) if self.upper else ndtri(self.P*np.asarray(q))

    def values(self,q):
        z=self.toz(q)
        return vertical_mass([self.domain],z,self.work)-vertical_mass(self.correct,z,self.work)

    def integrate(self,knots,order):
        x,w=np.polynomial.legendre.leggauss(order)
        lo,hi=np.asarray(knots[:-1]),np.asarray(knots[1:])
        q=(lo[:,None]+hi[:,None])/2+(hi-lo)[:,None]*x/2
        values=self.values(q.ravel()).reshape(q.shape)
        self.work['quadrature_nodes'] += q.size; self.work['quadrature_panels'] += len(lo)
        return float(np.sum(values*w*(hi-lo)[:,None]/2))

    def integrate_gaussian(self,knots,order):
        x,w=np.polynomial.legendre.leggauss(order)
        zknots=self.toz(knots); lo,hi=zknots[:-1],zknots[1:]
        z=(lo[:,None]+hi[:,None])/2+(hi-lo)[:,None]*x/2
        density=np.exp(-z*z/2)/np.sqrt(2*np.pi)/(1-self.P if self.upper else self.P)
        value=(vertical_mass([self.domain],z.ravel(),self.work)-
               vertical_mass(self.correct,z.ravel(),self.work)).reshape(z.shape)
        self.work['quadrature_nodes'] += z.size;self.work['quadrature_panels'] += len(lo)
        self.work['gaussian_exp_calls'] += z.size
        weights=density*w*(hi-lo)[:,None]/2
        return float(np.sum(value*weights)),float(weights.sum())
