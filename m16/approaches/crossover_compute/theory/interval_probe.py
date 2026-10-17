"""Scalar-affine integration on DEV tie lines; no estimator-class claim."""
import heapq
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'m16'))
sys.path.insert(0, str(ROOT/'m16/approaches/sci_tao'))
import core as tao
from leaderboard import models


def certificate(model, y, d, lo, hi, target):
    """Return scalar guaranteed and possible correct intervals, and unstable-gate count."""
    wi, wh, wo = model
    n, T = y.shape; H = len(wi)
    mid=(lo+hi)/2; half=(hi-lo)/2
    a=np.zeros((n,H)); b=np.zeros_like(a); r=np.zeros_like(a)
    ambiguous=np.zeros(n, int)
    for t in range(T):
        a=a@wh.T + (half*d[:,t])[:,None]*wi
        b=b@wh.T + (y[:,t]+mid*d[:,t])[:,None]*wi
        r=r@np.abs(wh).T
        L=b-np.abs(a)-r; U=b+np.abs(a)+r
        mixed=(L<0)&(U>0); positive=L>=0
        lam=np.zeros_like(a); mu=np.zeros_like(a)
        lam[positive]=1
        lam[mixed]=U[mixed]/(U[mixed]-L[mixed])
        mu[mixed]=-L[mixed]*lam[mixed]/2
        a*=lam; b=b*lam+mu; r=r*lam+mu
        ambiguous+=mixed.sum(axis=1)
    margins=wo[target,None,:]-wo[None,:,:]
    A=np.einsum('njh,nh->nj',margins,a)
    B=np.einsum('njh,nh->nj',margins,b)
    R=np.einsum('njh,nh->nj',np.abs(margins),r)
    # Numerical safeguard only; does not replace formally rounded interval arithmetic.
    R+=1e-12*(1+np.abs(A)+np.abs(B))
    rows=np.arange(n); A[rows,target]=0; B[rows,target]=1; R[rows,target]=0
    def intersect(const):
        with np.errstate(divide='ignore',invalid='ignore'):
            root=-const/A
        low=np.maximum(-1,np.where(A>0,root,-np.inf).max(axis=1))
        high=np.minimum(1,np.where(A<0,root,np.inf).min(axis=1))
        bad=((A==0)&(const<0)).any(axis=1)
        high=np.where(bad,low-1,high)
        return np.stack([mid+half*low,mid+half*high],axis=1)
    return intersect(B-R),intersect(B+R),ambiguous


def integrate(model,y,a,b,tau,kind,checkpoints=(8,32,128,256),tol=1e-6):
    n,T=y.shape
    d=np.zeros_like(y); d[np.arange(n),a]=1/np.sqrt(2); d[np.arange(n),b]=-1/np.sqrt(2)
    heaps=[[] for _ in range(n)]
    settled_wrong=np.zeros(n); counts=np.zeros(n,int); ambiguous=[]
    def push(i,l,h,k):
        if h>l:
            mass=float(ndtr(h)-ndtr(l))
            if mass>0:
                heapq.heappush(heaps[i],(-mass,float(l),float(h),int(k)))
    for i in range(n):
        push(i,-tau[i],0,b[i] if kind=='23' else a[i])
        push(i,0,tau[i],a[i] if kind=='23' else b[i])
    snapshots=[]
    for iteration in range(1,max(checkpoints)+1):
        ids=[i for i in range(n) if heaps[i] and -sum(q[0] for q in heaps[i])>tol]
        if ids:
            nodes=[heapq.heappop(heaps[i]) for i in ids]
            lo=np.array([q[1] for q in nodes]); hi=np.array([q[2] for q in nodes])
            target=np.array([q[3] for q in nodes])
            C,P,amb=certificate(model,y[ids],d[ids],lo,hi,target)
            ambiguous.extend(amb.tolist())
            for j,i in enumerate(ids):
                counts[i]+=1
                pl,ph=P[j]; cl,ch=C[j]
                pl=max(pl,lo[j]); ph=min(ph,hi[j]); cl=max(cl,pl); ch=min(ch,ph)
                pmass=max(0,float(ndtr(ph)-ndtr(pl))) if ph>pl else 0
                settled_wrong[i]+=-nodes[j][0]-pmass
                if pmass<=0:
                    continue
                intervals=[(pl,ph)] if ch<=cl else [(pl,cl),(ch,ph)]
                for l,h in intervals:
                    if h<=l:
                        continue
                    midpoint=float(ndtri((ndtr(l)+ndtr(h))/2))
                    push(i,l,midpoint,target[j]); push(i,midpoint,h,target[j])
        if iteration in checkpoints:
            unresolved=np.array([-sum(q[0] for q in h) for h in heaps])
            snapshots.append(dict(max_nodes=iteration,counts=counts.tolist(),
                error_lower=settled_wrong.tolist(),error_upper=(settled_wrong+unresolved).tolist(),
                unresolved=unresolved.tolist(),mean_ambiguous_gates=float(np.mean(ambiguous))))
    return snapshots


def main():
    start=time.process_time(); wall=time.monotonic(); out={'models':[]}
    dev=models(dev=True)
    for name in ['dev_s0.0025_0','dev_s0.02_0']:
        for kind in ['23','12']:
            M=dev[name]
            # Fixed Sobol outer directions serve only as a diagnostic set, not a scored estimator.
            y,a,b,tau=tao.face_points(10,kind,tao.sobol(10,64,344))
            truth,legacy_cost,cells=tao.walk_mass(M,y,a,b,tau,kind)
            t0=time.process_time(); w0=time.monotonic()
            snap=integrate(M,y,a,b,tau,kind)
            result={'name':name,'kind':kind,'n_lines':len(y),'reference_error_mass':truth.tolist(),
                'reference_cells':cells.tolist(),'reference_actual_fpe':int(2*cells.sum()),
                'bound_cpu_seconds':time.process_time()-t0,'bound_wall_seconds':time.monotonic()-w0,
                'checkpoints':[]}
            for q in snap:
                low=np.array(q['error_lower']); high=np.array(q['error_upper']); count=np.array(q['counts'])
                violation=float(max(np.max(low-truth),np.max(truth-high),0))
                if violation>1e-8:
                    raise AssertionError((name,kind,violation))
                q.update(max_reference_violation=violation,
                    mean_unresolved=float(np.mean(q['unresolved'])),
                    max_unresolved=float(np.max(q['unresolved'])),
                    frac_tol=float(np.mean(np.array(q['unresolved'])<=1e-6)),
                    mean_nodes=float(count.mean()),bound_recurrence_fpe=int(3*count.sum()),
                    recurrence_cost_ratio_to_reference=float(3*count.sum()/(2*cells.sum())))
                result['checkpoints'].append(q)
                print(name,kind,{k:q[k] for k in ['max_nodes','mean_nodes','mean_unresolved','frac_tol','recurrence_cost_ratio_to_reference','max_reference_violation']},flush=True)
            out['models'].append(result)
    out.update(cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall,
               classification='DEV diagnostic; floating-point enclosures, not formally rounded certificates',
               network_source='Public AlgZoo checkpoint only; no external solution prose')
    Path(__file__).with_name('interval_results.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':
    main()
