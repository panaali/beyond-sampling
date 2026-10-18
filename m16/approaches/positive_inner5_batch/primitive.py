"""Branchwise batched binary64 primitive; no whole-mass or cost certificate."""
import numpy as np

def total(a):
    """Preserve scalar left-fold order, without an initial zero addition."""
    s=a[:,0].copy()
    for k in range(1,a.shape[1]):s=s+a[:,k]
    return s

def integrate(P,D):
    if not (isinstance(P,np.ndarray) and isinstance(D,np.ndarray) and P.dtype==np.float64 and D.dtype==np.float64 and P.ndim==D.ndim==2 and P.shape==D.shape and P.shape[1]==5 and len(P)>0):
        raise ValueError('nonempty matching float64 arrays of shape(N,5) required')
    rows=[dict(status='unresolved',value=None,branch=None) for _ in range(len(P))]
    def status(ids,name):
        for i in ids:rows[int(i)]['status']=name
    def branch(ids,name):
        for i in ids:rows[int(i)]['branch']=name
    def divide_failure(ids):
        for i in ids:rows[int(i)].update(status='arithmetic_failure',exception='ZeroDivisionError: float division by zero')
    def finish(ids,j,scale):
        scale2=scale*scale;scale5=(scale2*scale2)*scale
        bad=scale5==0;divide_failure(ids[bad]);keep=~bad
        values=j[keep]/scale5[keep]
        for i,v in zip(ids[keep],values):
            v=float(v);rows[int(i)].update(value=v,status='ok' if np.isfinite(v) and v>0 else 'nonfinite_or_underflow_result')
    with np.errstate(all='ignore'):
        finite=np.all(np.isfinite(P),axis=1)&np.all(np.isfinite(D),axis=1)
        status(np.flatnonzero(~finite),'nonfinite_input');ids=np.flatnonzero(finite)
        if len(ids):
            p0=P[ids];d0=D[ids];constant=np.all(d0==0,axis=1)
            scale=np.maximum.reduce(np.abs(np.concatenate((p0,d0),axis=1)),axis=1)
            origin=scale==0;status(ids[origin],'origin');keep=~origin
            ids=ids[keep];p0=p0[keep];d0=d0[keep];scale=scale[keep];constant=constant[keep]
            p=p0/scale[:,None];d=d0/scale[:,None];a=total(d*d);c=total(p*p)
            ci=np.flatnonzero(constant);branch(ids[ci],'constant')
            bad=c[ci]==0;status(ids[ci[bad]],'underflow_geometry');ci=ci[~bad]
            if len(ci):
                cc=c[ci];j=1.0/((cc*cc)*np.sqrt(cc));finish(ids[ci],j,scale[ci])
            ni=np.flatnonzero(~constant);bad=a[ni]==0;status(ids[ni[bad]],'underflow_geometry');ni=ni[~bad]
            if len(ni):
                ii=ids[ni];pn=p[ni];dn=d[ni];aa=a[ni];sc=scale[ni]
                b=total(pn*dn);squares=[]
                for k in range(5):
                    for l in range(k+1,5):
                        w=pn[:,k]*dn[:,l]-pn[:,l]*dn[:,k];squares.append(w*w)
                delta=total(np.stack(squares,axis=1));bad=~(delta>0);status(ii[bad],'collinear_or_underflow');keep=~bad
                ii=ii[keep];aa=aa[keep];b=b[keep];delta=delta[keep];sc=sc[keep]
                t0=b;t1=aa+b;r0sq=t0*t0+delta;r1sq=t1*t1+delta;r0=np.sqrt(r0sq);r1=np.sqrt(r1sq);ap=aa*np.sqrt(aa)
                same=((t0>0)&(t1>0))|((t0<0)&(t1<0))
                si=np.flatnonzero(same);branch(ii[si],'same_sign')
                if len(si):
                    x0=np.abs(t0[si]);x1=np.abs(t1[si]);den=(r0[si]*r1[si])*(x1*r0[si]+x0*r1[si]);bad=den==0;divide_failure(ii[si[bad]])
                    keep=~bad;si=si[keep];x0=x0[keep];x1=x1[keep];den=den[keep]
                    V=(aa[si]*(x0+x1))/den
                    inner=.5*(1.0/r0sq[si]+1.0/r1sq[si])+(delta[si]*(V*V))/6.0
                    j=(ap[si]*V)*inner;finish(ii[si],j,sc[si])
                ci=np.flatnonzero(~same);branch(ii[ci],'cross_or_endpoint')
                if len(ci):
                    ds=np.abs(t0[ci]/r0[ci])+np.abs(t1[ci]/r1[ci])
                    inner=.5*(delta[ci]/r0sq[ci]+delta[ci]/r1sq[ci])+(ds*ds)/6.0
                    den=delta[ci]*delta[ci];bad=den==0;divide_failure(ii[ci[bad]])
                    keep=~bad;ci=ci[keep];ds=ds[keep];inner=inner[keep];den=den[keep]
                    j=(ap[ci]/den)*(ds*inner);finish(ii[ci],j,sc[ci])
    assert all(r['status']!='unresolved' for r in rows)
    return rows
