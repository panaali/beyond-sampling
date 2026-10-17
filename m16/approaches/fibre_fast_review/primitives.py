"""Independent stationary-point ranges and prescribed-gate recurrent maps."""
import math,itertools,json,time
from pathlib import Path
import numpy as np
from loader import load

def direct(model,G):
    wi,wh,wo=model;H=len(wi);T=len(wo);A=np.zeros((H,T));rows=[]
    for t in range(T):
        if t:A=wh@(A*G[(t-1)*H:t*H,None])
        A[:,t]+=wi;rows.append(A.copy())
    return np.vstack(rows),wo@(A*G[(T-1)*H:T*H,None])

def main():
    started=time.process_time();mod,_,_=load();intervals=[(0.,math.pi),(0.,math.pi/2),(-math.pi/2,math.pi/2),(-math.pi,-.1),(.2,1.7),(1.,1.+1e-12),(2.7,5.7),(-4.,-1.)];range_rows=[]
    for al,be in itertools.product((-2.,-1.,0.,1.,2.),repeat=2):
        for t0,t1 in intervals:
            lo,hi=mod._range(np.array([al]),np.array([be]),t0,t1);points=[t0,t1];base=math.atan2(be,al)
            for k in range(-5,6):
                t=base+k*math.pi
                if t0<t<t1:points.append(t)
            vals=[al*math.cos(t)+be*math.sin(t) for t in points];error=max(abs(lo[0]-min(vals)),abs(hi[0]-max(vals)));assert error<=2e-14
            range_rows.append(dict(al=al,be=be,interval=[t0,t1],actual=[lo[0],hi[0]],reference=[min(vals),max(vals)],error=error))
    model=(np.array([.5,-1.]),np.array([[.25,-.5],[.75,.125]]),np.array([[1.,-.5],[.25,.75],[-.125,.5]]));P,U,nn=np.eye(3);seg=(0.,math.pi,2.);x=np.array([1.,-.5,.25]);ops=mod.Ops();base=mod.Piece.at(model,x,P,U,nn,seg,ops);flip_rows=[]
    for bits in itertools.product((False,True),repeat=6):
        group=np.flatnonzero(bits);child=base._flip(model,group,ops);R,L=direct(model,child.G)
        exact={key:np.array_equal(getattr(child,key),value) for key,value in dict(R=R,L=L,RP=R@P,RU=R@U,Rn=R@nn,LP=L@P,LU=L@U,Ln=L@nn).items()};assert all(exact.values()),(bits,exact)
        child.use(seg,ops);freshlo,freshhi=mod._range(child.gal,child.gbe,seg[0],seg[1]);cand=child.ok&(freshhi>0)&(freshlo<seg[2]);assert np.array_equal(child.rlo,freshlo) and np.array_equal(child.rhi,freshhi) and np.array_equal(child.cand,cand)
        flip_rows.append(dict(group=group.tolist(),gate_pattern=child.G.astype(int).tolist(),exact=exact,cached_ranges_equal_fresh=True))
    out=dict(range_rows=range_rows,flip_rows=flip_rows,range_max_error=max(r['error'] for r in range_rows),cpu_seconds=time.process_time()-started,scope='finite range controls and exact dyadic fixed-gate algebra; floating child grouping/topology not certified')
    Path(__file__).with_name('primitive_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('range_rows','flip_rows')}))
if __name__=='__main__':main()
