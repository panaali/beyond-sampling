"""Finite chronology, rank partition, first-index and scale controls."""
from collections import defaultdict
from pathlib import Path
import json,time
import numpy as np
from scipy.integrate import quad
from terminal import MiddleTarget,kernel

HERE=Path(__file__).resolve().parent


def scalar_reference(model,h,u,label,limit=6.):
    """Separate scalar recurrence; determine final y intervals directly."""
    wi,wh,wo=model;stats=defaultdict(float)
    L=kernel.mass(-limit,u,stats);U=kernel.mass(u,limit,stats)
    def inner(x,side):
        state=np.maximum(wh@h+wi*x,0.)
        state=np.maximum(wh@state+wi*u,0.)
        offset=wh@state
        lo,hi=(u,limit) if side else (-limit,u)
        gates=[lo,hi]+[-b/a for a,b in zip(wi,offset) if a and lo<-b/a<hi]
        gates=np.unique(gates);mass=0.
        for a,b in zip(gates[:-1],gates[1:]):
            mask=offset+wi*((a+b)/2)>0
            slope=wo@(wi*mask);intercept=wo@(offset*mask)
            roots=[a,b]
            for j in range(len(wo)):
                d=slope[label]-slope[j]
                if d:
                    r=-(intercept[label]-intercept[j])/d
                    if a<r<b:roots.append(r)
            roots=np.unique(roots)
            for c,d in zip(roots[:-1],roots[1:]):
                if np.argmax(slope*((c+d)/2)+intercept)==label:
                    mass+=kernel.mass(c,d,stats)
        return np.exp(-x*x/2)/np.sqrt(2*np.pi)*mass
    raw=np.zeros((2,2));errors=np.zeros((2,2));warnings=[]
    for a in range(2):
        lo,hi=(u,limit) if a else (-limit,u)
        for b in range(2):
            result=quad(lambda x:inner(x,b),lo,hi,epsabs=2e-10,epsrel=2e-10,limit=250,full_output=1)
            raw[a,b],errors[a,b]=result[:2]
            if len(result)>3:warnings.append(dict(rank=[a,b],message=result[3]))
    return np.array([raw[0,0]/L**2,(raw[0,1]+raw[1,0])/(2*L*U)]),float(errors.max()),warnings


def run():
    start=time.process_time()
    wi=np.array([.8,-.5]);wh=np.array([[.3,-.4],[.5,.2]])
    wo=np.array([[1.,-.2],[-.1,.8],[.2,.1],[-.3,-.1]])
    models={'ordinary':(wi,wh,wo),'constant':(wi,wh,np.ones((4,2))),
            'duplicate':(wi,wh,np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])),
            'zero':(np.zeros(2),np.zeros((2,2)),wo),
            'memoryless':(wi,np.zeros((2,2)),wo)}
    rows=[];partition=axis=scale=direct=analytic=0.;direct_count=0
    for name,model in models.items():
        for h in (np.zeros(2),np.array([.3,.7])):
            for u in (-1.,0.,1.):
                values=[]
                for label in range(4):
                    obj=MiddleTarget(model,h,u,label,diagnostic=True)
                    v,_=obj.probabilities();sw,_=obj.probabilities(swap=True)
                    axis=max(axis,float(np.max(abs(v-sw))));values.append(v)
                    if name in ('constant','zero'):assert np.max(abs(v-int(label==0)))<1e-12
                    if name=='memoryless':
                        stats=defaultdict(float)
                        L=kernel.mass(-6.,u,stats);U=kernel.mass(u,6.,stats)
                        positive_low=kernel.mass(0.,u,stats)/L
                        positive_high=kernel.mass(max(u,0.),6.,stats)/U
                        expected=np.array([positive_low,(positive_low+positive_high)/2])
                        if label==1:expected=1-expected
                        elif label>1:expected=np.zeros(2)
                        analytic=max(analytic,float(np.max(abs(v-expected))))
                    factor=2.**-30
                    sm=(model[0]*factor,model[1],model[2]/factor)
                    sv,_=MiddleTarget(sm,h*factor,u,label,diagnostic=True).probabilities()
                    scale=max(scale,float(np.max(abs(v-sv))))
                    if name=='ordinary':
                        ref,err,warnings=scalar_reference(model,h,u,label)
                        direct=max(direct,float(np.max(abs(v-ref))));direct_count+=1
                        rows.append(dict(hidden=h.tolist(),u=u,label=label,value=v.tolist(),reference=ref.tolist(),scalar_quad_error=err,scalar_warnings=warnings))
                partition=max(partition,float(np.max(abs(np.sum(values,axis=0)-1))))
    obj=MiddleTarget(models['ordinary'],np.zeros(2),0.)
    try:obj.probabilities(.1)
    except ValueError:guard=True
    else:raise AssertionError('construction target guard failed')
    assert partition<1e-10 and axis<1e-10 and scale<1e-10 and analytic<1e-10 and direct<1e-7
    return dict(panels=30,partition_max=partition,axes_max=axis,scale_max=scale,
                direct_max=direct,direct_queries=direct_count,direct_cases=rows,analytic_max=analytic,
                changed_target_rejected=guard,cpu_seconds=time.process_time()-start)


if __name__=='__main__':
    r=run();(HERE/'controls.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
