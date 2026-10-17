from collections import defaultdict
import json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from terminal import PairTerminal,mass,SQRT2PI
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'rank_conditioned_transport'))
from engine import decision_intervals
start=time.process_time();out={'rows':[],'direct_rows':[],'scale_rows':[]};ledger=defaultdict(float)
wi=np.array([1.,-.7]);wh=np.array([[.4,-.8],[.3,.6]]);wo=np.array([[1.,-.2],[-.4,.7],[.3,.1],[-1.,-.6]])
models={'ordinary':(wi,wh,wo),'constant':(wi,wh,np.zeros((4,2))),'duplicate':(wi,wh,np.tile(wo[0],(4,1))),'zero_gates':(np.zeros(2),np.zeros((2,2)),wo),'wh0':(wi,np.zeros((2,2)),wo),'narrow':(np.array([1.,-1.]),np.eye(2),np.array([[1.,0.],[0.,1.],[1.-1e-9,1.],[-1.,-1.]]))}
for name,model in models.items():
    for h in (np.zeros(2),np.array([.3,.7])):
        objects=[PairTerminal(model,h,j,diagnostic=True) for j in range(4)]
        for u in (-1.,0.,1.):
            vals=[];errors=[];swaps=[]
            for obj in objects:
                value,stats=obj.probabilities(u);other,_=obj.probabilities(u,swap=True)
                vals.append(value);errors.append(stats['quadrature_error_estimates']);swaps.append(float(np.max(abs(value-other))))
            defect=float(np.max(abs(np.sum(vals,axis=0)-1.)))
            row=dict(model=name,hidden=h.tolist(),u=u,values=np.array(vals).tolist(),partition_defect=defect,max_axis_disagreement=max(swaps),quadrature_error_estimates=errors)
            out['rows'].append(row)
            (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
            assert defect<1e-10 and max(swaps)<1e-10
            if name in ('constant','duplicate','zero_gates'):assert np.max(abs(np.array(vals)-np.array([[1,1],[0,0],[0,0],[0,0]])))<1e-10
        for obj in objects:
            for k,v in obj.stats.items():ledger[k]+=v
# Different scalar-terminal recursion; same complete polygon endpoints only delimit quadrature.
model=models['ordinary'];h=np.array([.3,.7])
for label in (0,1):
    obj=PairTerminal(model,h,label)
    for u in (-1.,0.,1.):
        value,_=obj.probabilities(u);points=np.unique([-6.,6.,u,*[v for p in obj.pieces[False] for v in p[:2]]]);raw=np.zeros((2,2));stats=defaultdict(float)
        for lo,hi in zip(points[:-1],points[1:]):
            first=int((lo+hi)/2>u)
            for second in (0,1):
                def integrand(x):
                    hx=np.maximum(model[1]@h+model[0]*x,0.)
                    intervals=decision_intervals(model,hx,u,label,6.,stats,target_last=False)
                    inner=sum(mass(max(a,u) if second else a,b if second else min(b,u),stats) for a,b in intervals)
                    return math.exp(-x*x/2)/SQRT2PI*inner
                raw[first,second]+=quad(integrand,lo,hi,epsabs=1e-13,epsrel=1e-11)[0]
        low,high=mass(-6.,u,stats),mass(u,6.,stats)
        direct=np.array([raw[0,0]/low**2,(raw[0,1]+raw[1,0])/(2*low*high)])
        discrepancy=float(np.max(abs(value-direct)));out['direct_rows'].append(dict(label=label,u=u,value=value.tolist(),direct=direct.tolist(),discrepancy=discrepancy));assert discrepancy<1e-10
for label in (0,1):
    ref=PairTerminal(model,h,label);scaled=PairTerminal((model[0]*1e-16,model[1],model[2]),h*1e-16,label)
    for u in (-1.,0.,1.):
        a,_=ref.probabilities(u);b,_=scaled.probabilities(u);error=float(np.max(abs(a-b)));out['scale_rows'].append(dict(label=label,u=u,error=error));assert error<1e-10
out['ledger']=dict(ledger);out['total_cpu_seconds']=time.process_time()-start
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cpu=out['total_cpu_seconds'],rows=len(out['rows']),partition=max(r['partition_defect'] for r in out['rows']),axis=max(r['max_axis_disagreement'] for r in out['rows']),direct=max(r['discrepancy'] for r in out['direct_rows']),scale=max(r['error'] for r in out['scale_rows']))))
