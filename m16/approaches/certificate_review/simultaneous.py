"""Exact recurrent counterexample and unchanged-source finite replay."""
from pathlib import Path
from fractions import Fraction as F
from math import isqrt
import json,time
import numpy as np
from loader import load,Ops
HERE=Path(__file__).parent

def serial(v):
    if isinstance(v,F):return str(v)
    if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [serial(x) for x in v]
    return v

def recurrence(model,x,freeze=None):
    wi,wh,wo=model;h=[F(0)]*len(wi);states=[];pre=[]
    for t,u in enumerate(x):
        z=[sum(wh[i][j]*h[j] for j in range(len(h)))+wi[i]*u for i in range(len(h))]
        h=[z[i]*int(freeze[i]) for i in range(len(h))] if freeze is not None and t==len(x)-1 else [max(F(0),v) for v in z]
        pre.append(z);states.append(h)
    out=[sum(a*b for a,b in zip(row,h)) for row in wo];winner=max(range(len(out)),key=lambda i:out[i]);return dict(preactivations=pre,states=states,outputs=out,winner=winner,margin=out[2]-max(out[:2]))

def sqrt_interval(x,bits=64):
    scale=1<<bits;k=isqrt((x.numerator*scale*scale)//x.denominator);return F(k,scale),F(k+1,scale)

def exact_forms(wi,wh,T):
    A=[[wi[i]*int(j==0) for j in range(T)] for i in range(len(wi))];out=[A]
    for t in range(1,T):
        A=[[sum(wh[i][k]*A[k][j] for k in range(len(wi)))+wi[i]*int(j==t) for j in range(T)] for i in range(len(wi))];out.append(A)
    return out

def main():
    start=time.process_time();wi=(F(1),F(1),F(1,100));wh=((F(-2),F(0),F(0)),(F(-39,20),F(0),F(0)),(F(3,4),F(0),F(0)));rays=((F(1),F(3),F(3,2)),(F(1),F(3),F(5,2)),(F(1),F(13,4),F(7,4)));A=exact_forms(wi,wh,3)[-1];rows=[]
    for name,target in [('two_gates',(F(-1),F(-1),F(1))),('single_adverse_gate',(F(-1),F(0),F(1))),('individual_trigger',(F(-1),F(-1),F(1,4)))]:
        wo=((F(0),)*3,(F(0),F(0),F(-1)),target);model=(wi,wh,wo);direct=[]
        for x in rays:
            assert x[1]>x[2]>x[0]>0
            actual=recurrence(model,x);frozen=recurrence(model,x,(0,0,1));assert all(v>0 for layer in actual['preactivations'][:2] for v in layer)
            assert actual['preactivations'][-1]==[sum(a*b for a,b in zip(row,x)) for row in A]
            direct.append(dict(input=x,actual=actual,frozen=frozen))
        # Exact interval check of the normalized-ray mean used by _ref.
        ref_forms=[]
        for row in A:
            low=high=F(0)
            for x in rays:
                v=sum(a*b for a,b in zip(row,x));lo,hi=sqrt_interval(sum(z*z for z in x));low+=v/(hi if v>=0 else lo);high+=v/(lo if v>=0 else hi)
            ref_forms.append((low/3,high/3))
        assert ref_forms[0][1]<0 and ref_forms[1][1]<0 and ref_forms[2][0]>0
        env=load();R=np.array(rays,float);N=np.linalg.inv(R.T);c=env['Cone'].from_normals(N);ops=Ops();captured=[];original=env['_frozen_margin']
        def capture(model,t,A,g,truth,P,ops):
            margins=original(model,t,A,g,truth,P,ops);captured.append(dict(t=t,g=g.tolist(),points=P.tolist(),margins=margins.tolist()));return margins
        env['_frozen_margin']=capture
        pieces=env['_margin_split'](c,np.array(A,float),tuple(np.array(m,float) for m in model),2,2,np.array([-1.,0.,1.]),.1,ops,False)
        stats=env['STATS'].copy();gates=[g.tolist() for _,g in pieces]
        if name=='two_gates':
            assert stats['resolved']==0 and len(pieces)==1 and gates==[[0.,0.,1.]]
            point=direct[1];assert point['actual']['margin']==F(-11,40) and point['frozen']['margin']==F(31,40)
            allowance=[abs(point['actual']['preactivations'][-1][i]) for i in (0,1)];assert allowance==[F(1,2),F(11,20)] and all(v<point['frozen']['margin'] for v in allowance) and sum(allowance)>point['frozen']['margin']
        elif name=='single_adverse_gate':assert stats['resolved']==0 and all(p['actual']['winner']==2 for p in direct)
        else:assert stats['resolved']>0 and len(pieces)>1
        rows.append(dict(name=name,model=model,rays=rays,final_forms=A,direct=direct,exact_reference_form_intervals=ref_forms,source_stats=stats,source_gates=gates,source_margin_calls=captured,inherited_ops=ops.n))
    out=dict(cases=rows,cpu_seconds=time.process_time()-start,exact_recurrences=18,source_cell_calls=3,trained_forwards=0,scope='Generic cell-rule soundness counterexample; no saved trained accuracy result refuted.')
    (HERE/'simultaneous.json').write_text(json.dumps(serial(out),indent=2)+'\n');print(json.dumps(dict(rows=[dict(name=r['name'],stats=r['source_stats'],gates=r['source_gates']) for r in rows],cpu_seconds=out['cpu_seconds'])))
if __name__=='__main__':main()
