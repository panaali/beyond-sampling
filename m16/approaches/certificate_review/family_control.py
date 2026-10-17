"""Fresh exact control contained strictly in one initial half-family."""
from fractions import Fraction as F
from pathlib import Path
import json,time
import numpy as np
from simultaneous import recurrence,exact_forms,sqrt_interval,serial
from loader import load,Ops

def main():
    start=time.process_time();wi=(F(1),F(1),F(1,100));wh=((F(-2),F(0),F(0)),(F(-39,20),F(0),F(0)),(F(2,5),F(0),F(0)));wo=((F(0),)*3,(F(0),F(0),F(-1)),(F(-1),F(-1),F(1)));model=(wi,wh,wo)
    rays=((F(1),F(14,5),F(7,5)),(F(1),F(14,5),F(9,5)),(F(1),F(57,20),F(29,20)));A=exact_forms(wi,wh,3)[-1];direct=[];initial_rows=((F(1),F(0),F(0)),(F(0),F(1),F(-1)),(F(-1),F(0),F(1)),(F(1),F(1),F(-2)));slacks=[]
    for x in rays:
        z=[sum(a*b for a,b in zip(row,x)) for row in initial_rows];assert min(z)>0;slacks.append(z)
        actual=recurrence(model,x);frozen=recurrence(model,x,(0,0,1));assert all(v>0 for layer in actual['preactivations'][:2] for v in layer)
        direct.append(dict(input=x,actual=actual,frozen=frozen))
    p=direct[1];assert p['actual']['margin']==F(-51,500) and p['frozen']['margin']==F(169,500);allowances=p['actual']['preactivations'][-1][:2];assert allowances==[F(1,5),F(6,25)] and max(allowances)<p['frozen']['margin']<sum(allowances)
    ref=[]
    for row in A:
        low=high=F(0)
        for x in rays:
            v=sum(a*b for a,b in zip(row,x));lo,hi=sqrt_interval(sum(z*z for z in x));low+=v/(hi if v>=0 else lo);high+=v/(lo if v>=0 else hi)
        ref.append((low/3,high/3))
    assert ref[0][1]<0 and ref[1][1]<0 and ref[2][0]>0
    env=load();c=env['Cone'].from_normals(np.linalg.inv(np.array(rays,float).T));ops=Ops();calls=[];original=env['_frozen_margin']
    def capture(model,t,A,g,truth,P,ops):
        m=original(model,t,A,g,truth,P,ops);calls.append(dict(g=g.tolist(),points=P.tolist(),margins=m.tolist()));return m
    env['_frozen_margin']=capture;pieces=env['_margin_split'](c,np.array(A,float),tuple(np.array(m,float) for m in model),2,2,np.array([-1.,0.,1.]),.1,ops,False)
    assert env['STATS']['cut']==2 and env['STATS']['resolved']==0 and len(pieces)==1 and pieces[0][1].tolist()==[0.,0.,1.]
    out=dict(model=model,rays=rays,initial_half_family_constraints=initial_rows,exact_initial_slacks=slacks,final_forms=A,reference_form_intervals=ref,direct=direct,source_calls=calls,source_stats=env['STATS'],source_gates=pieces[0][1].tolist(),individual_allowances=allowances,sum_allowance=sum(allowances),cpu_seconds=time.process_time()-start,exact_recurrences=6,source_cell_calls=1,scope='Valid full-dimensional recurrent subcone strictly inside one initial half-family; not a claim default full solver visits this precise subcone or fails a trained panel.')
    Path(__file__).with_suffix('.json').write_text(json.dumps(serial(out),indent=2)+'\n');print(json.dumps(serial(dict(stats=out['source_stats'],frozen_margin=p['frozen']['margin'],actual_margin=p['actual']['margin'],allowances=allowances,cpu_seconds=out['cpu_seconds']))))
if __name__=='__main__':main()
