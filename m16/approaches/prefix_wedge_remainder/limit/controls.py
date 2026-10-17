"""Nine fixed exact partitions and their known sector masses; no model calls."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from partition import partition,mass,dyadic,I
from rational_bounds import serial,accounting

HERE=Path(__file__).resolve().parent
OUT=HERE/'controls.json'
assert not OUT.exists(),'Preserve earlier controls'
sys.set_int_max_str_digits(0)
t0=time.process_time()
rounding=[]
for interval in [I.point(F(1,4)),I.point(F(1,3)),I(F(-2,3),F(-1,3)),I(F(-1,3),F(2,3))]:
    got=dyadic(interval);rounding.append({'input':interval,'output':got})
Wpos=[[0,0],[0,1],[1,0],[-1,-1]]
Wthreshold=[[0,0],[1,0],[0,1],[-1,0]]
fixtures=[
 ('zero',[0,0],[0,0],[[0,0]]*4,F(0)),
 ('always_target',[1,0],[0,0],[[0,0],[-1,0],[1,0],[0,0]],F(1,12)),
 ('earlier_tie',[1,0],[0,0],[[0,0],[1,0],[1,0],[-1,0]],F(0)),
 ('later_tie',[1,0],[0,0],[[0,0],[-1,0],[1,0],[1,0]],F(1,12)),
 ('positive_ratio',[0,0],[1,-1],Wpos,F(5,96)),
 ('negative_ratio',[0,0],[1,-1],[[0,0],[1,0],[0,1],[-1,-1]],F(1,32)),
 ('right_rank_boundary',[1,0],[0,1],Wthreshold,F(1,24)),
 ('left_rank_boundary',[1,0],[0,-1],Wthreshold,F(1,48)),
 ('disjoint_tails',[-1,-1],[1,-1],[[0,0],[-1,-1],[1,1],[-2,-2]],F(1,16)),
]
rows=[]
for name,a,b,w,truth in fixtures:
    p=partition(a,b,w);value=mass(p);q=value['joint_correctness_mass']
    assert q.lo<=truth<=q.hi,(name,float(q.lo),float(q.hi),truth)
    assert q.hi-q.lo<F(1,10**30)
    rows.append({'name':name,'partition':p,'mass':value,'exact_truth':truth})
byname={row['name']:row for row in rows}
assert all(r['winner']==0 for r in byname['zero']['partition']['intervals'])
assert all(r['winner']==1 for r in byname['earlier_tie']['partition']['intervals'])
assert all(r['winner']==2 for r in byname['later_tie']['partition']['intervals'])
for name,r in [('right_rank_boundary',F(1)),('left_rank_boundary',F(-1))]:
    boundary=next(b for b in byname[name]['partition']['boundaries'] if b['r']==r)
    assert boundary['winner']==1 and boundary['maximizers']==[1,2]
assert all(b['winner']==0 for b in byname['disjoint_tails']['partition']['boundaries'])
out={'complete':True,'rounding_controls':rounding,'rows':rows,'accounting':accounting(),
     'cpu_seconds':time.process_time()-t0,'network_forward_MC_calls':0,
     'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                      for name in ['partition.py','controls.py','PROTOCOL.md','../rational_bounds.py']},
     'scope':'Nine fixed analytic affine/ReLU partitions; exact known Gaussian sector masses. No trained application or original network forward.'}
with OUT.open('x') as f:json.dump(serial(out),f,indent=2);f.write('\n')
print(json.dumps({'cases':len(rows),'rounding_controls':len(rounding),'cpu_seconds':out['cpu_seconds'],
    'masses':{r['name']:float(r['mass']['joint_correctness_mass'].lo) for r in rows},'accounting':accounting()},indent=2))
