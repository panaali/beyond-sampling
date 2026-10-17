"""Fixed exact radial controls and scale-invariant truncation checks; no integrations."""
import time
START=time.process_time()
from pathlib import Path
from fractions import Fraction as F
import json,resource
import numpy as np
import mpmath as mp
import primitive as A
P=Path(__file__).resolve().parent

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(20,22));tracker,radial=A.load();rows=[]
 nonacute=np.array([[1,0,0,0],[-2,1,0,0],[0,0,1,0],[0,0,0,1]],float)
 cases=[('orthant',np.eye(4),F(1,4),None),('diagonal',np.diag([1.,2.,3.,4.]),F(144,205),None),('nonacute_uncovered',nonacute,F(1,12),None),('nonacute_kkt',nonacute,F(1,12),dict(method='exact_kkt',certificate=dict(a='1/12',b='5',det='1',y=['7/12','1/4','1/12','1/12']))),('boundary',np.array([[1,0,0,0],[2,1,0,0],[2,0,1,0],[2,0,0,1]],float),F(1),None)]
 for name,R,minimum,reused in cases:
  for power in (-5,0,5):
   cert=None
   if reused:
    cert=dict(method=reused['method'],certificate=dict(reused['certificate']))
    for key,degree in (('a',2),('b',2),('det',4)):cert['certificate'][key]=str(F(cert['certificate'][key])*F(2)**(power*degree))
   row=A.certificate(R*2.**power,tracker,radial,cert);row.update(name=name,power=power,exact_minimum=str(minimum*F(2)**(2*power)))
   if row['mode']=='bounded':
    assert F(row['a'])<=F(row['exact_minimum'])<=F(row['b']);row['quadrature']=A.bound(row);high=A.bound(row,100)
    with mp.workdps(100):row['precision_relative_difference']=float(abs(mp.mpf(row['quadrature']['bound_decimal'])-mp.mpf(high['bound_decimal']))/mp.mpf(high['bound_decimal']));assert row['precision_relative_difference']<1e-70
   else:assert name=='nonacute_uncovered'
   rows.append(row)
 for name,_,_,_ in cases:
  group=[r for r in rows if r['name']==name];assert len({r['mode'] for r in group})==1
  if group[0]['mode']=='bounded':assert len({r['quadrature']['bound_decimal'] for r in group})==1
 singular=A.certificate(np.zeros((4,4)),tracker,radial);assert singular['mode']=='uncovered'
 out=dict(complete=True,rows=rows,singular=singular,cpu=time.process_time()-START,no_integrations=True)
 (P/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,cases=len(rows),uncovered=sum(r['mode']=='uncovered' for r in rows),cpu=out['cpu'])))
if __name__=='__main__':main()
