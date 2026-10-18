"""Analytic obtuse cones, exact scaling/minimum and tail specialization controls."""
import json,math,time,resource
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from .radial import bounds,probability_bound,integrate
HERE=Path(__file__).parent

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();out=dict(rows=[])
    for a in (0.,.4,.9):
        for b in (0.,.4,.9):
            N=np.array([[1,0,0,0],[a,math.sqrt(1-a*a),0,0],[0,0,1,0],[0,0,b,math.sqrt(1-b*b)]]);R=np.linalg.inv(N).T;truth=(.25+math.asin(a)/(2*math.pi))*(.25+math.asin(b)/(2*math.pi));base=bounds(R)
            for kind,Y in [('base',R),('scale',R*8),('permute',R[:,[2,0,3,1]])]:
                cert=bounds(Y);bd=probability_bound(cert,20);val=integrate(Y,20)
                if kind=='scale':assert F(cert['a'])==64*F(base['a']) and F(cert['b'])==64*F(base['b']) and abs(F(cert['det']))==4096*abs(F(base['det']))
                out['rows'].append(dict(a=a,b=b,transform=kind,certificate=cert,value=val,truth=truth,difference=val-truth,**bd,passes=abs(val-truth)<=bd['bound']+5e-14))
    eye=bounds(np.eye(4));assert F(eye['a'])==F(1,4) and F(eye['b'])==1
    expected=249/(6*math.pi**2*3.**19);assert abs(probability_bound(eye,20)['bound']/expected-1)<1e-14
    const=probability_bound(dict(a='1',b='1',det='1'),20);assert const['bound']==0
    out.update(complete=True,all_pass=all(r['passes'] for r in out['rows']),cpu=time.process_time()-start,diagonal_certificate=eye,constant_tail=const)
    (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2));assert out['all_pass']
if __name__=='__main__':main()
