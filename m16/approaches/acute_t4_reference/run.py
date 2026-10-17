"""Coordinate-free acute-ray Gaussian simplex reference and exact sign audit."""
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import hashlib,json,math,time,resource
import numpy as np
from numpy.polynomial.legendre import leggauss
HERE=Path(__file__).parent
CAPTURE=HERE.parent/'class_region_t4/quadratic_exact_mass_calls.npz'
HASH='07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581'

def exact_signs(R):
    A=[[Fraction(float(x)) for x in row] for row in R];dots=[sum(A[i][k]*A[j][k] for k in range(4)) for i,j in combinations(range(4),2)]
    return [int(d>0)-int(d<0) for d in dots]

def positive_acute(R,n=20):
    R=R/np.sqrt(np.sum(R*R,axis=1))[:,None]
    x,w=leggauss(n);x=(x+1)/2;w=w/2;u,v,z=np.meshgrid(x,x,x,indexing='ij');a,b,c=np.meshgrid(w,w,w,indexing='ij');y=np.stack([u,(1-u)*v,(1-u)*(1-v)*z,(1-u)*(1-v)*(1-z)],axis=-1).reshape(-1,4);weights=(a*b*c*(1-u)**2*(1-v)).ravel();points=y@R;det=abs(np.linalg.det(R));value=det*np.dot(weights,1/np.sum(points*points,axis=1)**2)/(2*math.pi**2);bound=det*(12*n+9)/(6*math.pi**2*3.**(n-1));return float(value),float(bound),float(det),float(np.linalg.cond(R))

def controls():
    start=time.process_time();rows=[];Q=np.eye(4);Q[0,0]=Q[2,2]=.6;Q[0,2]=.8;Q[2,0]=-.8
    for a in (-.9,-.4,0):
        for b in (-.9,-.4,0):
            N=np.array([[1,0,0,0],[a,math.sqrt(1-a*a),0,0],[0,0,1,0],[0,0,b,math.sqrt(1-b*b)]]);R=np.linalg.inv(N).T;truth=(.25+math.asin(a)/(2*math.pi))*(.25+math.asin(b)/(2*math.pi))
            for kind,Y in [('base',R),('scale',R*np.exp2([-20.,20.,-10.,10.])[:,None]),('rotate',R@Q)]:
                val,bnd,det,cond=positive_acute(Y);rows.append(dict(a=a,b=b,transform=kind,value=val,truth=truth,difference=val-truth,integration_only_bound=bnd,passes=abs(val-truth)<=bnd+5e-14))
    neg=np.eye(4);neg[1,0]=-.5;signs=exact_signs(neg);assert -1 in signs
    out=dict(rows=rows,negative_control_signs=signs,all_pass=all(r['passes'] for r in rows),cpu=time.process_time()-start)
    (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');assert out['all_pass'];return out

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();ctrl=controls();assert hashlib.sha256(CAPTURE.read_bytes()).hexdigest()==HASH
    rays=np.load(CAPTURE)['rays'];ids=json.loads((HERE.parent/'positive_t4_corpus/inventory.json').read_text())['uncovered'];source=np.load(HERE.parent/'cone_mass4_review/quadratic_values.npz')['values'];out=dict(rows=[],uncovered=[],complete=False,full_accuracy_certified=False,full_cost_certified=False,controls_cpu=ctrl['cpu']);values=np.full(len(rays),np.nan);bounds=np.full(len(rays),np.nan)
    def save():
        ok=np.isfinite(values);out.update(cpu=time.process_time()-start,covered=int(ok.sum()),positive_mass_sum=math.fsum(values[ok]),integration_only_bound_sum=math.fsum(bounds[ok]),source_covered_sum=math.fsum(source[ok]),source_minus_positive=math.fsum(source[ok]-values[ok]),known_integrand_arithmetic=int(ok.sum())*312000)
        (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(HERE/'acute_values.npz',values=values,bounds=bounds)
    for i in ids:
        if time.process_time()-start>18:out['stop_reason']='18CPU soft cap within20CPU review lane';break
        signs=exact_signs(rays[i]);row=dict(index=i,exact_pair_signs=signs);out['rows'].append(row)
        if -1 in signs:out['uncovered'].append(i);row['rotation_obstruction']=True
        else:
            values[i],bounds[i],det,cond=positive_acute(rays[i]);row.update(value=values[i],bound=bounds[i],normalized_det=det,condition=cond,source=float(source[i]),difference=float(source[i]-values[i]))
        if len(out['rows'])%64==0:save()
    else:out['complete']=True
    save();print(json.dumps({k:v for k,v in out.items() if k not in ('rows','uncovered')},indent=2));print('nonacute',len(out['uncovered']))
if __name__=='__main__':main()
