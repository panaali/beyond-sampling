import time
START=time.process_time()
from fractions import Fraction as F
import itertools,json,math,sys
from pathlib import Path
import numpy as np
import kernel as K
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'leaf_mass_review'));import mass_check as M
E=M.E;RULES={'wide':E._rule(24,2),'thin':E._rule(96,3)}
def determinant(A):
 n=len(A);total=0
 for p in itertools.permutations(range(n)):
  sign=(-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n));term=sign
  for i in range(n):term*=A[i][p[i]]
  total+=term
 return total
def exact_controls():
 count=0
 for seed in range(8):
  A=[[F(((i+2)*(j+3)+seed*(i-j+1))%11-5,7) for j in range(4)] for i in range(4)]
  C=[[sum(A[i][k]*A[j][k] for k in range(4))+int(i==j) for j in range(4)] for i in range(4)]
  coeff=K.exact_coefficients(C)
  for t in [F(0),F(1,7),F(3,5),F(1)]:
   R=[[int(i==j)*(1-t)+t*C[i][j] for j in range(4)] for i in range(4)];vs,ps=K.polynomials(coeff,t)
   for omit in range(4):
    ids=[i for i in range(4) if i!=omit];assert vs[omit]==determinant([[R[i][j] for j in ids] for i in ids]);count+=1
   for value,(_,_,k,l) in zip(ps,K.PAIRS):
    # inverse[k,l] numerator is cofactor[l,k], so negative precision numerator.
    rows=[i for i in range(4) if i!=l];cols=[j for j in range(4) if j!=k]
    target=-((-1)**(k+l))*determinant([[R[i][j] for j in cols] for i in rows]);assert value==target;count+=1
 return count
def covariance(rays):
 B=np.linalg.inv(rays.T);S=B@B.T;dg=np.sqrt(np.diag(S));return S/np.outer(dg,dg)
def value(rays):
 C=covariance(rays);p=1/16+sum(math.asin(np.clip(C[i,j],-1,1)) for i,j,_,_ in K.PAIRS)/(8*math.pi)
 if abs(np.linalg.det(rays))<1e-12:return 0.,{}
 key='thin' if max(abs(C[i,j]) for i,j,_,_ in K.PAIRS)>.95 else 'wide';acc,diag=K.evaluate(C,RULES[key]);return p+acc/(4*math.pi**2),diag
class Ops:
 def add(self,n):pass
def main():
 out=dict(exact_identities=exact_controls(),analytic=[],near_singular=[],complete=False);E._RULES=RULES
 for name,rays,truth in [('orthant',np.eye(4),1/16),('order',np.array([[1.,1,0,0],[1.,0,0,0],[0,0,1.,0],[0,0,0,1.]]),1/32),('pairs',np.array([[1.,0,0,0],[.5,math.sqrt(3)/2,0,0],[0,0,1.,0],[0,0,.5,math.sqrt(3)/2]]),1/36)]:
  candidate,diag=value(rays);source=E.orthant(rays,Ops());row=dict(name=name,candidate=candidate,source=source,truth=truth,diagnostics=diag);out['analytic'].append(row);assert abs(candidate-truth)<1e-12
 for rho in [0.,.5,.95,1-2**-20,1-2**-40]:
  s=np.array([1.,-1.,1.,-1.]);C=(1-rho)*np.eye(4)+rho*np.outer(s,s);coeff,setup=K.coefficients(C);row=dict(rho=rho,max_pc_difference=0.,max_exact_pc_error=0.,nodes=0,failures=[],setup=setup)
  for key,rules in RULES.items():
   for idx,(t,w,jac) in enumerate(rules):
    try:
     pcs=K.correlations(coeff,float(t));O=np.linalg.inv(np.eye(4)+t*(C-np.eye(4)));old=[-O[k,l]/math.sqrt(O[k,k]*O[l,l]) for _,_,k,l in K.PAIRS]
     row['max_pc_difference']=max(row['max_pc_difference'],max(abs(a-b) for a,b in zip(pcs,old)));r=F(float(t))*F(rho);truth=[float(s[k]*s[l])*float(r/(1+2*r)) for _,_,k,l in K.PAIRS];row['max_exact_pc_error']=max(row['max_exact_pc_error'],max(abs(a-b) for a,b in zip(pcs,truth)));row['nodes']+=1
    except Exception as exc:row['failures'].append(dict(rule=key,node=idx,t=float(t),error=repr(exc)))
  out['near_singular'].append(row)
 out.update(complete=True,gate_pass=all(not r['failures'] and r['max_exact_pc_error']<=1e-12 for r in out['near_singular']),cpu=time.process_time()-START);(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
