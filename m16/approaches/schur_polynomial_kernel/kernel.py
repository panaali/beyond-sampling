"""General-diagonal cubic Schur cofactors, with explicit invalid states."""
import itertools,math
import numpy as np
PAIRS=[(i,j,*[k for k in range(4) if k not in (i,j)]) for i in range(4) for j in range(i+1,4)]
def det3(a):
 return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
def coefficients(C):
 # Generic scalar arithmetic also permits independent exact Fraction controls.
 minors=[]
 for omit in range(4):
  ids=[i for i in range(4) if i!=omit];a=[[C[i][j] for j in ids] for i in ids]
  trace=sum(a[i][i] for i in range(3))
  two=sum(a[i][i]*a[j][j]-a[i][j]*a[j][i] for i in range(3) for j in range(i+1,3))
  minors.append((trace,two,det3(a)))
 nums=[]
 for i,j,k,l in PAIRS:
  di,dj=C[i][i]-1,C[j][j]-1;r=C[i][j];e=C[k][l]
  ac=C[i][k]*C[i][l];bd=C[j][k]*C[j][l]
  cross=C[i][k]*C[j][l]+C[j][k]*C[i][l]
  nums.append((e,e*(di+dj)-ac-bd,e*(di*dj-r*r)-dj*ac-di*bd+r*cross))
 return minors,nums
def polynomials(coeff,t):
 minors,nums=coeff;u=1-t;u3=u*u*u;u2t=u*u*t;ut2=u*t*t;t3=t*t*t
 vs=[u3+a*u2t+b*ut2+c*t3 for a,b,c in minors]
 ps=[t*(a+t*(b+t*c)) for a,b,c in nums]
 return vs,ps
class InvalidSchur(ArithmeticError):pass
def correlations(coeff,t):
 vs,ps=polynomials(coeff,t)
 if min(vs)<=0 or not all(math.isfinite(v) for v in vs):raise InvalidSchur(f'nonpositive/nonfinite minor at t={t}: {vs}')
 return [p/math.sqrt(vs[k]*vs[l]) for p,(_,_,k,l) in zip(ps,PAIRS)]
def evaluate(C,rules):
 coeff=coefficients(C.tolist());acc=0.;excursions=0;maxexcursion=0.
 for t,w,jac in rules:
  pcs=correlations(coeff,float(t))
  for pc,(i,j,k,l) in zip(pcs,PAIRS):
   if not math.isfinite(pc):raise InvalidSchur('nonfinite partial correlation')
   if abs(pc)>1:excursions+=1;maxexcursion=max(maxexcursion,abs(pc)-1)
   acc+=w*jac*C[i,j]*math.asin(np.clip(pc,-1,1))/math.sqrt(max(1e-300,1-(t*C[i,j])**2))
 return acc,dict(excursions=excursions,max_excursion=maxexcursion)
