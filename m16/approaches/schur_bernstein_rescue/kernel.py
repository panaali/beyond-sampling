"""Exact preprocessed Bernstein Schur cofactors; no fallback or PSD repair."""
from fractions import Fraction as F
import math,time
import numpy as np
PAIRS=[(i,j,*[k for k in range(4) if k not in (i,j)]) for i in range(4) for j in range(i+1,4)]
def det3(a):
 return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
def exact_coefficients(C):
 minors=[]
 for omit in range(4):
  ids=[i for i in range(4) if i!=omit];a=[[C[i][j] for j in ids] for i in ids]
  minors.append((sum(a[i][i] for i in range(3)),sum(a[i][i]*a[j][j]-a[i][j]*a[j][i] for i in range(3) for j in range(i+1,3)),det3(a)))
 nums=[]
 for i,j,k,l in PAIRS:
  ac=C[i][k]*C[i][l];bd=C[j][k]*C[j][l];r=C[i][j];e=C[k][l]
  b=e*(C[i][i]+C[j][j])-ac-bd
  c=e*(C[i][i]*C[j][j]-r*r)-C[j][j]*ac-C[i][i]*bd+r*(C[i][k]*C[j][l]+C[j][k]*C[i][l])
  nums.append((e,b,c))
 return minors,nums
def coefficients(C):
 start=time.process_time();exact=exact_coefficients([[F(float(x)) for x in row] for row in C]);flat=[x for group in exact for row in group for x in row]
 return tuple([tuple(float(x) for x in row) for row in group] for group in exact),dict(coefficient_count=len(flat),max_numerator_bits=max(x.numerator.bit_length() for x in flat),max_denominator_bits=max(x.denominator.bit_length() for x in flat),setup_cpu=time.process_time()-start)
def polynomials(coeff,t):
 minors,nums=coeff;u=1-t;u3=u*u*u;u2t=u*u*t;ut2=u*t*t;t3=t*t*t
 return [u3+a*u2t+b*ut2+c*t3 for a,b,c in minors],[a*u2t+b*ut2+c*t3 for a,b,c in nums]
class InvalidSchur(ArithmeticError):pass
def correlations(coeff,t):
 vs,ps=polynomials(coeff,t)
 if min(vs)<=0 or not all(math.isfinite(v) for v in vs):raise InvalidSchur(f'nonpositive/nonfinite minor at t={t}: {vs}')
 return [p/math.sqrt(vs[k]*vs[l]) for p,(_,_,k,l) in zip(ps,PAIRS)]
def evaluate(C,rules):
 coeff,diag=coefficients(C);acc=0.;excursions=0;maxexcursion=0.
 for t,w,jac in rules:
  for pc,(i,j,k,l) in zip(correlations(coeff,float(t)),PAIRS):
   if not math.isfinite(pc):raise InvalidSchur('nonfinite partial correlation')
   if abs(pc)>1:excursions+=1;maxexcursion=max(maxexcursion,abs(pc)-1)
   acc+=w*jac*C[i,j]*math.asin(np.clip(pc,-1,1))/math.sqrt(max(1e-300,1-(t*C[i,j])**2))
 diag.update(excursions=excursions,max_excursion=maxexcursion)
 return acc,diag
