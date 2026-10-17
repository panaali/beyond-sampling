"""Independent scalar feature and span/count controls; no fitting."""
import cmath,json,math,time
from pathlib import Path
import numpy as np
from run import features
HERE=Path(__file__).parent

def legendre(degree,x):
 if degree==0:return 1.
 a,b=1.,x
 for n in range(1,degree):a,b=b,((2*n+1)*x*b-n*a)/(n+1)
 return b

def scalar(x,family):
 ids=sorted(range(9),key=lambda i:(-float(x[i]),i));r=[float(x[i]) for i in ids];out=[r[0],r[1],float(x[9])]
 if family=='fourier':
  for value,pos in [(r[0]-r[1],ids[0]),(r[1],ids[1])]:
   for freq in range(1,5):z=cmath.exp(2j*math.pi*freq*pos/9);out.extend([value*z.real,value*z.imag])
 elif family=='position_polynomial':
  for value,pos in [(r[0]-r[1],ids[0]),(r[1],ids[1])]:
   for degree in range(1,9):
    v=[legendre(degree,2*i/8-1) for i in range(9)];mean=sum(v)/9;norm=math.sqrt(sum((z-mean)**2 for z in v)/9);out.append(value*(v[pos]-mean)/norm)
 else:out.extend(r[2:]);out.extend(map(abs,r));out.append(abs(float(x[9])))
 if family!='rank_values':out.append(r[1]*cmath.exp(2j*math.pi*(ids[0]-ids[1])/9).real)
 return out

def main():
 start=time.process_time();x=np.array([[((i*17+j*13)%31-15)/7 for j in range(10)] for i in range(31)]+[[0.]*10,[1.]*10,[-1.]*10]);rows=[]
 for family in ('fourier','position_polynomial','rank_values'):
  actual,names=features(x,family);ref=np.array([scalar(v,family) for v in x]);err=float(np.max(abs(actual-ref)));assert actual.shape==(34,20) and len(names)==20 and err<1e-12
  hom=max(float(np.max(abs(features(x*s,family)[0]-s*actual))) for s in (0,.1,2,10));assert hom<1e-11;rows.append(dict(family=family,max_feature_error=err,positive_homogeneity_error=hom))
 # Exact complete position-function dimension, independent of model/input data.
 f=np.array([[cmath.exp(2j*math.pi*k*p/9).real if trig==0 else cmath.exp(2j*math.pi*k*p/9).imag for k in range(1,5) for trig in (0,1)] for p in range(9)])
 p=np.array([[legendre(k,2*j/8-1) for k in range(1,9)] for j in range(9)]);p-=p.mean(0);p/=np.sqrt((p*p).mean(0));coef=np.linalg.lstsq(f,p,rcond=None)[0];res=float(np.max(abs(f@coef-p)))
 assert np.linalg.matrix_rank(f)==np.linalg.matrix_rank(p)==8 and res<1e-13
 out=dict(feature_controls=rows,position_span_rank=8,position_span_max_residual=res,transform_condition=float(np.linalg.cond(coef)),cpu_seconds=time.process_time()-start);(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
