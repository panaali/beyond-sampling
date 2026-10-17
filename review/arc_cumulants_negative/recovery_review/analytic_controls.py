"""Exact finite moment controls and analytic scalar-input covariance; no model calls."""
from fractions import Fraction as F
from math import comb,pi,sqrt
from pathlib import Path
import json,resource,time
HERE=Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
x=[F(i)-F(5,2) for i in range(6)]
a=[F(comb(5,i),16) if i%2==0 else F(0) for i in range(6)]
b=[F(comb(5,i),16) if i%2==1 else F(0) for i in range(6)]
def moments(w):return [sum((wi*xi**k for wi,xi in zip(w,x)),F(0)) for k in range(5)]
ma,mb=moments(a),moments(b);assert ma==mb==[F(1),F(0),F(5,4),F(0),F(65,16)]
pa=sum((wi for wi,xi in zip(a,x) if xi>0),F(0));pb=sum((wi for wi,xi in zip(b,x) if xi>0),F(0));assert(pa,pb)==(F(5,16),F(11,16))
# Independent uniform(-1/4,1/4) jitter: moments combine by binomial convolution.
u=[F(1),F(0),F(1,48),F(0),F(1,1280)]
ca=[sum((F(comb(k,j))*ma[j]*u[k-j] for j in range(k+1)),F(0)) for k in range(5)]
cb=[sum((F(comb(k,j))*mb[j]*u[k-j] for j in range(k+1)),F(0)) for k in range(5)]
assert ca==cb and min(abs(z) for z in x)>F(1,4)
mu=1/sqrt(2*pi);diag=.5-1/(2*pi);off=-1/(2*pi);eig=[.5,.5-1/pi];det=.25-1/(2*pi)
assert abs((diag*diag-off*off)-det)<1e-16 and min(eig)>0
out=dict(complete=True,scalar_gaussian_relu_pair=dict(mean=[mu,mu],covariance=[[diag,off],[off,diag]],eigenvalues=eig,determinant=det,exact_proof='E[X+]=1/sqrt(2pi); E[X+²]=1/2; X+(-X)+=0. Eigenvalues1/2 and1/2−1/pi>0 because pi>2. Support is union of two half-rays, dimension1; covariance rank2.'),moment_pair=dict(support=list(map(str,x)),weights_a=list(map(str,a)),weights_b=list(map(str,b)),moments_0_to_4=list(map(str,ma)),positive_probabilities=[str(pa),str(pb)],common_uniform_jitter_moments=list(map(str,ca)),continuous_jitter_preserves_sign=True,minimum_worst_case_absolute_error=str((pb-pa)/2),minimum_worst_case_squared_error=str(((pb-pa)/2)**2),scope='Information limitation for arbitrary laws and estimators using only these four population moments; not a trained-RNN impossibility or a bound on higher moments/conditioning/weights.'),model_calls=0,random_inputs=0,cpu_seconds=time.process_time()-start)
(HERE/'analytic_controls.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(complete=True,moment_match=True,covariance_rank=2,latent_dimension=1,cpu=out['cpu_seconds'])))
