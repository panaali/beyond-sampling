from fractions import Fraction as F
import json,time,itertools
from pathlib import Path
import numpy as np
from families import quadratic,nearest,engine,Ops
start=time.process_time();masses=[];parts=[]
def positive_probability(T,a,b):
 if a==0:return 1-F(1,2**T)
 if b==0:return 1-F(T+1,2**T)
 return (F(T,2)-2+F(T+2,2**T))/(T-2)
for T in range(3,11):
 byclass=[F(0) for _ in range(T)];pos=F(0)
 for a in range(T):
  for b in range(T):
   if a==b:continue
   p=positive_probability(T,a,b);assert 0<p<1
   pos+=p/(T*(T-1));byclass[b]+=F(1,T*(T-1))
 assert pos==F(1,2) and byclass==[F(1,T)]*T
for T in (3,4):
 for kind,fn in [('quadratic',quadratic),('nearest',nearest)]:
  ops=Ops();hs=fn(T,ops);v=0;errors=[];bylabel=[0.]*T
  for h in hs:
   z=engine.mass(h['cone'],ops);v+=z;bylabel[h['truth']]+=z
   if kind=='quadratic':
    a,b=h['pair'];p=positive_probability(T,a,b)
    if h['sigma']<0:p=1-p
    errors.append(z-float(p/(T*(T-1))))
  masses.append(dict(T=T,kind=kind,families=len(hs),total=v,bylabel=bylabel,max_chart_error=max(map(abs,errors)) if errors else None,ops_proxy=ops.n))
# Test deterministic rational witnesses with none on nearest-facet split boundaries.
for T in (3,4,5):
 hs=nearest(T,Ops());n=0;bad=0;maxover=0
 vals=[2**k for k in range(T)]
 for p in itertools.permutations(vals):
  for shift in(-3*2**T,1):
   x=np.array(p)+shift
   # Count closed cone membership; ignore true/split boundaries explicitly.
   ds=[h['cone'].N@x for h in hs]
   if any(np.any(np.abs(z)<1e-10) for z in ds):continue
   matched=[i for i,z in enumerate(ds) if min(z)>0]
   bad+=int(len(matched)!=1);n+=1;maxover=max(maxover,len(matched))
 parts.append(dict(T=T,families=len(hs),points=n,failures=bad,max_membership=maxover))
 assert bad==0 and n>0
out=dict(mass_checks=masses,nearest_partition=parts,cpu=time.process_time()-start)
Path(__file__).with_name('geometry_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
