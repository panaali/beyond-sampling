import hashlib,json,math
from pathlib import Path
import numpy as np
out=Path(__file__).parent
cases=[]
def add(name,lo,hi): cases.append(dict(name=name,lo=lo.copy(),hi=hi.copy()))
lo=[-4.]+[0.]*8; hi=[4.]*9
add('baseline_root',lo,hi)
for d in range(7):
    hi[d]=(lo[d]+hi[d])/2
    add('baseline_split_'+str(d+1),lo,hi)
rng=np.random.default_rng(463)
for i in range(16):
    z=np.sort(rng.uniform(size=(2,9)),axis=0)*4; z[:,0]=z[:,0]*2-4
    add('random_'+str(i),z[0].tolist(),z[1].tolist())
for k in [20,40,50,51,52,53,54,55,56,60,80,100]:
    add('thin_a_'+str(k),[-.75,2.**-k]+[0.]*7,[-.25,2.**(-k+1)]+[1.]*7)
lo=[-.75,0.]+[0.]*7; hi=[-.25,1.]+[1.]*7
for d in [2,8]:
    l=lo.copy();h=hi.copy();l[d]=1.;h[d]=math.nextafter(1.,math.inf)
    add('ulp_low_'+str(d),l,h)
for k in [52,54,80]:
    h=hi.copy();h[2]=2.**-k;add('thin_low_'+str(k),lo,h)
h=hi.copy();h[1]=0.;add('zero_a',lo,h)
for r in [-.5,4.]:
    l=lo.copy();h=hi.copy();l[0]=r;h[0]=math.nextafter(r,math.inf);add('narrow_r_'+str(r),l,h)
assert len(cases)==44
payload=json.dumps(dict(seed=463,cases=cases),sort_keys=True,indent=2)+'\n'
(out/'controls.json').write_text(payload)
print(hashlib.sha256(payload.encode()).hexdigest())
