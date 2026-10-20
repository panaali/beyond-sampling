#!/usr/bin/env python3
"""Independent off-suite k=57 counterexample; rational density/volume witness."""
import hashlib, importlib.util, json, math, sys, time
from fractions import Fraction as F
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
p=Path(sys.argv[1]); spec=importlib.util.spec_from_file_location('producer',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
start=time.process_time();torch.set_num_threads(1)
k=57; a0,a1=F(1,2**k),F(1,2**(k-1))
lo=[-.75,float(a0)]+[0.]*7;hi=[-.25,float(a1)]+[1.]*7
value=float(m.upper(torch.tensor([lo],dtype=torch.float64),torch.tensor([hi],dtype=torch.float64),nr=4,ng=8)[0])
# Different subregion from author/root: r[-3/8,-1/4], g[5/2,11/4], seven lows[1/8,1/4].
gl,gh=F(5,2),F(11,4)
qmax=F(3,8)**2+F(3,4)**2+(F(3,4)+a1)**2+7*F(5,8)**2
power=math.ceil(gh**2*qmax/2)
lower=gl**9*F(7,44)**5/F(3)**power*F(1,8)*(a1-a0)*F(1,8)**7*(gh-gl)
assert lower>F(value)>0
res=dict(k=k,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),float_upper=value,rational_mass_lower=str(lower),
         lower_as_float=float(lower),underestimate_factor_lower=float(lower/F(value)),passed=bool(lower>F(value)),
         cpu_seconds=time.process_time()-start,scope='off-suite synthetic box; no claim about production frontier')
Path(__file__).with_name('T463_MASS_REVIEW.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res))
