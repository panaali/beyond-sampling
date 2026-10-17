"""Frozen nonzero-sine, conjugacy, identity and complement equivalence."""
import json
from pathlib import Path
import time
import numpy as np
from operator_space import Space,dense,to_complex,to_real

HERE=Path(__file__).resolve().parent
start=time.process_time();out=dict(rows=[])
real=Space(8,4,64);old=dense.Space(8,4,64,optimized=True)
c=np.sin(np.arange(real.size)*.31+.4)/(1+np.arange(real.size))
ac=to_complex(c,real.M)
out['norm_difference']=abs(float(c@c)-float(np.vdot(ac,ac).real))
out['roundtrip_error']=float(np.max(abs(to_real(ac,real.M)-c)))
assert out['norm_difference']<1e-13 and out['roundtrip_error']<1e-13
normals=[[0.,0.,1.],[.6,-.7,.2],[1.,.1,.7],[-.1,.2,1e-6],
         [1.,0.,0.],[-np.sqrt(3),1.,0.]]
for normal in normals:
    normal=np.array(normal)
    for sign in (1,-1):
        for name,coeff in (('arbitrary',c),('constant',real.one())):
            result=real.cut(coeff,sign*normal)
            reference=to_real(old.cut(to_complex(coeff,real.M),sign*normal),real.M)
            complement=real.cut(coeff,-sign*normal)
            row=dict(normal=(sign*normal).tolist(),coefficients=name,
                     parity_error=float(np.max(abs(result-reference))),
                     complement_error=float(np.max(abs(result+complement-coeff))))
            out['rows'].append(row)
            assert row['parity_error']<1e-10 and row['complement_error']<1e-12
out.update(cpu_seconds=time.process_time()-start,real_work=dict(real.work),dense_work=dict(old.work))
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(maximum_parity_error=max(r['parity_error'] for r in out['rows']),
                     cpu_seconds=out['cpu_seconds'])))
