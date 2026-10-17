"""Inherited analytic fixtures plus exact256-term Dirichlet checks."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from collections import Counter as Counts
import json,math,time
import numpy as np
from ..positive_mass_budget import controls as original
from .primitive import estimate,geometry,full_geometry,moments,Counter
HERE=Path(__file__).parent

def main():
    original.HERE=HERE;original.estimate=estimate;original.geometry=geometry;original.main();start=time.process_time();rays=np.load(HERE.parent/'class_region_t4/quadratic_exact_mass_calls.npz')['rays'];rows=[]
    for i in (0,2115,4231,6346,3222,3219,3221,3224,5962,3251,3254,3836,2567,2577,5448,2020):
        A=[[F(float(x)) for x in row] for row in rays[i]];G=[[sum(x*y for x,y in zip(a,b)) for b in A] for a in A];mu=(sum(map(sum,G))+sum(G[k][k] for k in range(4)))/20;eq2=F(0)
        for j,k,l,m in product(range(4),repeat=4):
            factor=math.prod(math.factorial(v) for v in Counts((j,k,l,m)).values());eq2+=G[j][k]*G[l][m]*F(factor,840)
        var=eq2-mu*mu;_,_,_,data=full_geometry(rays[i],Counter());mf,vf,ef=moments(data);rows.append(dict(index=i,exact_mean=str(mu),exact_second=str(eq2),exact_variance=str(var),mean_error=float(F(mf.v)-mu),second_error=float(F(ef.v)-eq2),variance_error=float(F(vf.v)-var),floating_variance=vf.v,relative_variance_error=float(F(vf.v)/var-1) if var else None))
    out=dict(rows=rows,cpu=time.process_time()-start,complete=True);(HERE/'moment_controls.json').write_text(json.dumps(out,indent=2)+'\n');print('maximum absolute moment error',max(abs(r[k]) for r in rows for k in ('mean_error','second_error','variance_error')))
if __name__=='__main__':main()
