"""Independent direct angular integration of a fresh two-input hidden law."""
import importlib.util,json,math,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import mpmath as mp
start=time.process_time();mp.mp.dps=50
path=Path('/workspace/work/recover-small_h2_78/m16/approaches/history_distribution/distribution.py')
spec=importlib.util.spec_from_file_location('history_moment_review',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
wi=np.array([3/5,-7/4]);wh=np.array([[-2/3,5/4],[7/8,-3/7]])
radial,w,_=module.rayleigh(4,defaultdict(float));rows=[]
for sign in (-1,1):
    states,weights,report=module.conditional(wi,wh,sign,2,radial,w,defaultdict(float))
    # Treat stored weights as exact reals; direct trig integration uses its
    # own breakpoint construction and never reads the author's panel moments.
    W=[mp.mpf(float(x)) for x in wi]
    A=[[mp.mpf(float(x)) for x in row] for row in wh]
    c=[sum(A[i][j]*max(sign*W[j],0) for j in range(2)) for i in range(2)]
    cuts=[-mp.pi/2,mp.pi/2]
    for a,b in zip(c,W):
        root=mp.atan2(-a,b)
        for shift in (-mp.pi,mp.mpf(0),mp.pi):
            if -mp.pi/2<root+shift<mp.pi/2:cuts.append(root+shift)
    cuts=sorted(set(cuts))
    def d(theta,i):return max(c[i]*mp.cos(theta)+W[i]*mp.sin(theta),0)
    mean=[mp.sqrt(mp.pi/2)*mp.quad(lambda theta:d(theta,i),cuts)/mp.pi for i in range(2)]
    second=[[2*mp.quad(lambda theta:d(theta,i)*d(theta,j),cuts)/mp.pi for j in range(2)] for i in range(2)]
    me=float(np.max(abs(np.array([float(x) for x in mean])-weights@states)))
    se=float(np.max(abs(np.array([[float(x) for x in row] for row in second])-report['second_moment'])))
    assert me<1e-13 and se<1e-13
    rows.append(dict(sign=sign,direct_mean=[str(x) for x in mean],direct_second=[[str(x) for x in row] for row in second],
                     maximum_mean_error=me,maximum_second_error=se,
                     covariance_loss_eigenvalues=report['covariance_loss_eigenvalues']))
out=dict(rows=rows,cpu_seconds=time.process_time()-start,
         scope='Independent50-digit direct angular moments on fresh mixed-sign model; no suffix accuracy or quadrature enclosure claim.')
Path(__file__).with_name('history_moments_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
