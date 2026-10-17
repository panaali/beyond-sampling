"""Exact-rational contraction control; independently injected correction terms."""
import sys,json,itertools,time
from pathlib import Path
from fractions import Fraction as F
import numpy as np
sys.path.insert(0,'/workspace/work/recover-small_h2_78')
from m16.approaches.tensor_enrichment.additive import Additive
start=time.process_time()
weights=[[F(1,3),F(2,3)],[F(1,6),F(1,3),F(1,2)],[F(1,10),F(2,10),F(3,10),F(4,10)],[F(1,5),F(4,5)]]
shape=tuple(map(len,weights));center=[n//2 for n in shape]
rows=[]
for scale in (F(1),F(-3),F(1,1000000)):
    terms=[]
    for c in range(shape[0]):
        r0=scale*F(c+1)
        fibers=[[r0+scale*F((k+1)*(i-center[k+1]),7) for i in range(shape[k+1])] for k in range(3)]
        terms.append((r0,fibers))
    model=Additive.__new__(Additive);model.weights=[np.array(list(map(float,w))) for w in weights]
    model.terms=[(float(a),[np.array(list(map(float,f))) for f in fs]) for a,fs in terms]
    model.baseline_integral=0.;model.work={'contraction_proxy':0}
    truth=F(0)
    for ix in itertools.product(*map(range,shape)):
        r0,fs=terms[ix[0]];val=sum(fs[k][ix[k+1]] for k in range(3))-2*r0
        w=F(1)
        for k,i in enumerate(ix):w*=weights[k][i]
        truth+=w*val
    value=model.integral();rows.append(dict(scale=str(scale),exact=str(truth),computed=value,error=value-float(truth)))
# Configuration-local central spike: interpolation is exact on every fiber,
# but additive extension is negative when all noncategorical indices differ.
spike_integral=sum(weights[k+1][center[k+1]] for k in range(3))-2
spike_true=np.prod([float(weights[k+1][center[k+1]]) for k in range(3)])
out=dict(rows=rows,spike_additive_integral=str(spike_integral),spike_true_integral=spike_true,spike_off_fiber_value=-2,cpu_seconds=time.process_time()-start)
Path(__file__).with_name('additive_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
