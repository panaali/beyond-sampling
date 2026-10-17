"""Independent rule5 label-cost correction on two frozen T114 panels."""
from pathlib import Path
from decimal import Decimal as D,getcontext
import json,time
import numpy as np
getcontext().prec=60
start=time.process_time();base=Path('/workspace/work/recover-small_h2_56/m16/approaches/training_axis_review')
out=[]
for filename in ['absolute_k24_s4.json','confirmation_k24_s4.json']:
    data=json.loads((base/filename).read_text(),parse_float=D);rows=data['rows']
    loss=[];variance=[];oldvariance=[]
    for r in rows:
        count=r['t75']['sample_counts']
        assert count['mul']==56 and count['add']==41 and count['cmp']==17 and count['correctness_comparison']==1
        assert r['t75']['sample_ops']==115
        cost=D(r['t75']['own_ops'])
        p=r['truth'];loss.append((r['t75']['value']-p)**2)
        variance.append(p*(1-p)*D(112)/cost);oldvariance.append(p*(1-p)*D(115)/cost)
    ratio=sum(loss)/sum(variance);old=sum(loss)/sum(oldvariance)
    assert abs(ratio-old*D(115)/D(112))<D('1e-55')
    l=np.array(list(map(float,loss)));v=np.array(list(map(float,variance)));rng=np.random.default_rng(991);boot=[]
    for _ in range(1000):
        counts=np.bincount(rng.integers(len(rows),size=len(rows)),minlength=len(rows));boot.append(float(counts@l/(counts@v)))
    out.append({'file':filename,'n':len(rows),'ratio_112':str(ratio),'historical_ratio_115':str(old),
                'bootstrap_ci_112':np.quantile(boot,[.025,.975]).tolist(),
                'individual_losses_112':sum(a>=b for a,b in zip(loss,variance))})
result={'panels':out,'cpu_seconds':time.process_time()-start,'sample_scalar_ops_excluding_free_input_label':112,
        'scope':'T75 reimplementation scalar convention; fibre hybrid counter is not reconciled by this correction'}
Path(__file__).with_name('label_cost_review_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
