"""Algebra checks independent of the pending exact walker."""
import numpy as np
from probe import metrics, weighted_top_share, linear_proxies, walker

f = np.array([1.,2.,4.,8.])
c = np.array([2.,3.,5.,7.])
m = metrics(f, f, c, 0.)
assert m['importance_variance'] < 1e-12
assert abs(m['expected_proposal_cost']-np.dot(f,c)/f.sum()) < 1e-12
m = metrics(f, np.ones(4), c, 0.)
assert abs(m['variance_gain']-1) < 1e-12
assert abs(m['cost_adjusted_gain']-1) < 1e-12
assert metrics(f, np.array([0.,1.,1.,1.]), c, 0.)['support_failure']
assert metrics(np.array([1e-13,1.]), np.array([0.,1.]), np.ones(2), 0.)['support_failure']
# Exact discrete importance variance by summing under q independently.
g = np.array([1.,5.,2.,7.]); q = g/g.sum(); val = f*g.mean()/g
v = np.dot(q, val*val)-np.dot(q,val)**2
assert abs(metrics(f,g,c,0.)['importance_variance']-v) < 1e-12
# Top1% proposal mass splits the first atom fractionally: .01/.1=.1.
assert abs(weighted_top_share(np.array([4.,2.]), np.array([.1,.9]), np.array([2.,3.]))-.04) < 1e-12
rng = np.random.default_rng(411)
model = (rng.normal(size=4), rng.normal(size=(4,4)), rng.normal(size=(5,4)))
for kind in ('23','12'):
    y,a,b,tau = walker.face_points(5,kind,rng.random((200,5)))
    out,_,_ = linear_proxies(model,y,a,b,tau,kind)
    assert np.all(out['pair'] <= out['envelope']+1e-12)
    for name,v in out.items():
        assert np.all(v >= 0) and np.all(v <= out['window']+1e-12)
print('Importance variance, changed-law tail, cost weighting, support, and proxy bounds: passed')
