"""Synthetic exactness tests for cross algebra, with no neural model calls."""
import numpy as np
from estimator import TensorCross
for shape in ([7,3,3,3],[5,5,5],[2,2,2,2]):
    factors=[np.linspace(.1,.9,n) for n in shape]
    def evaluate(points):
        x=np.asarray([[factors[k][i] for k,i in enumerate(p)] for p in points])
        return np.prod(1+x,axis=1)+.3*np.prod(2-x,axis=1)
    fit=TensorCross(evaluate,shape,rank=2,sweeps=2)
    pts=np.asarray(list(np.ndindex(*shape)))
    assert np.max(np.abs(fit.predict(pts)-evaluate(pts)))<1e-10
    assert abs(fit.integral([np.ones(n)/n for n in shape])-evaluate(pts).mean())<1e-10
print('Three rank-2 product-sum tensors: exact reconstruction and integration passed')
