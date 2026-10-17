"""Closed-form near-coincident and simultaneous-root regression, not legacy parity."""
import json
from pathlib import Path
import warnings
import numpy as np
from scipy.special import ndtr
import core as C
import compiled as K


def main():
    model=(np.ones(1),np.array([[4.]]),np.array([[0.],[1.],[0.]]))
    records=[]
    for gap in (.01,.1,.3):
        for q in (-1e-6,-1e-10,-1e-12,-1e-16,0.,1e-16,1e-12,1e-10,1e-6):
            for tau in (.1, np.sqrt(2)*(gap+q)/3):
                left=(4*q-gap)*np.sqrt(2)/4
                right=(gap-20*q)*np.sqrt(2)/12
                exact=(ndtr(min(left,0))-ndtr(-tau) if left>-tau else 0.) + ndtr(min(tau,right))-.5
                for eps in (0.,1e-10):
                    args=(model,np.array([[q,q,-gap]]),np.array([0]),np.array([1]),np.array([tau]),'12')
                    py=C.walk_mass(*args,eps=eps,details=True)
                    cpp=K.walk_mass(*args,eps=eps,details=True)
                    # Probing may skip tiny cells of O(eps) mass, not macroscopic
                    # tails. eps=0 has no intentional interval skipping.
                    tol=1e-13 if eps==0 else 2e-9
                    assert abs(py[0][0]-exact)<tol,(gap,q,tau,eps,exact,py[0])
                    assert abs(cpp[0][0]-exact)<tol,(gap,q,tau,eps,exact,cpp[0])
                    assert np.array_equal(py[1],cpp[1]) and np.array_equal(py[2],cpp[2])
                    assert all(np.array_equal(py[3][k],cpp[3][k]) for k in py[3])
                    records.append({'gap':gap,'q':q,'tau':tau,'eps':eps,'exact':float(exact),
                                    'python':float(py[0][0]),'cpp':float(cpp[0][0]),'cells':int(py[2][0])})
    overflowing=(np.array([1e200]),np.array([[1e200]]),np.array([[0.],[1.],[0.]]))
    for backend in (C,K):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                backend.walk_mass(overflowing,np.array([[0.,0.,-.1]]),np.array([0]),np.array([1]),np.array([.1]),'12')
            raise AssertionError('Expected overflow rejection')
        except FloatingPointError:
            pass
    result={'cases':len(records),'max_error':max(abs(r['python']-r['exact']) for r in records),
            'max_error_eps0':max(abs(r['python']-r['exact']) for r in records if r['eps']==0),
            'overflow_guard':'both passed','records':records}
    Path(__file__).with_name('root_regression.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='records'})


if __name__=='__main__':main()
