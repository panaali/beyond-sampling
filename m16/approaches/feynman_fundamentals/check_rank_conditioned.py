"""Scalar checks of the rank-conditioned measure and exact sign-network toys."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


def main():
    start=time.process_time();rows=[];calls=0
    for T in range(3,11):
        def density(q):
            return T*(T-1)*q**(T-2)*(1-q)
        def low(q):
            if q<=.5:
                return (T-2)/(T-1)+(.5-q)/((T-1)*(1-q))
            return (T-2)/(T-1)*.5/q
        def integrate(fn):
            nonlocal calls
            result=0.
            for a,b in [(0.,.5),(.5,1.)]:
                val,err,info=quad(lambda q:density(q)*fn(q),a,b,
                                  epsabs=1e-13,epsrel=1e-13,full_output=True)
                calls+=info['neval'];result+=val
            return result
        normalization=integrate(lambda q:1.)
        zero=integrate(low)
        last=integrate(lambda q:float(q>.5))
        middle=integrate(lambda q:1-low(q))
        estimate_last=(zero+last)/T
        exact_last=1/T+(.5-(T+1)/2**T)/(T-1)
        rows.append({'T':T,'beta_normalization':normalization,
                     'constant_accuracy':normalization/T,'middle_class_accuracy':(zero+middle)/T,
                     'last_class_accuracy':estimate_last,'last_class_exact':exact_last,
                     'last_absolute_error':abs(estimate_last-exact_last)})
    result={'cases':rows,'max_last_error':max(x['last_absolute_error'] for x in rows),
            'max_constant_error':max(abs(x['constant_accuracy']-1/x['T']) for x in rows),
            'max_middle_error':max(abs(x['middle_class_accuracy']-1/x['T']) for x in rows),
            'T10_missing_box_mass':{str(L):float(-np.expm1(10*np.log1p(-2*ndtr(-L)))) for L in [4.,5.,6.]},
            'scalar_integrand_evaluations':calls,'model_forward_passes':0,'sampled_model_inputs':0,
            'cpu_seconds':time.process_time()-start}
    Path(__file__).with_name('rank_conditioned_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':
    main()
