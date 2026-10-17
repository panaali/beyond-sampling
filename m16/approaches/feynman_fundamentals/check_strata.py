"""Independent integrations of the exactly derived four-stratum toy law."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


def main():
    start = time.process_time()
    phi = lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)
    calls = 0
    def integrate(fn):
        nonlocal calls
        val, err, info = quad(fn, 0, np.inf, epsabs=1e-13, epsrel=1e-13, full_output=True)
        calls += info['neval']
        return val, err
    origin, e0 = integrate(lambda x: phi(x)*ndtr(-2*x))
    origin += .25
    ray, e1 = integrate(lambda q: .5*phi(q))
    axis, e2 = integrate(lambda s: phi(s/np.sqrt(5))/np.sqrt(5)*ndtr(-3*s/np.sqrt(5)))
    # Integrate b=a+x, x>0; integral over a>0 is Phi(x).
    interior, e3 = integrate(lambda x: phi(x)*ndtr(x))
    exact = [.25+np.arctan(.5)/(2*np.pi), .25,
             (np.arctan(2)-np.pi/4)/(2*np.pi), .375]
    actual = [origin,ray,axis,interior]
    result = {'component_order':['origin','input_ray','axis','interior'],
              'exact_angle_masses':exact, 'independent_integrals':actual,
              'quadrature_error_estimates':[e0,e1,e2,e3],
              'max_absolute_difference':float(np.max(np.abs(np.array(actual)-exact))),
              'total_mass':sum(actual),'scalar_integrand_evaluations':calls,
              'model_forward_passes':0,'sampled_model_inputs':0,
              'cpu_seconds':time.process_time()-start}
    Path(__file__).with_name('strata_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
