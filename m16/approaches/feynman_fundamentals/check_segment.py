"""Fixed independent high-precision quadrature check of the segment formula."""
import json
import time
from pathlib import Path
import mpmath as mp
import numpy as np
from segment import inverse_six_segment


def main():
    start = time.process_time()
    mp.mp.dps = 80
    cases = []
    for r, s in [(1., 1.), (.01, 100.), (100., .01), (.3, 2.)]:
        for c in [0., .5, np.sqrt(.75)-1e-8, np.sqrt(.75)+1e-8, .99, 1-1e-12, 1.]:
            cases.append((np.array([r, 0.]), np.array([s*c, s*np.sqrt((1-c)*(1+c))])))
    a = np.array([p[0] for p in cases])
    b = np.array([p[1] for p in cases])
    actual, stats = inverse_six_segment(a, b, diagnostics=True)
    rows = []
    for (aa, bb), val in zip(cases, actual):
        # Reference uses original supplied endpoint components, not their cosine.
        ma = list(map(mp.mpf, aa)); mb = list(map(mp.mpf, bb))
        def fun(u):
            return sum(((1-u)*x+u*y)**2 for x,y in zip(ma,mb))**-3
        # Endpoint refinement handles the deliberately unequal-norm pairs.
        ref = mp.quad(fun, [0, mp.mpf('.00001'), mp.mpf('.001'), mp.mpf('.1'),
                            mp.mpf('.5'), mp.mpf('.9'), mp.mpf('.999'), mp.mpf('.99999'), 1])
        rel = abs(mp.mpf(float(val))/ref-1)
        rows.append({'a':aa.tolist(), 'b':bb.tolist(), 'actual':float(val),
                     'reference':str(ref), 'relative_error':float(rel)})
    scaled = inverse_six_segment(a*7, b*7)*7**6
    swapped = inverse_six_segment(b,a)
    series_bound = stats.pop('series_truncation_bound')
    result = {'cases': rows, 'stats':stats,
              'max_relative_error':max(x['relative_error'] for x in rows),
              'max_scaling_relative_error':float(np.max(np.abs(scaled/actual-1))),
              'max_swapping_relative_error':float(np.max(np.abs(swapped/actual-1))),
              'series_bounds':[None if np.isnan(x) else float(x) for x in series_bound],
              'author_cpu_seconds':time.process_time()-start,
              'independent_reference_method':'80-digit direct scalar quadrature, fixed subintervals',
              'total_endpoint_pairs_including_scale_and_swap':3*len(cases),
              'classification':'deductive analytic segment formula; reference quadrature only'}
    Path(__file__).with_name('segment_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','series_bounds')},indent=2))


if __name__ == '__main__':
    main()
