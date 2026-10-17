"""Weighted pivot selection does not bound off-fiber amplification."""
import json, time
from pathlib import Path
import numpy as np


def main():
    start = time.process_time(); out = []
    for a in (1e-6, 1e-12):
        w = np.array([a**4, 1., a**4]); w /= w.sum()
        points = sorted({tuple(1 if j != k else i for j in range(3))
                         for k in range(3) for i in range(3)})
        residual = np.array([a if p == (1,1,1) else 1. for p in points])
        scores = np.array([np.prod(w[list(p)]) for p in points])*residual**2
        pivot = points[int(np.argmax(scores))]
        assert pivot == (1,1,1)
        fiber = np.array([1., a, 1.])/a
        prediction = lambda p: a*np.prod(fiber[list(p)])
        defect = max(abs(prediction(p)-r) for p,r in zip(points,residual))
        assert defect < 1e-14
        out.append(dict(amplitude=a, weights=w.tolist(), pivot=pivot,
                        fiber_defect=defect, off_fiber_value=prediction((0,0,0)),
                        expected_off_fiber_value=1/a**2, maximum_ratio=1/a,
                        represented_integral=float(a*np.dot(w,fiber)**3)))
    result = dict(cases=out, cpu_seconds=time.process_time()-start,
                  conclusion='Weighted score dominance alone does not bound extrapolation.')
    (Path(__file__).parent/'pivot_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
