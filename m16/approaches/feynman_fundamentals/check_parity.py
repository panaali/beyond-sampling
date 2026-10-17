"""Parity evaluator versus unchanged full moments, with fixed CPU comparison."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from moments import polygon_moments,polygon_moment,unit


def main():
    started=time.process_time()
    cases=[np.diag([1.,sy,sz]) for sy in (-1,1) for sz in (-1,1)]
    cases.append(np.array([unit(v) for v in ((1,.2,.1),(1,-.3,.8),(.1,.8,.6))]))
    for eps in (1e-3,1e-7):
        cases.append(np.array([unit(v) for v in ((eps,1,0),(eps,np.cos(.4),np.sin(.4)),
                                                  (2*eps,np.cos(.2),np.sin(.2)))]))
    for eps in (1e-3,1e-8):
        cases.append(np.array([unit(v) for v in ((1,0,0),(1,eps,0),(1,0,eps))]))
    rows=[];stats={}
    for i,vs in enumerate(cases):
        full=polygon_moments(vs,9)
        new=np.array([polygon_moment(vs,p,stats) for p in range(10)])
        rows.append({'case':i,'full':full.tolist(),'parity':new.tolist(),
                     'bit_identical':bool(np.array_equal(full,new)),
                     'max_absolute_difference':float(np.max(np.abs(full-new)))})
    repeats=300
    t=time.process_time()
    for _ in range(repeats):
        polygon_moments(cases[4],7)[-1]
    full_cpu=time.process_time()-t
    bench_stats={};t=time.process_time()
    for _ in range(repeats):
        polygon_moment(cases[4],7,bench_stats)
    parity_cpu=time.process_time()-t
    result={'cases':rows,'all_bit_identical':all(r['bit_identical'] for r in rows),
            'validation_stats':stats,'benchmark_repeats_per_method':repeats,
            'full_cpu_seconds':full_cpu,'parity_cpu_seconds':parity_cpu,
            'parity_benchmark_stats':bench_stats,
            'full_benchmark_fourier_terms':repeats*3*28,
            'full_benchmark_edge_integrals':repeats*3*7,
            'full_benchmark_area_calls':repeats,
            'source_sha256':hashlib.sha256(Path(__file__).with_name('moments.py').read_bytes()).hexdigest(),
            'model_forward_passes':0,'sampled_model_inputs':0,'cpu_seconds':time.process_time()-started}
    Path(__file__).with_name('parity_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':
    main()
