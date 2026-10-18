"""Once-only analytic controls; no trained arrays or random inputs."""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import os
import resource
import sys
import time
import traceback
import numpy as np
from primitive import project

HERE = Path(__file__).resolve().parent
OUT = HERE/'controls'


def fixtures():
    # Expected labels: actual hidden, original lookup, projected lookup.
    return [
        ('improves', [[1,1]], [[-2,1]], [[-1,0],[0,1]], [1], [[1],[0],[1]]),
        ('worsens_strict', [[0,0,1]], [[1,-2,0]], [[0,0,0],[-1,-1,1]], [1], [[1],[1],[0]]),
        ('worsens_tie', [[1,1]], [[-2,-.5]], [[0,0],[-1,2]], [1], [[1],[1],[0]]),
        ('unchanged_positive', [[2,1]], [[1,2]], [[1,0],[0,1]], [0], [[0],[1],[1]]),
        ('zero', [[0,0]], [[0,0]], [[1,0],[0,1]], [0], [[0],[0],[0]]),
        ('signed_zero', [[0,1]], [[-0.,1]], [[1,0],[0,1]], [1], [[1],[1],[1]]),
        ('later_truth_tie', [[0,1]], [[-1,-1]], [[1,0],[0,1]], [1], [[1],[0],[0]]),
        ('first_truth_tie', [[1,0]], [[-1,-1]], [[1,0],[0,1]], [0], [[0],[0],[0]]),
        ('mixed_batch', [[1,1],[0,0]], [[-1,3],[2,-4]], [[1,0],[0,1]], [0,1], [[0,0],[1,0],[1,0]]),
        ('readout_cancellation', [[1,1,1]], [[-1,1,1]], [[2**53,1,-2**53],[0,0,0]], [0], [[0],[1],[1]]),
    ]


def guards():
    return [
        ('nan_lookup', [[0,0]], [[float('nan'),0]], [[1,0],[0,1]], ValueError),
        ('infinite_readout', [[0,0]], [[0,0]], [[float('inf'),0],[0,1]], ValueError),
        ('negative_actual', [[-1,0]], [[0,0]], [[1,0],[0,1]], ValueError),
        ('shape_mismatch', [[0,0]], [[0]], [[1,0],[0,1]], ValueError),
        ('empty_batch', np.empty((0,2)), np.empty((0,2)), [[1,0],[0,1]], ValueError),
        ('overflow_sse', [[0,0]], [[1e308,0]], [[0,0],[0,0]], FloatingPointError),
    ]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frac(x): return F.from_float(float(x))
def ratio(x): return [str(x.numerator), str(x.denominator)]


def main():
    assert not OUT.exists(), 'retain all earlier attempts'
    p = json.loads((HERE/'protocol.json').read_text())
    g = json.loads((HERE/'execution_gate.json').read_text())
    assert g['clearance'] is True and g['protocol_sha256'] == sha(HERE/'protocol.json')
    for path, digest in {**p['pins'], **g['pins']}.items(): assert sha(path) == digest, path
    assert 0 < resource.getrlimit(resource.RLIMIT_CPU)[0] <= 20
    assert 0 < resource.getrlimit(resource.RLIMIT_AS)[0] <= 2147483648
    OUT.mkdir(); start = time.process_time()
    result = dict(status='running', rows=[], guards=[], protocol_sha256=sha(HERE/'protocol.json'), gate=g,
                  runtime=dict(executable=sys.executable, prefix=sys.prefix, numpy_version=np.__version__, numpy_file=np.__file__))
    def save():
        tmp = OUT/'results.json.tmp'
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        with tmp.open('rb') as f: os.fsync(f.fileno())
        os.replace(tmp, OUT/'results.json')
    def event(**kw):
        with (OUT/'progress.jsonl').open('a') as f:
            f.write(json.dumps(kw, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    save(); event(phase='attempt', stage='runtime')
    try:
        assert sys.executable == p['runtime']['executable'] and sys.prefix == p['runtime']['prefix']
        assert np.__version__ == p['runtime']['numpy_version'] and np.__file__ == p['runtime']['numpy_file']
        event(phase='result', stage='runtime')
        fs, gs = fixtures(), guards(); assert len(fs)==10 and len(gs)==6
        for name, h0, z0, w0, truth, expected in fs:
            assert time.process_time()-start < 17
            event(phase='attempt', name=name)
            h,z,w = (np.asarray(x,dtype=np.float64) for x in (h0,z0,w0))
            out = project(h,z,w)
            actual, baseline = h@w.T, z@w.T
            labels = [a.argmax(axis=1).tolist() for a in (actual,baseline,out['projected_logits'])]
            exact = []
            for hr,zr,pr in zip(h,z,out['projected'], strict=True):
                old = sum((frac(a)-frac(b))**2 for a,b in zip(zr,hr,strict=True))
                new = sum((frac(a)-frac(b))**2 for a,b in zip(pr,hr,strict=True))
                coordinate_drop = [frac(a)**2-2*frac(a)*frac(b) if a<0 else F(0) for a,b in zip(zr,hr,strict=True)]
                assert all(x>=0 for x in coordinate_drop) and old-new == sum(coordinate_drop)
                exact.append(dict(before=ratio(old),after=ratio(new),drop=[ratio(x) for x in coordinate_drop]))
            row=dict(name=name, hidden=h.tolist(), lookup=z.tolist(), wo=w.tolist(), truth=truth,
                     expected_labels=expected, labels=labels, actual_logits=actual.tolist(), baseline_logits=baseline.tolist(),
                     exact_sse=exact, **{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in out.items()})
            row['passed']=labels==expected and all(F(*map(int,x['after']))<=F(*map(int,x['before'])) for x in exact)
            result['rows'].append(row); save(); event(phase='result',name=name,passed=row['passed'])
            assert row['passed'], name
        for name,h,z,w,exc in gs:
            event(phase='attempt',name=name)
            caught=None
            try: project(h,z,w)
            except Exception as e: caught=type(e).__name__
            row=dict(name=name,expected=exc.__name__,caught=caught,passed=caught==exc.__name__)
            result['guards'].append(row);save();event(phase='result',**row)
            assert row['passed'],name
        assert len(result['rows'])==10 and len(result['guards'])==6
        result['status']='complete';result['pass_all']=True;event(phase='complete')
    except BaseException as e:
        result.update(status='failure',failure=repr(e),traceback=traceback.format_exc());event(phase='failure',error=repr(e));raise
    finally:
        result['cpu_seconds']=time.process_time()-start;result['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;save()


if __name__ == '__main__': main()
