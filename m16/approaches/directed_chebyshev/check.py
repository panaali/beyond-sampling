"""Independent high-precision derivative checks of directed positive tails."""
import hashlib,importlib.util,json,math,resource,time,traceback,warnings
from pathlib import Path
import mpmath as mp
from directed import Directed,Interval,polynomial
HERE=Path(__file__).parent
OLD=HERE.parent/'small_h2_56/exact_tail.py'
spec=importlib.util.spec_from_file_location('old_tail',OLD);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def scalar(x):
    if isinstance(x,float):n,d=x.as_integer_ratio();return mp.mpf(n)/d
    return mp.mpf(str(x))

def reference(a,b,nu,degree):
    a,b=scalar(a),scalar(b);m=(a+b)/2;delta=(b-a)/2;N=degree+1
    if delta==0:return mp.mpf(0)
    def first(center):
        s=mp.sqrt(center**2-delta**2);r=delta/(center+s)
        return 2*r**N/(s*(1-r))
    return (-1)**(nu-1)*mp.diff(first,m,nu-1)/mp.factorial(nu-1)

def run():
    start=time.process_time();mp.mp.dps=160;out={'cases':[],'old_float_comparisons':[],'mass_controls':[],'precision':[50,80]}
    intervals=[(1/6,1.),(.9,1.),(math.nextafter(1.,0.),1.),(1e-100,2e-100)]
    for precision in (50,80):
        for a,b in intervals:
            for nu in (1,2,3,4,5):
                for degree in (0,3,15,63,255,2047):
                    ctx=Directed(precision);value=ctx.tail(a,b,nu,degree);truth=reference(a,b,nu,degree)
                    lo,hi=scalar(value.lo),scalar(value.hi)
                    row=dict(a=a,b=b,nu=nu,degree=degree,precision=precision,lo=str(value.lo),hi=str(value.hi),reference=mp.nstr(truth,170),contains=bool(lo<=truth<=hi),relative_width=float((hi-lo)/truth),counts=dict(ctx.counts))
                    out['cases'].append(row)
                    assert row['contains'],row
                    # Outward float conversion should include even underflowed tiny tails.
                    try:
                        low,high=value.floats();assert scalar(low)<=truth<=scalar(high),(row,'float conversion');row['float_conversion']='enclosed'
                    except OverflowError:
                        assert truth>scalar(float.fromhex('0x1.fffffffffffffp+1023'));row['float_conversion']='correctly rejected overflow'
                    if precision==50 and nu==3:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always');prev=float(old.tail(a,b,degree))
                        out['old_float_comparisons'].append(dict(a=a,b=b,degree=degree,old=prev,undershoots=bool(scalar(prev)<truth),relative_error=float((scalar(prev)-truth)/truth),warnings=[str(w.message) for w in caught]))
        ctx=Directed(precision);pi=ctx.pi();assert scalar(pi.lo)<mp.pi<scalar(pi.hi)
        for nu in (1,2,3,4,5):
            value=ctx.simplex_bound(1/6,1.,nu,15,0.125)
            truth=reference(1/6,1.,nu,15)*mp.mpf(1)/8*mp.factorial(nu-1)/(mp.pi**nu*mp.factorial(2*nu-1))
            assert scalar(value.lo)<=truth<=scalar(value.hi)
            out['mass_controls'].append(dict(precision=precision,nu=nu,lo=str(value.lo),hi=str(value.hi),reference=mp.nstr(truth,170)))
        zero=ctx.tail(1.,1.,3,2047);assert zero.lo==zero.hi==0
        total=ctx.sum([pi,pi,zero]);assert scalar(total.lo)<2*mp.pi<scalar(total.hi)
    out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,max_relative_width=max(r['relative_width'] for r in out['cases']),source_sha256=hashlib.sha256((HERE/'directed.py').read_bytes()).hexdigest(),old_source_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest())
    (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    print({k:out[k] for k in ('cpu_seconds','peak_rss_kb','max_relative_width')},'cases',len(out['cases']),'old undershoots',sum(r['undershoots'] for r in out['old_float_comparisons']),flush=True)
if __name__=='__main__':
    try:run()
    except Exception as exc:
        (HERE/f'failure_{time.time_ns()}.json').write_text(json.dumps({'exception':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n');raise
