"""Exact-rational radial/determinant setup for sixteen frozen stored T6 cones."""
from collections import Counter
from fractions import Fraction as F
import hashlib,json,resource,time,traceback
from pathlib import Path
import numpy as np
import mpmath as mp
from directed import Directed
HERE=Path(__file__).parent;SOURCE=HERE.parent/'small_h2_56/cycle14_cold_rays.npz'

def setup(raw):
    counts=Counter();v=[[F.from_float(float(x)) for x in row] for row in raw]
    counts['float_to_rational']=36
    def dot(a,b):
        result=F(0)
        for x,y in zip(a,b):result+=x*y;counts['rational_multiply']+=1;counts['rational_add']+=1
        return result
    cols=list(map(list,zip(*v)));gram=[[dot(a,b) for b in cols] for a in cols]
    center=[sum(row,F(0)) for row in v];counts['rational_add']+=36
    projections=[dot(center,c) for c in cols];norm=dot(center,center)
    lower=min(map(min,gram));upper=max(gram[j][j] for j in range(6))
    if min(projections)>=0 and norm>0:
        support=min(projections)**2/norm;counts['rational_multiply']+=1;counts['rational_divide']+=1
        lower=max(lower,support)
    assert 0<lower<=upper
    matrix=[row[:] for row in v];det=F(1)
    for k in range(6):
        pivot=next((j for j in range(k,6) if matrix[j][k]),None)
        if pivot is None:det=F(0);break
        if pivot!=k:matrix[k],matrix[pivot]=matrix[pivot],matrix[k];det=-det;counts['rational_negate']+=1
        diag=matrix[k][k];det*=diag;counts['rational_multiply']+=1
        for i in range(k+1,6):
            ratio=matrix[i][k]/diag;counts['rational_divide']+=1
            for j in range(k+1,6):matrix[i][j]-=ratio*matrix[k][j];counts['rational_multiply']+=1;counts['rational_subtract']+=1
    maxbits=max(max(x.numerator.bit_length(),x.denominator.bit_length()) for row in matrix for x in row)
    return lower,upper,abs(det),dict(counts),maxbits

def q3(a,b,N):
    # Independent closed inverse-cube tail, rather than coefficient recurrence.
    s=mp.sqrt(a*b);alpha=(a+b)/(2*s);r=(b-a)/(a+b+2*s)
    return r**N/(s**3*(1-r))*(N*N+3*alpha*N+3*alpha**2-1+(2*N+3*alpha)*r/(1-r)+r*(1+r)/(1-r)**2)

def mf(x):return mp.mpf(x.numerator)/x.denominator if isinstance(x,F) else mp.mpf(str(x))

def run():
    tick=time.process_time();mp.mp.dps=160;fixture=HERE/'t6_fixture.json'
    if fixture.exists():saved=json.loads(fixture.read_text());rays=np.array(saved['rays'],float)
    else:
        rays=np.load(SOURCE)['rays'][:16].copy();saved=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),indices=list(range(16)),rays=rays.tolist());fixture.write_text(json.dumps(saved,indent=2)+'\n')
    out={'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((HERE/'directed.py').read_bytes()).hexdigest(),'cones':[]};sums={n:[] for n in (4,8,12,16)};refs={n:[] for n in sums}
    for index,raw in enumerate(rays):
        begin=time.process_time();lo,hi,det,events,maxbits=setup(raw);setupcpu=time.process_time()-begin
        mpdet=abs(mp.det(mp.matrix([[mf(F.from_float(float(x))) for x in row] for row in raw])))
        assert abs(mpdet-mf(det))<mp.mpf('1e-145')*max(mf(det),mp.mpf('1e-145'))
        ctx=Directed(50);a=ctx.rational(lo).lo;b=ctx.rational(hi).hi;D=ctx.rational(det)
        assert mf(a)<=mf(lo) and mf(b)>=mf(hi) and mf(D.lo)<=mf(det)<=mf(D.hi)
        row=dict(index=index,radial_lower=str(a),radial_upper=str(b),det_lo=str(D.lo),det_hi=str(D.hi),rational_setup_cpu=setupcpu,rational_events=events,max_elimination_integer_bits=maxbits,bounds=[])
        for n in sums:
            begin=time.process_time();value=ctx.simplex_bound(a,b,3,n-1,D);cpu=time.process_time()-begin
            truth=mf(det)*q3(mf(a),mf(b),n)/(60*mp.pi**3)
            assert mf(value.lo)<=truth<=mf(value.hi),(index,n,value,truth)
            row['bounds'].append(dict(order=n,lo=str(value.lo),hi=str(value.hi),reference=mp.nstr(truth,170),cpu=cpu));sums[n].append(value);refs[n].append(truth)
        row['directed_events']=dict(ctx.counts);out['cones'].append(row)
    ctx=Directed(50);out['sums']=[]
    for n,values in sums.items():
        value=ctx.sum(values);truth=mp.fsum(refs[n]);assert mf(value.lo)<=truth<=mf(value.hi)
        out['sums'].append(dict(order=n,lo=str(value.lo),hi=str(value.hi),reference=mp.nstr(truth,170)))
    out.update(cpu_seconds=time.process_time()-tick,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,directed_sum_events=dict(ctx.counts),total_rational_setup_cpu=sum(r['rational_setup_cpu'] for r in out['cones']),total_bound_cpu=sum(b['cpu'] for r in out['cones'] for b in r['bounds']))
    (HERE/'t6_results.json').write_text(json.dumps(out,indent=2)+'\n');print({k:out[k] for k in ('cpu_seconds','peak_rss_kb','total_rational_setup_cpu','total_bound_cpu')},flush=True)
if __name__=='__main__':
    try:run()
    except Exception as exc:
        (HERE/f't6_failure_{time.time_ns()}.json').write_text(json.dumps({'exception':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n');raise
