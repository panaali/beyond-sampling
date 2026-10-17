"""Frozen binary64 enclosure controls and overhead-only repeated workload."""
import json,time,resource,hashlib,traceback
from pathlib import Path
import numpy as np
import mpmath as mp
from decimal import Decimal
from directed import Interval
from binary import Binary
HERE=Path(__file__).parent

def mf(x):
    n,d=float(x).as_integer_ratio();return mp.mpf(n)/d

def run():
    start=time.process_time();mp.mp.dps=180;out={'primitive':[],'t6':[],'timings':[]}
    fixed=json.loads((HERE/'results.json').read_text())
    for row in fixed['cases']:
        if row['precision']!=50 or row['nu']!=3:continue
        ctx=Binary();lo,hi=ctx.tail(row['a'],row['b'],row['degree']);truth=mp.mpf(row['reference']);assert mf(lo)<=truth<=mf(hi)
        out['primitive'].append(dict(a=row['a'],b=row['b'],degree=row['degree'],lo=float(lo),hi=float(hi),relative_width=mp.nstr((mf(hi)-mf(lo))/truth,24),decimal_interval_contained=bool(mf(lo)<=mp.mpf(row['lo']) and mf(hi)>=mp.mpf(row['hi'])),counts=dict(ctx.counts)))
    saved=json.loads((HERE/'t6_results.json').read_text());a=[];b=[];dl=[];dh=[]
    for c in saved['cones']:
        x=Interval(Decimal(c['radial_lower']),Decimal(c['radial_lower'])).floats();a.append(x[0])
        x=Interval(Decimal(c['radial_upper']),Decimal(c['radial_upper'])).floats();b.append(x[1])
        x=Interval(Decimal(c['det_lo']),Decimal(c['det_hi'])).floats();dl.append(x[0]);dh.append(x[1])
    # Binary radial widening changes the analytic tail slightly; compare with a fresh
    # independent q^-3 expression on those endpoints, retaining exact determinant.
    from t6_subset import q3,setup,mf as fraction_mp
    fixture=json.loads((HERE/'t6_fixture.json').read_text())
    determinants=[setup(raw)[2] for raw in fixture['rays']]
    for order in (4,8,12,16):
        ctx=Binary();lo,hi=ctx.bound(a,b,order-1,dl,dh)
        for i,c in enumerate(saved['cones']):
            truth=fraction_mp(determinants[i])*q3(mf(a[i]),mf(b[i]),order)/(60*mp.pi**3)
            assert mf(lo[i])<=truth<=mf(hi[i])
            out['t6'].append(dict(index=i,order=order,lo=float(lo[i]),hi=float(hi[i]),relative_width=mp.nstr((mf(hi[i])-mf(lo[i]))/truth,24)))
    for repeats in (1,64):
        arrays=[np.tile(x,repeats) for x in (a,b,dl,dh)];tick=time.process_time();wall=time.perf_counter();events={}
        for order in (4,8,12,16):
            ctx=Binary();ctx.bound(arrays[0],arrays[1],order-1,arrays[2],arrays[3]);events[str(order)]=dict(ctx.counts)
        out['timings'].append(dict(repeats=repeats,distinct_cones=16,bound_evaluations=64*repeats,cpu=time.process_time()-tick,wall=time.perf_counter()-wall,events=events))
    out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=hashlib.sha256((HERE/'binary.py').read_bytes()).hexdigest())
    (HERE/'binary_results_final.json').write_text(json.dumps(out,indent=2)+'\n');print(out['cpu_seconds'],out['timings'],flush=True)
if __name__=='__main__':
    try:run()
    except Exception as exc:
        (HERE/f'binary_failure_{time.time_ns()}.json').write_text(json.dumps({'exception':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n');raise
