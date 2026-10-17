from pathlib import Path
import json,math,time
import mpmath as mp
from controls import polygons
from recurrence import endpoint_arc,endpoint,baseline

def main():
    t=time.process_time();mp.mp.dps=100
    rows=[];original=json.loads(Path(__file__).with_name('controls_results.json').read_text())['arcs']
    for old in original:
        row={k:old[k] for k in ('a','b','theta','degree','reference')};ref=mp.mpf(old['reference'])
        value=endpoint_arc(row['a'],row['b'],row['theta'],row['degree'])[-1]
        row.update(value=value,relative=float(abs(mp.mpf(value)-ref)/ref) if ref else 0.);rows.append(row)
    extra=[]
    for theta in (1e-200,math.nextafter(math.pi,0)):
        for a,b in ((1.,0.),(0.,1.),(1.,-1.),(1e-12,1.)):
            for n in (0,2,4,6):
                # Map to [0,1] and extract theta: avoids absolute tiny-interval
                # quadrature stopping before resolving its shape.
                ref=mp.mpf(theta)*mp.quad(lambda u:(mp.mpf(a)*mp.cos(mp.mpf(theta)*u)+mp.mpf(b)*mp.sin(mp.mpf(theta)*u))**n,[0,1])
                value=endpoint_arc(a,b,theta,n)[-1]
                extra.append(dict(a=a,b=b,theta=theta,degree=n,reference=str(ref),value=value,relative=float(abs(mp.mpf(value)-ref)/ref) if ref else 0.,reference_underflows_float=float(ref)==0 and ref!=0))
    ps=[]
    for i,vs in enumerate(polygons()):
        for k in (2,4,6,8):
            row={'case':i,'k':k,'raw_moment':endpoint.polygon_moment(vs,k-1)};stats={};tt=time.process_time()
            try:row['guarded']=endpoint.tail_mass(vs,k,stats)
            except ArithmeticError as e:
                row['guard_failure']=str(e);row['fallback_mass']=endpoint.tail_mass_with_fallback(vs,k,stats)
            row['stats']=stats;row['cpu_seconds']=time.process_time()-tt;ps.append(row)
    bench=[];vs=polygons()[4]
    for k in (2,4,6,8):
        r={'k':k,'repeats':300}
        for name,mod in (('fourier',baseline),('endpoint',endpoint)):
            tt=time.process_time();st={}
            for _ in range(300):mod.polygon_moment(vs,k-1,st)
            r[name+'_cpu']=time.process_time()-tt;r[name+'_stats']=st
        bench.append(r)
    out={'original_cases':rows,'extra_cases':extra,'polygons':ps,'benchmarks':bench,'cpu_seconds':time.process_time()-t,'roundoff_certified':False}
    Path(__file__).with_name('endpoint_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'max_relative_original':max(r['relative'] for r in rows),'max_relative_extra_representable':max(r['relative'] for r in extra if not r['reference_underflows_float']),'underflow_cases':sum(r['reference_underflows_float'] for r in extra),'polygon_fallbacks':sum('guard_failure'in r for r in ps),'benchmarks':[{k:v for k,v in r.items() if not k.endswith('_stats')} for r in bench]},indent=2))
if __name__=='__main__':main()
