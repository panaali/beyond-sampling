"""Same frozen controls for predeclared stable and positive alternatives."""
from pathlib import Path
import json,time
import mpmath as mp
from controls import polygons
from recurrence import stable_arc,positive_arc,baseline,variant,stable,positive

def main():
    t=time.process_time();mp.mp.dps=80
    old=json.loads(Path(__file__).with_name('controls_results.json').read_text());rows=[]
    for row in old['arcs']:
        ref=mp.mpf(row['reference']);out={k:row[k] for k in ('a','b','theta','degree','reference','real_relative','fourier_relative')}
        for name,fn in (('stable',stable_arc),('positive',positive_arc)):
            value=fn(row['a'],row['b'],row['theta'],row['degree'])[-1]
            out[name]=value;out[name+'_relative']=float(abs(mp.mpf(value)-ref)/ref) if ref else 0.
        rows.append(out)
    prows=[];fixtures=polygons()
    for i,vs in enumerate(fixtures):
        for k in (2,4,6,8):
            row={'case':i,'k':k}
            for name,mod in (('old',baseline),('raw',variant),('stable',stable),('positive',positive)):
                row[name]=mod.polygon_moment(vs,k-1)
                try:row[name+'_guarded']=mod.tail_mass(vs,k)
                except ArithmeticError as e:row[name+'_failure']=str(e)
            prows.append(row)
    benches=[]
    for k in (2,4,6,8):
        row={'k':k,'repeats':300}
        for name,mod in (('old',baseline),('raw',variant),('stable',stable),('positive',positive)):
            stats={};start=time.process_time()
            for _ in range(300):mod.polygon_moment(fixtures[4],k-1,stats)
            row[name+'_cpu']=time.process_time()-start;row[name+'_stats']=stats
        benches.append(row)
    out={'arcs':rows,'polygons':prows,'benchmarks':benches,'cpu_seconds':time.process_time()-t,'reference_source':'controls_results.json, 80-digit positive quadrature','certified_roundoff':False}
    Path(__file__).with_name('rescue_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'max_relative':{name:max(r[name+'_relative'] for r in rows) for name in ('stable','positive')},'negative':{name:sum(r[name]<0 for r in rows) for name in ('stable','positive')},'guard_failures':{name:sum(name+'_failure'in r for r in prows) for name in ('old','raw','stable','positive')},'benchmarks':[{k:v for k,v in r.items() if not k.endswith('_stats')} for r in benches]},indent=2))
if __name__=='__main__':main()
