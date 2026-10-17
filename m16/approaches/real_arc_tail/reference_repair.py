import json,time
from pathlib import Path
import mpmath as mp
from recurrence import even_arc,stable_arc,positive_arc,endpoint_arc,baseline

def main():
    t=time.process_time();mp.mp.dps=100;rows=[]
    original=json.loads(Path(__file__).with_name('controls_results.json').read_text())['arcs']
    extra=json.loads(Path(__file__).with_name('endpoint_results.json').read_text())['extra_cases']
    for row in original+extra:
        a,b,theta,n=row['a'],row['b'],row['theta'],row['degree']
        aa,bb,th=map(mp.mpf,(a,b,theta));amp=abs(aa)+abs(bb)*min(th,1)
        ref=th*amp**n*mp.quad(lambda u:((aa*mp.cos(th*u)+bb*mp.sin(th*u))/amp)**n,[0,1]) if amp else (th if n==0 else mp.mpf(0))
        out={k:row[k] for k in ('a','b','theta','degree')};out['old_reference']=row['reference'];out['reference']=str(ref);out['true_float_underflow']=float(ref)==0 and ref!=0
        for name,fn in (('raw',even_arc),('stable',stable_arc),('positive',positive_arc),('endpoint',endpoint_arc)):
            v=fn(a,b,theta,n)[-1];out[name]=v;out[name+'_relative']=float(abs(mp.mpf(v)-ref)/ref) if ref else 0.
        v=baseline.arc_power(a,b,theta,n);out['fourier']=v;out['fourier_relative']=float(abs(mp.mpf(v)-ref)/ref) if ref else 0.
        rows.append(out)
    out={'cases':rows,'cpu_seconds':time.process_time()-t,'reference':'100-digit positive normalized integral on [0,1] with amplitude and theta factored out'}
    Path(__file__).with_name('reference_repair_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'max_relative_representable':{name:max(r[name+'_relative'] for r in rows if not r['true_float_underflow']) for name in ('fourier','raw','stable','positive','endpoint')}},indent=2))
if __name__=='__main__':main()
