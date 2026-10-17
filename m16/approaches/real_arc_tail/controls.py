"""Frozen same-parity primitive comparison with independent positive integrals."""
import hashlib,json,math,time
from pathlib import Path
import mpmath as mp
import numpy as np
from recurrence import even_arc,polygon_moment,baseline,variant,SOURCE

def polygons():
    cases=[np.diag([1.,sy,sz]) for sy in (-1,1) for sz in (-1,1)]
    cases.append(np.array([baseline.unit(v) for v in ((1,.2,.1),(1,-.3,.8),(.1,.8,.6))]))
    for eps in (1e-3,1e-7):
        cases.append(np.array([baseline.unit(v) for v in ((eps,1,0),(eps,np.cos(.4),np.sin(.4)),(2*eps,np.cos(.2),np.sin(.2)))]))
    for eps in (1e-3,1e-8):
        cases.append(np.array([baseline.unit(v) for v in ((1,0,0),(1,eps,0),(1,0,eps))]))
    return cases

def main():
    start=time.process_time();mp.mp.dps=80;rows=[]
    for a,b in ((1.,0.),(0.,1.),(1.,-1.),(1.,1.),(1e-12,1.)):
        for theta in (1e-12,1e-6,.3,math.pi/2,math.pi-1e-8):
            refs={n:mp.quad(lambda t:(mp.mpf(a)*mp.cos(t)+mp.mpf(b)*mp.sin(t))**n,[0,mp.mpf(theta)]) for n in (0,2,4,6)}
            for scale in (1e-6,1.,1e6):
                aa,bb=a*scale,b*scale
                vals=even_arc(aa,bb,theta,6)
                for i,n in enumerate((0,2,4,6)):
                    # Reference uses the exact stored scaled inputs; account for
                    # the rounded scale product rather than assuming covariance.
                    ref=mp.quad(lambda t:(mp.mpf(aa)*mp.cos(t)+mp.mpf(bb)*mp.sin(t))**n,[0,mp.mpf(theta)])
                    old=baseline.arc_power(aa,bb,theta,n);new=vals[i]
                    rows.append(dict(a=aa,b=bb,theta=theta,degree=n,reference=str(ref),fourier=old,real=new,real_relative=float(abs(mp.mpf(new)-ref)/ref) if ref else 0.,fourier_relative=float(abs(mp.mpf(old)-ref)/ref) if ref else 0.,real_negative=new<0,fourier_negative=old<0))
    prows=[]
    for i,vs in enumerate(polygons()):
        for k in (2,4,6,8):
            row=dict(case=i,k=k,old=baseline.polygon_moment(vs,k-1),real=polygon_moment(vs,k-1))
            for name,module in (('old',baseline),('real',variant)):
                try:row[name+'_guarded']=module.tail_mass(vs,k)
                except ArithmeticError as e:row[name+'_failure']=str(e)
            prows.append(row)
    benches=[]
    for k in (2,4,6,8):
        row={'k':k,'repeats':300}
        for name,fn in (('old',baseline.polygon_moment),('real',polygon_moment)):
            stats={};t=time.process_time()
            for _ in range(300):fn(polygons()[4],k-1,stats)
            row[name+'_cpu']=time.process_time()-t;row[name+'_stats']=stats
        benches.append(row)
    result={'arcs':rows,'polygons':prows,'benchmarks':benches,'cpu_seconds':time.process_time()-start,'source_sha256':hashlib.sha256((SOURCE/'moments.py').read_bytes()).hexdigest(),'sampled_model_inputs':0,'model_forwards':0}
    Path(__file__).with_name('controls_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':result['cpu_seconds'],'max_relative_real':max(r['real_relative'] for r in rows),'max_relative_fourier':max(r['fourier_relative'] for r in rows),'negative_real':sum(r['real_negative'] for r in rows),'negative_fourier':sum(r['fourier_negative'] for r in rows),'guard_failures_real':sum('real_failure'in r for r in prows),'guard_failures_old':sum('old_failure'in r for r in prows),'benchmarks':benches},indent=2))
if __name__=='__main__':main()
