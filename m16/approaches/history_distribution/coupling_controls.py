"""Independent target-label partition and constant-prefix checks."""
from collections import defaultdict
import json,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
from distribution import atoms
from engine import calculate
from coupling import rank_mass,normal_mass,fixed_prefix_probabilities

HERE=Path(__file__).parent
def run():
    started=time.process_time();stats=defaultdict(float);out=dict(intervals=[],constant_prefixes=[])
    for prefix in ((-1.3,.7),(.7,-1.3),(.4,1.2),(-2.,-.4)):
        for y in (-4.,-1.,0.,.3,2.):
            lo=-2.5+.1*y;hi=1.4+.05*y
            measured=np.array([rank_mass(np.array([y]),np.array([lo]),np.array([hi]),prefix,j,stats)[0] for j in range(4)])
            edges=sorted(set([lo,hi]+[v for v in (*prefix,y) if lo<v<hi]));expected=np.zeros(4)
            for a,b in zip(edges[:-1],edges[1:]):
                x=(a+b)/2;values=(*prefix,y,x);target=sorted(range(4),key=lambda k:(values[k],k))[-2]
                expected[target]+=ndtr(b)-ndtr(a)
            error=float(np.max(abs(measured-expected)))
            assert error<1e-14
            assert abs(measured.sum()-(ndtr(hi)-ndtr(lo)))<1e-14
            out['intervals'].append(dict(prefix=prefix,y=y,error=error,measured=measured.tolist()))
        analytic=fixed_prefix_probabilities(prefix);numeric=[]
        for label in range(4):
            def f(y):return float(np.exp(-y*y/2)/np.sqrt(2*np.pi)*rank_mass(np.array([y]),np.array([-np.inf]),np.array([np.inf]),prefix,label,stats)[0])
            points=[-10.,*sorted(prefix),10.]
            numeric.append(sum(quad(f,a,b,epsabs=1e-12)[0] for a,b in zip(points[:-1],points[1:])))
        assert np.max(abs(np.array(numeric)-analytic))<1e-12
        out['constant_prefixes'].append(dict(prefix=prefix,analytic=analytic.tolist(),numeric=numeric))
    wi=np.array([.7,-1.1]);wh=np.array([[-.4,.3],[.8,-.2]]);wo=np.zeros((4,2));model=wi,wh,wo
    d,r=atoms(model,4,4,rank_aligned=True);target=np.zeros(4);maximum=0.
    for sign in (1,-1):
        states,weights=d[sign];inputs=np.array(r['conditional'][sign]['prefix_inputs'])
        direct=np.zeros_like(states)
        for index in range(2):direct=np.maximum(direct@wh.T+inputs[:,index,None]*wi,0)
        maximum=max(maximum,float(np.max(abs(direct-states))))
        assert np.all(inputs[:,0]!=inputs[:,1])
        target+=sum((w*fixed_prefix_probabilities(v)/2 for w,v in zip(weights,inputs)),np.zeros(4))
    assert maximum<1e-12
    value,report=calculate(model,4,4,16,rank_aligned=True,coupled=True)
    assert abs(value-target[0])<report['omitted_y_mass_bound']+1e-11
    out.update(atom_direct_forward_error=maximum,finite_atom_target_mass=target.tolist(),finite_atom_chance_defect=(target-.25).tolist(),constant_network=value,constant_prefix_expected=float(target[0]),constant_network_stats=report,rank_work=dict(stats),cpu_seconds=time.process_time()-started)
    (HERE/'coupling_controls_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print('rank/forward controls pass','target masses',target,'CPU',out['cpu_seconds'])

if __name__=='__main__':
    try:run()
    except Exception as exc:
        import traceback
        (HERE/'coupling_controls_failure.json').write_text(json.dumps(dict(exception=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n');raise
