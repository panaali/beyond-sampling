"""Frozen exact fixtures for legacy tie, gap, cap and close-root behavior."""
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.special import ndtr
from rational_line import wrong_mass

HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/clean-neurons/m16/approaches')


def load(label,path):
    spec=importlib.util.spec_from_file_location('review_'+label,ROOT/path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixtures():
    y=np.full(10,-1.); y[0:2]=0.; y[9]=1.
    zero=(np.ones(1),np.zeros((1,1)),np.zeros((10,1)))
    rows=[('all_zero',zero,y.copy(),0,1)]
    for scale in (1.,1e-14,1e-16):
        Wo=np.zeros((10,1)); Wo[1]=scale
        rows.append((f'positive_gap_{scale:g}',(np.ones(1),np.zeros((1,1)),Wo),y.copy(),0,1))
    y=np.full(10,-1.); y[0]=1.; y[8:]=0.
    Wo=np.zeros((10,1)); Wo[9]=1.
    rows.append(('gate_at_zero',(np.ones(1),np.zeros((1,1)),Wo),y,8,9))
    for gap in (1e-5,1e-9):
        y=np.full(10,.2); y[0]=3.; y[8:]=1.
        total=y.sum(); r1=-.125; r2=r1+gap
        Wo=np.zeros((10,2)); Wo[:,0]=-100.
        Wo[0]=0.
        for label,slope,intercept in [(9,1.,-r1),(8,2.,-(r1+r2))]:
            Wo[label,1]=-slope*np.sqrt(2)
            Wo[label,0]=(intercept-Wo[label,1])/total
        for scale in (1.,1e-16):
            rows.append((f'close_roots_{gap:g}_scale_{scale:g}',
                        (np.ones(2),np.diag([1.,0.]),Wo*scale),y.copy(),8,9))
    return rows


def main():
    start=time.process_time(); deadline=start+105
    paths=dict(tao='sci_tao/core.py',feynman='sci_feynman/core.py',wtool='sci_feynman/wtool.py',
               zones='boundary/zones.py',tube='tube/tube.py')
    output=dict(complete=False,random_model_inputs=0,rows=[],caps=[],tube_controls=[],
                source_sha256={label:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for label,path in paths.items()})
    modules={label:load(label,path) for label,path in paths.items()}
    output['module_setup_cpu']=time.process_time()-start
    def evaluate(label,model,y,a,b,**kw):
        x=y[None]; aa=np.array([a]); bb=np.array([b]); tau=np.array([.25]); zero=np.zeros(1)
        direction=np.zeros((1,10)); direction[0,a]=1/np.sqrt(2); direction[0,b]=-1/np.sqrt(2)
        if label=='tao':return float(modules[label].walk_mass(model,x,aa,bb,tau,'23',**kw)[0][0])
        if label=='feynman':return float(modules[label].walk_mass(model,x,aa,bb,tau,zero,**kw)[0][0])
        if label=='wtool':return float(modules[label].walk(model,x,aa,bb,tau,**kw)['mass'][0])
        if label=='zones':return float(modules[label].walk_wrong(model,x,direction,tau,bb,aa,zero,tau,**kw)[0][0])
        raise ValueError(label)
    try:
        all_fixtures=fixtures()
        for name,model,y,a,b in all_fixtures:
            d=np.zeros(10); d[a]=1/np.sqrt(2); d[b]=-1/np.sqrt(2)
            exact=wrong_mass(model,y,d,-.25,.25,b,a,deadline=deadline)
            values={}
            for label in ('tao','feynman','wtool','zones'):
                clock=time.process_time()
                try:
                    value=evaluate(label,model,y,a,b)
                    values[label]=dict(mass=value,error=value-exact['mass'],cpu_seconds=time.process_time()-clock)
                except (IndexError,ValueError,RuntimeError) as exc:values[label]=dict(failure=repr(exc))
            output['rows'].append(dict(name=name,reference=exact,legacy=values))
        for name,model,y,a,b in [all_fixtures[4]]:
            for cap in (0,1):
                for label,keyword in [('tao','max_cells'),('feynman','max_iter'),('wtool','max_iter'),('zones','max_iter')]:
                    value=evaluate(label,model,y,a,b,**{keyword:cap})
                    output['caps'].append(dict(fixture=name,walker=label,cap=cap,returned_mass=value,
                                               completion_or_remainder_returned=False))
        name,model,y,a,b=all_fixtures[5]
        try:
            output['wtool_maxseg1']=dict(returned_mass=evaluate('wtool',model,y,a,b,maxseg=1))
        except (IndexError,ValueError,RuntimeError) as exc:output['wtool_maxseg1']=dict(failure=repr(exc))
        tube=modules['tube']; W=lambda a,b:ndtr(b)-ndtr(a)
        name,model,y,a,b=all_fixtures[0]
        d=np.zeros((1,10)); d[0,a]=1/np.sqrt(2); d[0,b]=-1/np.sqrt(2)
        for K in (0,1):
            err,rem,cost=tube.walk_side(model,y[None],d,np.array([0.]),np.array([.25]),np.array([1]),K,Wf=W)
            output['tube_controls'].append(dict(fixture='all_zero_fixed_target1',K=K,error=float(err[0]),
                remainder=float(rem[0]),truth=float(ndtr(.25)-.5),cost=float(cost[0]),declared_remainder=True))
        name,model,y,a,b=all_fixtures[4]
        d=np.zeros((1,10)); d[0,a]=1/np.sqrt(2); d[0,b]=-1/np.sqrt(2)
        for K in (0,1):
            # Fixed target8 is wrong on the whole symmetric segment for this model.
            err,rem,cost=tube.walk_side(model,y[None],d,np.array([-.25]),np.array([.25]),np.array([8]),K,Wf=W)
            output['tube_controls'].append(dict(fixture='gate_fixed_target8',K=K,error=float(err[0]),
                remainder=float(rem[0]),truth=float(ndtr(.25)-ndtr(-.25)),cost=float(cost[0]),declared_remainder=True))
        # The independent exact reference refuses a cap instead of returning a partial scalar.
        try:wrong_mass(model,y,d[0],-.25,.25,b,a,max_pieces=0,deadline=deadline)
        except RuntimeError as exc:output['reference_cap']=dict(explicit_failure=repr(exc))
        else:raise AssertionError('reference cap silently returned')
        output['complete']=True
    except (AssertionError,ValueError,TimeoutError,RuntimeError) as exc:
        output['failure']=repr(exc)
    output['cpu_seconds']=time.process_time()-start
    output['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (HERE/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('rows',)},indent=2))

if __name__=='__main__':main()
