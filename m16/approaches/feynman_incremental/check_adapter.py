"""Independent T30 exact-line fixtures, both target conventions and exclusions."""
import importlib.util,json,sys,time
from pathlib import Path
import numpy as np
from .adapter import walk_mass,WalkFailure
HERE=Path(__file__).resolve().parent
REVIEW=Path('/workspace/work/recover-small_h2_910/m16/approaches/walker_tolerance_review')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m

def main():
    start=time.process_time();reference=load('port_exact_reference',REVIEW/'rational_line.py')
    sys.modules['rational_line']=reference
    fixtures=load('port_independent_fixtures',REVIEW/'check.py').fixtures()
    out=dict(cases=[],cap_checks=[],thin_cases=[],random_model_inputs=0)
    def save():
        out['cpu_seconds']=time.process_time()-start
        (HERE/'adapter_results.json').write_text(json.dumps(out,indent=2)+'\n')
    for name,model,y,a,b in fixtures:
        direction=np.zeros(len(y));direction[a]=1/np.sqrt(2);direction[b]=-1/np.sqrt(2)
        for top in (False,True):
            neg,pos=(a,b) if top else (b,a)
            for tex in (0.,.0625,.25,.5,np.nextafter(.25,0.)):
                if time.process_time()-start>18:raise TimeoutError('Adapter primitive20CPU guard')
                truth=0.
                if tex<.25:
                    for lo,hi in ((-.25,-tex),(tex,.25)):
                        truth+=reference.wrong_mass(model,y,direction,lo,hi,neg,pos,deadline=start+18)['mass']
                row=dict(fixture=name,top=top,tex=float(tex),reference=truth)
                try:
                    mass,cost,ledger=walk_mass(model,y[None],np.array([a]),np.array([b]),
                        np.array([.25]),np.array([tex]),top=top,details=True)
                    error=float(mass[0])-truth
                    row.update(mass=float(mass[0]),absolute_error=abs(error),
                               relative_error=abs(error)/truth if truth else None,
                               fpe=float(cost.sum()),ledger=ledger)
                    assert abs(error)<1e-12
                    assert ledger['complete'] and float(cost.sum())>0
                    if tex>=.25:assert mass[0]==0 and ledger['shared_calls']==0
                    if tex==0:assert ledger['shared_calls']==1
                except WalkFailure as exc:
                    row.update(failure=str(exc),ledger=exc.ledger)
                (out['thin_cases'] if tex==np.nextafter(.25,0.) else out['cases']).append(row)
                save()
    name,model,y,a,b=fixtures[4]
    for cap in (0,1):
        try:walk_mass(model,y[None],np.array([a]),np.array([b]),np.array([.25]),np.zeros(1),max_iter=cap)
        except WalkFailure as exc:
            assert not exc.ledger['complete'] and exc.ledger['known_flops']>0
            out['cap_checks'].append(dict(cap=cap,explicit_failure=str(exc),ledger=exc.ledger))
        else:raise AssertionError('Cap unexpectedly returned completed mass')
    out.update(complete=True,maximum_absolute_error=max(r.get('absolute_error',0.) for r in out['cases']+out['thin_cases']),
               failure_count=sum('failure' in r for r in out['cases']+out['thin_cases']))
    save();print(json.dumps({k:out[k] for k in ('complete','maximum_absolute_error','failure_count','cpu_seconds')},indent=2))

if __name__=='__main__':main()
