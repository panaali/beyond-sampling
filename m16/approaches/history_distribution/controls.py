"""Moment/support/scaling and inherited-rank integration controls."""
import json,time
from pathlib import Path
import numpy as np
from distribution import atoms
from engine import calculate,prior

HERE=Path(__file__).parent
def run():
    start=time.process_time();out=dict(distributions=[],models=[])
    fixtures=[('ordinary',np.array([.7,-1.1]),np.array([[-.4,.3],[.8,-.2]])),
              ('zero_weight',np.array([0.,1.]),np.array([[-.4,.3],[.8,-.2]])),
              ('zero_recurrent',np.array([.7,-1.1]),np.zeros((2,2))),
              ('ordered_support',np.ones(2),np.diag([1.,2.]))]
    wo=np.array([[1.,0.],[0.,1.],[-1.,-.2],[.1,-.1]])
    for name,wi,wh in fixtures:
        for angular in (2,4):
            for radial in (2,4):
                model=wi,wh,wo;d,r=atoms(model,angular,radial)
                scale=np.array([.1,7.]);transformed=(wi*scale,wh*scale[:,None]/scale[None,:],wo/scale)
                ds,rs=atoms(transformed,angular,radial)
                delta=max(np.max(abs(ds[s][0]-d[s][0]*scale)) for s in (1,-1))
                assert delta<1e-11
                if name=='ordered_support':
                    for sign in (1,-1):assert np.all(d[sign][0][:,1]>=d[sign][0][:,0]-1e-14)
                if name=='zero_weight':
                    assert all(r['conditional'][s]['mean_error']<1e-12 for s in (1,-1))
                out['distributions'].append(dict(name=name,angular=angular,radial=radial,scale_covariance_error=float(delta),report=r))
    model=fixtures[0][1],fixtures[0][2],wo
    value,stats=calculate(model,2,2,16,collapse=True);old,oldstats=prior.calculate(model,order=16)
    assert abs(value-old)<1e-12
    out['models'].append(dict(kind='collapsed_mean',value=value,old=old,difference=value-old,stats=stats))
    for name,model in [('zero_readout',(model[0],model[1],np.zeros_like(wo))),('Wh0',(model[0],np.zeros((2,2)),wo))]:
        value,stats=calculate(model,2,2,16);old,oldstats=prior.calculate(model,order=16)
        assert abs(value-old)<1e-12
        if name=='zero_readout':assert abs(value-.25)<stats['omitted_y_mass_bound']+1e-10
        out['models'].append(dict(kind=name,value=value,old=old,difference=value-old,stats=stats))
    out['cpu_seconds']=time.process_time()-start
    (HERE/'controls_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print('controls',len(out['distributions']),len(out['models']),'CPU',out['cpu_seconds'])

if __name__=='__main__':
    try:run()
    except Exception as exc:
        import traceback
        (HERE/'controls_failure.json').write_text(json.dumps(dict(exception=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n');raise
