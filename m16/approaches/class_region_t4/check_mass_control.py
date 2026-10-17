"""Controls-only diagnosis of default Plackett rule error, no trained model."""
import json,time,resource
import numpy as np
from . import runner as R

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));start=time.process_time();saved=R.engine._RULES.copy();out=dict(rows=[],default_failure_preserved=True,trained_cases=0)
    model=(np.array([1.]),np.zeros((1,1)),np.zeros((4,1)))
    try:
        for kind in ('old','quadratic','nearest'):
            stats=None
            for mult in (1,2,4):
                ts=time.process_time();R.engine._RULES={'wide':R.engine._rule(12*mult,2),'thin':R.engine._rule(48*mult,3)}
                r=R.partial_run(model,kind,'exact',cpu=10);assert r['complete']
                if stats is None:stats=r['stats']
                else:assert stats==r['stats']
                ops=R.Ops(10);hs=R.families(kind,ops);per_family=[];groups={}
                for h,row in zip(hs,r['rows']):
                    mass=R.engine.mass(h['cone'],ops);expected=mass if h['truth']!=0 else 0.
                    pair=tuple(h['perm'][:2]) if kind=='old' else h['pair'];key=(int(h['sigma']),*pair)
                    groups[key]=groups.get(key,0.)+mass
                    per_family.append(dict(index=row['index'],direct_mass=mass,network_error=row['error'],residual=row['error']-expected))
                chart_res=[]
                for (sign,a,b),mass in groups.items():
                    positive=15/16 if a==0 else 11/16 if b==0 else 3/16
                    exact=(positive if sign>0 else 1-positive)/12
                    chart_res.append(dict(sign=sign,a=a,b=b,mass=mass,exact=exact,residual=mass-exact))
                out['rows'].append(dict(kind=kind,multiplier=mult,estimate=r['estimate'],analytic_error=r['estimate']-.25,stats=r['stats'],named_ops=r['named_ops'],diagnostic_mass_ops=ops.n,per_family=per_family,rank_sign_charts=chart_res,max_family_residual=max(abs(v['residual']) for v in per_family),max_chart_residual=max(abs(v['residual']) for v in chart_res),cpu_with_rule_setup=time.process_time()-ts))
                print(json.dumps({k:v for k,v in out['rows'][-1].items() if k not in ('per_family','rank_sign_charts')}),flush=True)
            assert abs(out['rows'][-1]['analytic_error'])<1e-10
        out.update(complete=True,cpu_seconds=time.process_time()-start)
    finally:R.engine._RULES=saved
    (R.HERE/'mass_control_results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
