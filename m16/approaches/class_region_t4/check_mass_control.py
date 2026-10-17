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
                out['rows'].append(dict(kind=kind,multiplier=mult,estimate=r['estimate'],analytic_error=r['estimate']-.25,stats=r['stats'],named_ops=r['named_ops'],cpu_with_rule_setup=time.process_time()-ts))
                print(json.dumps(out['rows'][-1]),flush=True)
            assert abs(out['rows'][-1]['analytic_error'])<1e-10
        out.update(complete=True,cpu_seconds=time.process_time()-start)
    finally:R.engine._RULES=saved
    (R.HERE/'mass_control_results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
