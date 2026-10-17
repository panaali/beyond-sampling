"""Interface-state controls using injected scalar callbacks, no model integrations."""
import json,time
from pathlib import Path
import numpy as np
from .estimator import Engine,TABLE_POINTS
from .adapter import WalkFailure

def main():
    start=time.process_time();model=(np.ones(1),np.ones((1,1)),np.ones((10,1)));out={}
    engine=Engine();engine.namespace['estimate']=lambda m,b:(.25,0.,{'injected':True})
    first=engine.estimate(model,1)[2];second=engine.estimate(model,1)[2]
    assert first['known_flops']-second['known_flops']==500*TABLE_POINTS
    assert second['saved_warm_setup_flops']==500*TABLE_POINTS
    out['cold_warm']=[first,second]
    def bad(*args):raise ValueError('Injected kernel error after validation')
    engine.namespace['estimate']=bad
    try:engine.estimate(model,1)
    except WalkFailure as exc:
        assert exc.ledger['failure_kind']=='numerical_or_implementation'
        assert exc.ledger['cost_is_partial'] and exc.ledger['unresolved_failed_kernel_cost']
        out['kernel_failure']=exc.ledger
    else:raise AssertionError('Injected failure lost')
    try:engine.estimate(model,float('nan'))
    except WalkFailure as exc:
        assert exc.ledger['failure_kind']=='invalid_input' and not exc.ledger['cost_is_partial']
        assert not exc.ledger['unresolved_failed_kernel_cost']
        out['invalid_budget']=exc.ledger
    else:raise AssertionError('Invalid budget accepted')
    out.update(complete=True,cpu_seconds=time.process_time()-start,model_integrations=0)
    Path(__file__).with_name('interface_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'complete':True,'cpu_seconds':out['cpu_seconds']}))
if __name__=='__main__':main()
