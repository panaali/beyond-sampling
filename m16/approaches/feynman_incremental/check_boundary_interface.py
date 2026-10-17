"""Setup error accounting; zero geometry/model integrations."""
import json,time
from pathlib import Path
import numpy as np
from .boundary_estimator import Engine
from .adapter import WalkFailure

def main():
    start=time.process_time();old=Engine._setup
    def fail(self):self.add('injected_setup',17);raise RuntimeError('Injected setup failure')
    Engine._setup=fail
    try:
        try:Engine(np.random.default_rng(0))
        except WalkFailure as exc:
            assert exc.ledger['known_flops']==11017 and exc.ledger['failure_kind']=='setup'
            assert not exc.ledger['complete'] and exc.ledger['unresolved_failed_kernel_cost']
            out=dict(complete=True,failure=str(exc),ledger=exc.ledger,model_integrations=0,cpu_seconds=time.process_time()-start)
        else:raise AssertionError('Setup failure lost')
    finally:Engine._setup=old
    Path(__file__).with_name('boundary_interface_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
