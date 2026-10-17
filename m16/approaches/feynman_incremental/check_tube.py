"""Frozen uniform-face radial tube parity checks; no accuracy scoring."""
import json,time,hashlib,warnings
from pathlib import Path
import numpy as np
import torch
from .tube_estimator import estimate
from .check_estimator import CHECKPOINT
from .adapter import WalkFailure
HERE=Path(__file__).parent

def main():
    start=time.process_time();state=torch.load(CHECKPOINT,weights_only=True,map_location='cpu')
    model=(state['rnn.weight_ih_l0'].numpy()[:,0].astype(float),state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float))
    out=dict(checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),cases=[],random_model_inputs=0,uniform_faces=True,accuracy_scoring=False)
    for name,budget,current in (('checkpoint512',512,model),('checkpoint2048',2048,model),('zero_readout512',512,(model[0],model[1],np.zeros_like(model[2])))):
        case=dict(name=name,budget=budget);out['cases'].append(case)
        for mode in ('native','legacy'):
            if time.process_time()-start>100:raise TimeoutError('110CPU cycle guard')
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',message='The balance properties of Sobol')
                    value,cost,ledger=estimate(current,budget,legacy=mode=='legacy')
                case[mode]=dict(estimate=value,cost=cost,ledger=ledger)
                assert ledger['known_flops']==sum(ledger['components'].values())
            except WalkFailure as exc:case[mode]=dict(failure=str(exc),ledger=exc.ledger)
            out['cpu_seconds']=time.process_time()-start
            (HERE/'tube_results.json').write_text(json.dumps(out,indent=2)+'\n')
        if all('estimate' in case[m] for m in ('native','legacy')):
            case['difference']=case['native']['estimate']-case['legacy']['estimate']
            for key in ('node_sha256','face_sha256'):assert case['native']['ledger'][key]==case['legacy']['ledger'][key]
        if name=='zero_readout512':case['analytic_truth']=.1
        print(json.dumps(dict(name=name,native=case['native'].get('estimate'),legacy=case['legacy'].get('estimate'),difference=case.get('difference'),cpu=time.process_time()-start)),flush=True)
    out.update(complete=True,cpu_seconds=time.process_time()-start)
    (HERE/'tube_results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
