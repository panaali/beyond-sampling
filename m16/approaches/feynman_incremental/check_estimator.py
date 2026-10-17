"""Frozen paired-node port checks; no accuracy truth or Monte Carlo inputs."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .estimator import Engine
from .adapter import WalkFailure

HERE=Path(__file__).parent
CHECKPOINT=Path('/workspace/work/t39_sources/example_checkpoint.pth')
def main():
    start=time.process_time()
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()=='498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'
    state=torch.load(CHECKPOINT,weights_only=True,map_location='cpu')
    model=(state['rnn.weight_ih_l0'].numpy()[:,0].astype(float),state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float))
    out=dict(checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),cases=[],random_model_inputs=0,
             accuracy_truth_not_evaluated=True,legacy_result_not_reference_truth=True)
    for name,budget,current in (('checkpoint512',512,model),('checkpoint2048',2048,model),('zero_readout512',512,(model[0],model[1],np.zeros_like(model[2])))):
        case=dict(name=name,budget=budget)
        for mode in ('native','legacy'):
            if time.process_time()-start>100:raise TimeoutError('Frozen110CPU cycle guard')
            try:
                value,cost,ledger=Engine(legacy=mode=='legacy').estimate(current,budget)
                case[mode]=dict(estimate=value,cost=cost,ledger=ledger)
                assert ledger['known_flops']==sum(ledger['components'].values())
            except WalkFailure as exc:case[mode]=dict(failure=str(exc),ledger=exc.ledger)
            out['cpu_seconds']=time.process_time()-start
            (HERE/'estimator_results.json').write_text(json.dumps(out|{'partial_case':case},indent=2)+'\n')
        if 'estimate' in case['native'] and 'estimate' in case['legacy']:
            a,b=case['native'],case['legacy']
            assert a['ledger']['node_sha256']==b['ledger']['node_sha256']
            assert a['ledger']['face_sha256']==b['ledger']['face_sha256']
            case['native_minus_legacy']=a['estimate']-b['estimate']
        out['cases'].append(case)
        print(json.dumps(dict(name=name,native=case['native'].get('estimate'),legacy=case['legacy'].get('estimate'),difference=case.get('native_minus_legacy'),cpu=time.process_time()-start)),flush=True)
    # Invalid requests fail explicitly; these small controls do not sample model inputs.
    engine=Engine();out['invalid_controls']=[]
    for budget in (float('nan'),0.,1e12):
        try:engine.estimate(model,budget)
        except WalkFailure as exc:
            assert exc.ledger['known_flops']>0 and not exc.ledger['complete']
            out['invalid_controls'].append(dict(request=str(budget),failure=str(exc),ledger=exc.ledger))
        else:raise AssertionError('Invalid request accepted')
    out.update(complete=True,cpu_seconds=time.process_time()-start)
    (HERE/'estimator_results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
