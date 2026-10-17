"""Frozen proposal algebra, failure ledger and bounded matched-model controls."""
import hashlib,json,time,warnings,resource
from pathlib import Path
import numpy as np
import torch
from . import estimator as E
from ..feynman_incremental.check_estimator import CHECKPOINT
HERE=Path(__file__).parent

def controls():
    groups=np.repeat(np.arange(4),3);values=np.array([0,1,2,3,3,3,1,3,5,0,0,9.]);costs=np.repeat([1.,4.,9.,16.],3)
    q,fit=E.fit_proposal(values,costs,groups,4)
    # Independent enumeration of the discrete change of measure.
    f=np.array([.3,-2.,7.,11.]);expected=sum(float(v) for v in f)/4
    actual=sum(float(q[j])*float(f[j])/(4*float(q[j])) for j in range(4))
    assert abs(actual-expected)<1e-14 and np.min(q)>=.05
    order=np.arange(11,-1,-1);qr,_=E.fit_proposal(values[order],costs[order],groups[order],4)
    assert np.max(abs(q-qr))<1e-15
    qzero,z=E.fit_proposal(np.zeros(12),costs,groups,4)
    qconst,c=E.fit_proposal(np.full(12,5.),costs,groups,4)
    assert np.array_equal(qzero,np.full(4,.25)) and np.array_equal(qconst,qzero)
    errors=[]
    for args in [(values[:3],costs[:3],groups[:3],4),(values,costs*0,groups,4),(values,costs,groups+.5,4)]:
        try:E.fit_proposal(*args)
        except ValueError as exc:errors.append(str(exc))
        else:raise AssertionError('Invalid pilot accepted')
    model=(np.ones(1),np.zeros((1,1)),np.zeros((10,1)))
    invalid=[]
    for budget in (0,float('nan'),1e12):
        try:E.estimate(model,budget)
        except E.WalkFailure as exc:
            assert exc.ledger['failure_kind']=='invalid_input' and exc.ledger['known_flops']>0
            invalid.append(exc.ledger)
        else:raise AssertionError('Invalid budget accepted')
    original=E.walk_mass
    def failure(*args,**kwargs):raise E.WalkFailure('injected walker failure',dict(known_flops=321,unresolved_failed_kernel_cost=False))
    E.walk_mass=failure
    try:
        try:E.estimate(model,512)
        except E.WalkFailure as exc:
            assert exc.ledger['failure_stage']=='pilot' and exc.ledger['components']['radial_walker']==321
            assert exc.ledger['cost_is_partial'] and not exc.ledger['complete']
            failed=exc.ledger
        else:raise AssertionError('Walker failure swallowed')
    finally:E.walk_mass=original
    return dict(change_of_measure_error=actual-expected,proposal=q.tolist(),permutation_error=float(np.max(abs(q-qr))),fallback_uniform=True,invalid_pilot_errors=errors,invalid_budgets=invalid,injected_failure=failed)

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(110,112));start=time.process_time()
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()=='498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'
    state=torch.load(CHECKPOINT,weights_only=True,map_location='cpu');model=(state['rnn.weight_ih_l0'].numpy()[:,0].astype(float),state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float))
    result=dict(checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),controls=controls(),cases=[],random_inputs=0,accuracy_or_mse_claim=False)
    target=HERE/'results.json'
    prior=json.loads((HERE.parent/'feynman_incremental/tube_results.json').read_text())
    for idx,(name,budget,m) in enumerate((('checkpoint512',512,model),('checkpoint2048',2048,model),('zero_readout512',512,(model[0],model[1],np.zeros_like(model[2]))))):
        case=dict(name=name,budget=budget);result['cases'].append(case)
        for mode in ('uniform','advice','cold'):
            if time.process_time()-start>95:raise TimeoutError('Frozen phase guard')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='The balance properties of Sobol')
                try:
                    value,cost,ledger=E.estimate(m,budget,mode)
                    assert ledger['known_flops']==sum(ledger['components'].values())
                    case[mode]=dict(estimate=value,cost=cost,ledger=ledger)
                    if mode=='uniform':
                        delta=value-prior['cases'][idx]['native']['estimate'];assert abs(delta)<1e-13;case['prior_uniform_delta']=delta
                    if mode=='cold':assert len(ledger['records'])==4 and ledger['complete_model_dependent_setup']
                    print(json.dumps(dict(case=name,mode=mode,value=value,cost=cost,cpu=ledger['cpu_seconds'])),flush=True)
                except E.WalkFailure as exc:
                    case[mode]=dict(failure=str(exc),ledger=exc.ledger)
                    print(json.dumps(dict(case=name,mode=mode,failure=str(exc))),flush=True)
            result['cpu_seconds']=time.process_time()-start;target.write_text(json.dumps(result,indent=2)+'\n')
    result.update(complete=True,cpu_seconds=time.process_time()-start,maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    target.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
