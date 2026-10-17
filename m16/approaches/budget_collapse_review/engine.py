"""Pinned corrected walker; cost-targeted baseline, not certified hard-budget estimator."""
import hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/clean-neurons/m16/approaches')
PINS={'sci_tao_v2/core.py':'aea5f9008fe9cc8a5436226f1057b5e0527f119d832df536127d8370bc7968f9',
      'sci_tao/core.py':'a78a45c1fae015507fb7b6a9764064a9b2425f52ff599f378408640f0617170e'}
for path,digest in PINS.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
spec=importlib.util.spec_from_file_location('budget_corrected_walker',ROOT/'sci_tao_v2/core.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
MANIFEST=HERE.parent/'walkfree_gain_review/checkpoint_manifest.json'
MANIFEST_SHA='c9ff1ffeb06ab88b23a371971fa3287160b7264732735aa99c00de0873be84d9'
assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest()==MANIFEST_SHA
MODELS={(r['H'],r['T'],r['seed']):r for r in json.loads(MANIFEST.read_text())['models']}


def model(T,seed):
    record=MODELS[16,T,seed];path=Path(record['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    state=torch.load(path,weights_only=True,map_location='cpu')
    return tuple(state[k].numpy().astype(float).reshape(-1) if i==0 else state[k].numpy().astype(float)
                 for i,k in enumerate(['rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight']))


def fixed_cost(T,H=16):
    unit=C.fpe_flops(H,T)
    return dict(unit=unit,table_flops=30_000_000,configuration_flops=32*T*(T-1)//2*(T-1),
                allocation_flops=256,setup_fpe=(30_000_000+32*T*(T-1)//2*(T-1)+256)/unit)


def faces(M,n,seed,keep_masses=False):
    T=M[2].shape[0];unit=C.fpe_flops(len(M[0]),T);_,Z23,Z12=C.tables(T)
    aggregate=0.;walk=0.;counts={};kindcost={};details={};started=time.process_time()
    for kind,Z,offset in [('23',Z23,0),('12',Z12,7)]:
        U=C.sobol(T,n,seed+offset);mass=[];cost=0.
        for lo in range(0,n,4096):
            y,a,b,tau=C.face_points(T,kind,U[lo:lo+4096])
            values,cs,cells,work=C.walk_mass(M,y,a,b,tau,kind,details=True)
            mass.extend(values.tolist());cost+=float(cs.sum())
            counts['cells']=counts.get('cells',0)+int(cells.sum())
            for name,vals in work.items():counts[name]=counts.get(name,0)+int(vals.sum())
        aggregate+=Z*float(np.mean(mass));walk+=cost;kindcost[kind]=cost
        if keep_masses:details[kind]=mass
    return dict(estimate=1-aggregate,walk_fpe=walk,node_flops=2*n*100*T,
                variable_fpe=walk+2*n*100*T/unit,kind_walk_fpe=kindcost,counts=counts,
                faces_per_kind=n,cpu_seconds=time.process_time()-started,**({'masses':details} if keep_masses else {}))


def calibrate(M):
    T=M[2].shape[0];pilot=faces(M,64,44001);fixed=fixed_cost(T)
    per_pair=pilot['walk_fpe']/64+200*T/fixed['unit']
    plans=[]
    for B in [16000,64000]:
        available=B-fixed['setup_fpe']-pilot['variable_fpe']
        n=int(np.floor(available/per_pair))
        plans.append(dict(budget=B,feasible=n>=16,faces_per_kind=max(16,n),
                          predicted_cold_fpe=fixed['setup_fpe']+pilot['variable_fpe']+max(16,n)*per_pair))
    # Do not retain pilot masses or estimates in the plan; allocation uses costs only.
    pilot.pop('estimate')
    return dict(fixed=fixed,pilot=pilot,plans=plans)


def est(M,budget,rng):
    """Complete per-call interface. Arbitrary positive budgets, expected-cost targeting."""
    if not np.isfinite(budget) or budget<=0:raise ValueError('positive finite budget required')
    T=M[2].shape[0];pilot=faces(M,64,44001);fixed=fixed_cost(T,len(M[0]))
    per_pair=pilot['walk_fpe']/64+200*T/fixed['unit']
    n=int(np.floor((budget-fixed['setup_fpe']-pilot['variable_fpe'])/per_pair))
    if n<16:raise ValueError('budget cannot fund frozen setup, pilot and minimum rule')
    result=faces(M,n,int(rng.integers(1,2**30)))
    return result['estimate'],fixed['setup_fpe']+pilot['variable_fpe']+result['variable_fpe']
