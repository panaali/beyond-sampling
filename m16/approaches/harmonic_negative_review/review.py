"""Bounded T57 cut checks plus fixed-model common-harness comparison."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from hemisphere import HemisphereSpace

HERE = Path(__file__).resolve().parent
AUTHOR = Path('/workspace/clean-neurons/m16/approaches/harmonic_sphere')
sys.path.insert(0,str(AUTHOR))
import sphere
import hprop


def cut_controls(deadline):
    normals = np.array([[1,0,0],[0,0,1],[1,2,3],[-2,3,1],[1,-1,0]],dtype=float)
    normals /= np.linalg.norm(normals,axis=1)[:,None]
    out = []
    for D in (2,4,8):
        start = time.process_time()
        independent = HemisphereSpace(3,D,deadline)
        author = sphere.Space(3,D)
        P = independent.at(author.Y)
        change = author.PhiTw @ P
        orth_error = np.max(abs(change.T@change-np.eye(independent.m)))
        c = np.cos(np.arange(independent.m)*.7)/(1+np.arange(independent.m))
        C = np.repeat(c[:,None],len(normals),axis=1)
        ours = independent.cut(C,normals)
        theirs = author.cut(change@C,normals)
        discrepancy = np.max(abs(change@ours-theirs))
        complement = np.max(abs(ours+independent.cut(C,-normals)-C))
        one = np.repeat(independent.one()[:,None],len(normals),axis=1)
        half = independent.cut(one,normals)
        mass = independent.mass_in(half,normals)
        n = D//2
        deficit = Fraction(math.comb(2*n,n)**2,4*16**n)
        expected = .5-float(deficit)
        row = dict(D=D,orthogonality_error=float(orth_error),cut_coefficient_error=float(discrepancy),
                   complement_error=float(complement),constant_mass_error=float(np.max(abs(half[0]-.5))),
                   same_hemisphere_readout=mass.tolist(),exact_readout_deficit=str(deficit),
                   repeated_readout_error=float(np.max(abs(mass-expected))),cpu=time.process_time()-start,
                   independent_counts=dict(independent.counts))
        if max(orth_error,discrepancy,complement,row['constant_mass_error'],row['repeated_readout_error'])>2e-10:
            raise AssertionError(row)
        out.append(row)
    return out


def load_model(seed):
    path = Path(f'/workspace/clean-neurons/work/checkpoints/H16_T3_seed{seed}.pth')
    data = torch.load(path,weights_only=True,map_location='cpu')
    model = tuple(data[k].numpy().astype(float) for k in
                  ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    model = (model[0][:,0],model[1],model[2])
    return model,hashlib.sha256(path.read_bytes()).hexdigest()


def run_model(model,D,independent,deadline):
    old = hprop.space
    hprop._ROOTS.clear(); hprop._SPACES.clear()
    start = time.process_time()
    if independent:
        sp = HemisphereSpace(3,D,deadline)
        hprop.space = lambda n,d:sp
    else:
        sp = None
    try:
        value,info = hprop.estimate(model,D,cells='first',return_info=True)
        row = dict(estimate=float(value),cpu=time.process_time()-start,
                   leaf_count=info['nleaf'],cut_count=info['ncut'],
                   charged_harness_flops=info['flops'],charged_geometry_flops=info['lp_flops'],
                   claimed_hilton_flops=info['hilton_flops'])
        if sp is not None:
            row['independent_counts']=dict(sp.counts)
            row['independent_setup_cpu']=sp.setup_cpu
            row['cost_warning']='harness count excludes basis special-function cost; not an FPE estimate'
        return row
    finally:
        hprop.space=old
        hprop._ROOTS.clear(); hprop._SPACES.clear()


def main():
    start=time.process_time(); deadline=start+105
    output=dict(complete=False,random_model_inputs=0,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [AUTHOR/'sphere.py',AUTHOR/'hprop.py',AUTHOR/'cones.py',HERE/'hemisphere.py']})
    def save():
        output['cpu_seconds']=time.process_time()-start
        output['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (HERE/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    try:
        output['cut_controls']=cut_controls(deadline); save()
        first,hash0=load_model(0)
        second,hash1=load_model(1)
        rng=np.random.default_rng(0)
        noisy=tuple(w+.02*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in second)
        output['checkpoint_sha256']={'seed0':hash0,'seed1':hash1}
        stored=json.loads((AUTHOR/'eval_T3.json').read_text())
        output['models']=[]
        for name,model in [('seed0',first),('seed1',second),('s0.02_0',noisy)]:
            for D in (2,4):
                ours=run_model(model,D,True,deadline)
                theirs=run_model(model,D,False,deadline)
                previous=stored[str(D)][name]
                row=dict(name=name,D=D,independent_cut=ours,original_cut=theirs,
                         cut_method_difference=ours['estimate']-theirs['estimate'],
                         stored_estimate=previous['est'],stored_truth=previous['truth'],
                         stored_truth_se=previous['se'],stored_cost_fpe=previous['cost'],
                         stored_estimate_difference=theirs['estimate']-previous['est'])
                output['models'].append(row); save()
                if abs(row['cut_method_difference'])>2e-9:
                    raise AssertionError('independent cut disagrees in common network harness')
        output['tie_controls']=[]
        for name,model in [('all_zero',(np.zeros(2),np.zeros((2,2)),np.zeros((3,2)))),
                           ('duplicate_readout',(np.array([1.,-1.]),np.eye(2),np.ones((3,2))))]:
            row=run_model(model,2,True,deadline)
            output['tie_controls'].append(dict(name=name,estimate=row['estimate'],exact_truth=1/3,
                                               correct_first_index_ties=abs(row['estimate']-1/3)<1e-12))
        output['complete']=True
    except (AssertionError,ValueError,TimeoutError,FileNotFoundError) as exc:
        output['failure']=repr(exc)
    save()
    print(json.dumps({k:v for k,v in output.items() if k not in ('models','cut_controls')},indent=2))

if __name__=='__main__':
    main()
