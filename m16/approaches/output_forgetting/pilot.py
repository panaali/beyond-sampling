from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ndtri
from certificate import parameter_certificate,uniform_output
from rank_mass import joint_mass,pair_owen,coefficients

def main():
    start=time.process_time()
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth');state=torch.load(path,map_location='cpu',weights_only=True)
    model=(state['rnn.weight_ih_l0'].reshape(-1).tolist(),state['rnn.weight_hh_l0'].tolist(),state['linear.weight'].tolist())
    cert=parameter_certificate(model);d=float(cert['d']);winner=cert['winner'];geometry=[]
    for word in ((1,1,1,1),(1,2,3,2),(3,1,F(1,2),1)):
        r=uniform_output(model,word);r['word']=[str(x) for x in word]
        if not r['uniform'] or r['winner']!=winner:raise AssertionError('event certificate fails')
        geometry.append(r)
    outside=uniform_output(model,(1,1,3,F(1,20)))
    controls=[]
    for name,wo,expected in (('zero',[[0,0]]*10,0),('duplicate',[[0,0],[0,2],[0,2]]+[[0,0]]*7,1)):
        r=uniform_output((model[0],model[1],wo),(1,1,1,1));r['name']=name
        if not r['uniform'] or r['winner']!=expected:raise AssertionError('readout tie control fails')
        controls.append(r)
    rows=[]
    for outer,inner in ((32,32),(64,32),(32,64),(64,64)):
        r=joint_mass(10,d,outer,inner);r['correct_event_mass']=r['joint_by_target'][winner];r['conditional_correct_given_event']=r['correct_event_mass']/r['event_mass'];rows.append(r)
    calls=0
    def integrand(Fvalue):
        nonlocal calls
        calls+=1;u=float(ndtri(Fvalue));pair=pair_owen(u,Fvalue,d)
        if min(pair)<0:raise ArithmeticError(('Owen pair cancellation',Fvalue,pair))
        return coefficients(10,d,u,Fvalue,pair)
    t=time.process_time();reference,error,info=quad_vec(integrand,.5,1,epsabs=1e-12,epsrel=1e-10,full_output=True)
    reference_vector=[reference[0]]*6+[reference[1],reference[1],reference[2],reference[3]]
    ref={'classes':reference.tolist(),'joint_by_target':[float(v) for v in reference_vector],'estimated_error':float(error),'success':bool(info.success),'neval':info.neval,'owen_calls':2*calls,'cpu_seconds':time.process_time()-t,'correct_event_mass':float(reference_vector[winner])}
    result={'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'certificate':{k:[str(x) for x in v] if isinstance(v,list) else str(v) if isinstance(v,F) else v for k,v in cert.items()},'geometry_cases':geometry,'outside_event_falsifier':outside,'readout_controls':controls,'quadrature':rows,'adaptive_reference':ref,'cpu_seconds':time.process_time()-start,'sampled_inputs':0,'complete_accuracy_estimate':False,'unresolved_event_mass':1-rows[0]['event_mass']}
    Path(__file__).with_name('pilot_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':result['cpu_seconds'],'winner':winner,'event_mass':rows[0]['event_mass'],'correct_contributions':[r['correct_event_mass'] for r in rows],'reference':ref,'outside_uniform':outside['uniform'],'max_partition_error':max(abs(r['partition_error']) for r in rows)},indent=2))
if __name__=='__main__':main()
