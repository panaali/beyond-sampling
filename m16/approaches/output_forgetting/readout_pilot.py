from pathlib import Path
from fractions import Fraction as F
import json,time,hashlib
from readout_cone import certificate
from certificate import parameter_certificate
from signed_rank import integrate
from selected import selected

def main():
    start=time.process_time()
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth');s=torch.load(path,map_location='cpu',weights_only=True)
    model=(s['rnn.weight_ih_l0'].reshape(-1).tolist(),s['rnn.weight_hh_l0'].tolist(),s['linear.weight'].tolist());parameter_certificate(model)
    wi,wh,wo=model;alpha,beta=-F(wi[0]),F(wi[1]);a,b,c,d=F(wh[0][0]),-F(wh[0][1]),F(wh[1][0]),-F(wh[1][1])
    t=time.process_time();certificates=[certificate(alpha,beta,a,c,wo,sign) for sign in ('+','-')];certificate_cpu=time.process_time()-t
    ks={'++':d,'-+':F(certificates[0]['threshold']),'+-':b*beta/alpha,'--':F(certificates[1]['threshold'])};rows=[];fast=[]
    for signs,k in ks.items():
        axis=int(signs[1]=='+');winner=max(range(len(wo)),key=lambda i:F(wo[i][axis]))
        assert winner==(9 if axis else 6),'selected class changed; full evaluator required'
        for outer,inner in ((64,32),(128,32),(64,64),(128,64)):
            r=integrate(10,float(k),signs,outer,inner);r['winner']=winner;r['correct_mass']=r['joint_by_target'][winner];rows.append(r)
        for outer in (64,128):fast.append(selected(10,float(k),signs,outer))
    unions=[]
    for outer,inner in ((64,32),(128,32),(64,64),(128,64)):
        rr=[r for r in rows if r['outer']==outer and r['inner']==inner]
        mass=sum(r['event_mass'] for r in rr)
        unions.append({'outer':outer,'inner':inner,'event_mass':mass,'correct_mass':sum(r['correct_mass'] for r in rr),'unresolved_mass':1-mass,'partition_error':sum(r['partition_error'] for r in rr)})
    out={'certificate_cpu_seconds':certificate_cpu,'certificates':certificates,'full_rank_rows':rows,'selected_rows':fast,'unions':unions,'cpu_seconds':time.process_time()-start,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sampled_inputs':0,'full_accuracy_claim':False}
    Path(__file__).with_name('readout_pilot_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'certificate_cpu_seconds':certificate_cpu,'thresholds':[(r['sign'],float(F(r['old_threshold'])),float(F(r['threshold']))) for r in certificates],'unions':unions,'selected_union64':sum(r['correct_mass'] for r in fast if r['outer']==64),'selected_union128':sum(r['correct_mass'] for r in fast if r['outer']==128)},indent=2))
if __name__=='__main__':main()
