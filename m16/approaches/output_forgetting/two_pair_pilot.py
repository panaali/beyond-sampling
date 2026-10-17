from pathlib import Path
from fractions import Fraction as F
import math,json,time,hashlib
from readout_cone import certificate as readout_certificate
from selected import selected
from two_pair import integrate
from certificate import parameter_certificate,uniform_output

def main():
    start=time.process_time()
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth');s=torch.load(path,map_location='cpu',weights_only=True)
    model=(s['rnn.weight_ih_l0'].reshape(-1).tolist(),s['rnn.weight_hh_l0'].tolist(),s['linear.weight'].tolist());parameter_certificate(model)
    wi,wh,wo=model;alpha,beta=-F(wi[0]),F(wi[1]);a,b,c,d=F(wh[0][0]),-F(wh[0][1]),F(wh[1][0]),-F(wh[1][1]);kappa=a+a*a*beta/(c*alpha)
    readouts={sign:readout_certificate(alpha,beta,a,c,wo,sign) for sign in ('+','-')}
    kread=F(readouts['+']['threshold']);kneg=F(readouts['-']['threshold'])
    # Exact attained supremum and strict below-threshold falsifier at prior(p,q)=(beta/c,0).
    prior=(beta/c,F(0));h1=a*prior[0]+alpha;h2=max(F(0),c*prior[0]-beta)
    assert a*h1-b*h2==alpha*kappa
    first_after=max(F(0),a*h1-b*h2-alpha*kappa)
    first_below=max(F(0),a*h1-b*h2-alpha*kappa/2)
    assert first_after==0 and first_below>0
    geometries=[];rows=[]
    for csign,k in (('+',d),('-',kread)):
        word=(-1,kappa+1,1 if csign=='+' else -1,k+1)
        r=uniform_output(model,word);assert r['uniform'] and r['winner']==9
        r.update(csign=csign,word=[str(v) for v in word],controls=[])
        for name,readout,winner in [('zero',[[0,0]]*10,0),('duplicate',[[0,0],[2,2],[2,2]]+[[0,0]]*7,1)]:
            q=uniform_output((wi,wh,readout),word);assert q['uniform'] and q['winner']==winner;q['name']=name;r['controls'].append(q)
        geometries.append(r)
        for outer,inner in ((64,32),(128,32),(64,64),(128,64)):
            r=integrate(10,float(kappa),float(k),csign,outer,inner);r['correct_mass']=r['joint_by_target'][9];rows.append(r)
    old_thresholds={'++':d,'-+':kread,'+-':b*beta/alpha,'--':kneg};old_rows=[]
    for outer in (64,128):
        for signs,k in old_thresholds.items():old_rows.append(selected(10,float(k),signs,outer))
    old_mass=sum(math.atan2(1.,float(k))/(8*math.pi) for k in old_thresholds.values());unions=[]
    for outer,inner in ((64,32),(128,32),(64,64),(128,64)):
        rr=[r for r in rows if r['outer']==outer and r['inner']==inner];mass=sum(r['event_mass'] for r in rr)
        contribution=sum(r['correct_mass'] for r in rr)
        unions.append({'outer':outer,'inner':inner,'new_event_mass':mass,'new_correct_mass':contribution,'total_event_mass':mass+old_mass,'total_correct_mass':contribution+sum(r['correct_mass'] for r in old_rows if r['outer']==outer),'unresolved_mass':1-mass-old_mass,'new_partition_error':sum(r['partition_error'] for r in rr)})
    out={'kappa':str(kappa),'exact_supremum':str(alpha*kappa),'exact_boundary_first_coordinate':str(first_after),'below_threshold_first_coordinate':str(first_below),'geometries':geometries,'readout_certificates':readouts,'new_rank_rows':rows,'old_selected_rows':old_rows,'unions':unions,'cpu_seconds':time.process_time()-start,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sampled_inputs':0,'complete_accuracy':False}
    Path(__file__).with_name('two_pair_pilot_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'kappa':float(kappa),'unions':unions},indent=2))
if __name__=='__main__':main()
