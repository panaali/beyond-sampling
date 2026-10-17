from pathlib import Path
from fractions import Fraction as F
import json,time,hashlib
from certificate import uniform_output,parameter_certificate
from signed_rank import integrate

def main():
    start=time.process_time()
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth');s=torch.load(path,map_location='cpu',weights_only=True)
    model=(s['rnn.weight_ih_l0'].reshape(-1).tolist(),s['rnn.weight_hh_l0'].tolist(),s['linear.weight'].tolist())
    parameter_certificate(model);wi,wh,wo=model;alpha,beta=-F(wi[0]),F(wi[1]);a,b,c,d=F(wh[0][0]),-F(wh[0][1]),F(wh[1][0]),-F(wh[1][1])
    thresholds={'++':d,'-+':a,'+-':b*beta/alpha,'--':c*alpha/beta};geometries=[];rows=[]
    for signs,k in thresholds.items():
        axis=int(signs[1]=='+');winner=max(range(10),key=lambda i:F(wo[i][axis]))
        word=(1,1,1 if signs[0]=='+' else -1,(k+1)*(1 if signs[1]=='+' else -1))
        r=uniform_output(model,word);r.update(signs=signs,threshold=str(k),word=[str(v) for v in word],axis=axis,expected_winner=winner)
        assert r['uniform'] and r['winner']==winner
        r['tie_controls']=[]
        for name,readout,expected in [('zero',[[0,0]]*10,0),('duplicate',[[0,0],[2,2],[2,2]]+[[0,0]]*7,1)]:
            q=uniform_output((wi,wh,readout),word);q['name']=name
            assert q['uniform'] and q['winner']==expected
            r['tie_controls'].append(q)
        geometries.append(r)
        for outer,inner in ((64,32),(128,32),(64,64),(128,64)):
            r=integrate(10,float(k),signs,outer,inner);r.update(winner=winner,correct_event_mass=r['joint_by_target'][winner]);rows.append(r)
    unions=[]
    for outer,inner in ((64,32),(128,32),(64,64),(128,64)):
        rr=[r for r in rows if r['outer']==outer and r['inner']==inner]
        union={'outer':outer,'inner':inner,'event_mass':sum(r['event_mass'] for r in rr),'correct_mass':sum(r['correct_event_mass'] for r in rr),'partition_error':sum(r['partition_error'] for r in rr),'cpu_seconds':sum(r['cpu_seconds'] for r in rr),'tail_bound_correct_mass':sum(r['tail_bound_per_class'] for r in rr)}
        union['unresolved_mass']=1-union['event_mass'];unions.append(union)
    out={'geometries':geometries,'quadrature':rows,'unions':unions,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cpu_seconds':time.process_time()-start,'sampled_inputs':0,'full_accuracy':False}
    Path(__file__).with_name('signed_pilot_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'events':[{key:r[key] for key in ('signs','k','winner','event_mass','correct_event_mass','partition_error')} for r in rows if r['outer']==128 and r['inner']==64],'unions':unions},indent=2))
if __name__=='__main__':main()
