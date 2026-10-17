from pathlib import Path
from fractions import Fraction as F
import json,time,math,hashlib
from scipy.integrate import quad
from scipy.special import ndtr
from direct_gaussian import phi
from selected import selected
from two_pair import integrate as positive_new
from negative_rank import integrate as negative_new
from readout_cone import certificate as readout_certificate
from certificate import parameter_certificate,uniform_output

def main():
    start=time.process_time()
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth');s=torch.load(path,map_location='cpu',weights_only=True)
    model=(s['rnn.weight_ih_l0'].reshape(-1).tolist(),s['rnn.weight_hh_l0'].tolist(),s['linear.weight'].tolist());parameter_certificate(model)
    wi,wh,wo=model;alpha,beta=-F(wi[0]),F(wi[1]);a,b,c,d=F(wh[0][0]),-F(wh[0][1]),F(wh[1][0]),-F(wh[1][1]);kappa=a+a*a*beta/(c*alpha)
    certs={sign:readout_certificate(alpha,beta,a,c,wo,sign) for sign in ('+','-')};kread=F(certs['+']['threshold']);kneg=F(certs['-']['threshold']);kpos=b*beta/alpha
    assert certs['+']['winner']==9 and certs['-']['winner']==6
    geometries=[]
    for csign,k in (('+',kpos),('-',kneg)):
        word=(-1,kappa+1,1 if csign=='+' else -1,-(k+1));r=uniform_output(model,word);assert r['uniform'] and r['winner']==6
        r.update(csign=csign,word=[str(x) for x in word],controls=[])
        for name,readout,winner in [('zero',[[0,0]]*10,0),('duplicate',[[0,0],[2,2],[2,2]]+[[0,0]]*7,1)]:
            q=uniform_output((wi,wh,readout),word);assert q['uniform'] and q['winner']==winner;q['name']=name;r['controls'].append(q)
        geometries.append(r)
    pair1=math.atan2(1.,float(kappa))/(2*math.pi);wrong_event=pair1*math.atan2(1.,float(kpos))/(2*math.pi)
    old_thresholds={'++':d,'-+':kread,'+-':kpos,'--':kneg};old_mass=sum(math.atan2(1.,float(k))/(8*math.pi) for k in old_thresholds.values())
    rows=[];unions=[]
    for outer,inner in ((64,32),(128,32),(64,64),(128,64)):
        stamp=time.process_time();old=[selected(10,float(k),signs,outer) for signs,k in old_thresholds.items()]
        pnew=[positive_new(10,float(kappa),float(k),csign,outer,inner) for csign,k in (('+',d),('-',kread))]
        neg=negative_new(10,float(kappa),float(kneg),outer,inner);neg['selected_mass']=neg['joint_by_target'][6];rows.append(neg)
        event=old_mass+sum(r['event_mass'] for r in pnew)+neg['event_mass']+wrong_event
        correct=sum(r['correct_mass'] for r in old)+sum(r['joint_by_target'][9] for r in pnew)+neg['selected_mass']
        unions.append({'outer':outer,'inner':inner,'event_mass':event,'correct_mass':correct,'unresolved_mass':1-event,'new_negative_mass':neg['event_mass'],'new_negative_correct_mass':neg['selected_mass'],'new_always_wrong_mass':wrong_event,'fresh_integration_cpu_seconds':time.process_time()-stamp,'old_rows':old,'positive_new_rows':pnew})
    reference=None
    if kneg==0:
        answer,error,info=quad(lambda z:float(phi(z))*float(ndtr(-float(kappa)*z))*float(ndtr(-z))**8,0,10,epsabs=1e-15,epsrel=1e-12,full_output=1)
        reference={'selected_mass':answer,'estimated_error':error,'nodes':info['neval'],'tail_bound':float(ndtr(-10)),'roundoff_or_quadrature_certificate':False}
    out={'geometries':geometries,'readout_certificates':certs,'negative_rank_rows':rows,'unions':unions,'adaptive_scalar_reference':reference,'cpu_seconds':time.process_time()-start,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sampled_inputs':0,'complete_accuracy':False,'negative_selected_target_accounted':True}
    Path(__file__).with_name('negative_pilot_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'unions':[{k:v for k,v in r.items() if k not in ('old_rows','positive_new_rows')} for r in unions],'adaptive_scalar_reference':reference},indent=2))
if __name__=='__main__':main()
