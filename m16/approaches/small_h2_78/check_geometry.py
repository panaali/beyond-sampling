import json,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scalar_tail import output_intervals,interior,rank_integral,conditional_tail
from envelope import envelope,reset_certificate


def main():
    start=time.process_time();import torch
    p=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth',map_location='cpu',weights_only=True)
    base=tuple(p[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    base=(base[0].reshape(-1),base[1],base[2])
    rng=np.random.default_rng(49005);count=0;points=0;max_error=0.;stats=[];rounding_cases=[]
    for sigma in (0.,.0025,.02):
        noise=np.random.default_rng(1)
        model=tuple(w+sigma*np.sqrt(np.mean(w*w))*noise.standard_normal(w.shape) for w in base)
        for h in [np.zeros(2),np.array([0.,1.]),np.array([1.,0.]),*rng.exponential(size=(29,2))]:
            fast,sf=output_intervals(model,h,use_envelope=True)
            full,sa=output_intervals(model,h,use_envelope=False)
            cuts=sorted({a for a,b,j in fast+full}|{b for a,b,j in fast+full})
            for a,b in zip(cuts,cuts[1:]):
                y=interior(a,b)
                pred=int(np.argmax(model[2]@np.maximum(model[1]@h+model[0]*y,0)))
                qfast=next(j for c,d,j in fast if c<=y<=d)
                qall=next(j for c,d,j in full if c<=y<=d)
                if not pred==qfast==qall:
                    mass=float(ndtr(b)-ndtr(a))
                    assert mass<=1e-13,(h,y,pred,qfast,qall,a,b,mass)
                    rounding_cases.append({'sigma':sigma,'hidden':h.tolist(),'y':float(y),'left':float(a),'right':float(b),'direct':pred,'fast':qfast,'all_pairs':qall,'gaussian_interval_mass':mass})
                points+=1
            count+=1
    # Degenerate rows touching a winning envelope vertex and duplicate rows.
    outputs=[np.ones((8,2)),np.array([[1.,0.],[0.,1.],[.5,.5],[1.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.]])]
    for wo in outputs:
        for h in ([1.,1.],[0.,0.],[1.,0.]):
            model=(base[0],base[1],wo)
            fast,_=output_intervals(model,np.array(h),use_envelope=True)
            full,_=output_intervals(model,np.array(h),use_envelope=False)
            cuts=sorted({a for a,b,j in fast+full}|{b for a,b,j in fast+full})
            for a,b in zip(cuts,cuts[1:]):
                y=interior(a,b)
                assert next(j for c,d,j in fast if c<=y<=d)==next(j for c,d,j in full if c<=y<=d)
                points+=1
            count+=1
    for known in ([.2],[1.4],[.7,-.5],[.1,.8],[.7,-.5,.2]):
        m=8-len(known)-3;fast,sf=conditional_tail(base,m,known)
        h=np.zeros(2)
        for x in known:h=np.maximum(base[1]@h+base[0]*x,0)
        full,sa=output_intervals(base,h,lower=0. if known[-1]<0 else -np.inf,use_envelope=False)
        reference=sum(rank_integral(m,j,list(ndtr(known)),float(ndtr(a)),float(ndtr(b)))[0] for a,b,j in full)
        max_error=max(max_error,abs(fast-reference));assert abs(fast-reference)<1e-12
        stats.append({'known':known,'fast_cells':sf['readout_cells'],'all_pair_cells':sa['readout_cells'],
                      'polynomial_calls':sf['polynomial_probability_calls']})
    try:conditional_tail(base,3,[.2,.2])
    except ValueError:pass
    else:raise AssertionError('known-input ties must be rejected')
    pilot_path=Path(__file__).with_name('LOCALPILOT_H2_T7_seed0.npz');data=np.load(pilot_path)
    pilot=tuple(data[k].astype(float) for k in ('Wi','Wh','Wo'))
    pilot_result={'readout':envelope(pilot[2]),'reset':reset_certificate(pilot),
                  'Wi':pilot[0].tolist(),'Wh':pilot[1].tolist()}
    out={'geometry_states':count,'comparison_intervals':points,'max_conditional_error':max_error,
         'stats':stats,'pilot':pilot_result,'cpu_seconds':time.process_time()-start,
         'random_hidden_states':87,'random_network_input_samples':0,'rounding_cases':rounding_cases,
         'total_mismatched_interval_mass':sum(r['gaussian_interval_mass'] for r in rounding_cases)}
    Path(__file__).with_name('geometry_checks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='pilot'},indent=2))
    print(json.dumps({'pilot_support':pilot_result['readout']['possible_labels'],'pilot_reset':pilot_result['reset'],
                      'Wi':pilot_result['Wi'],'Wh':pilot_result['Wh']},indent=2))
if __name__=='__main__':main()
