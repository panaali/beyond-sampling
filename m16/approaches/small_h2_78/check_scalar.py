import json,time
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.integrate import quad
from scalar_tail import conditional_tail,output_intervals,rank_integral
from prefix import normal_probability,cdf_probability


def main():
    start=time.process_time()
    import torch
    p=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth',map_location='cpu',weights_only=True)
    model=tuple(p[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    model=(model[0].reshape(-1),model[1],model[2]);rows=[];total_calls=0;reference_model_calls=0;reference_rank_calls=0
    for known in ([],[.2],[1.4],[.7,-.5],[.1,.8],[.7,-.5,.2]):
        m=8-len(known)-3
        result,stats=conditional_tail(model,m,known)
        wi,wh,wo=model;h=np.zeros(2)
        for x in known:h=np.maximum(wh@h+wi*x,0)
        lower=.5 if not known or known[-1]<0 else 0.
        intervals,_=output_intervals(model,h,lower=0. if lower else -np.inf)
        # Independent direct ReLU network at scalar quadrature points; shared
        # geometry gives breakpoints but correctness recomputed independently.
        from scipy.special import ndtr
        cuts=sorted({lower,1.,*[float(ndtr(a)) for a,b,j in intervals if lower<float(ndtr(a))<1.],
                     *[float(ndtr(x)) for x in known],.5})
        cuts=[v for v in cuts if lower<=v<=1.]
        def direct(v):
            nonlocal reference_model_calls
            reference_model_calls+=1
            y=float(ndtri(v));last=np.maximum(wh@h+wi*y,0);j=int(np.argmax(wo@last))
            return normal_probability(m,j,[*known,y])
        reference=sum(quad(direct,a,b,epsabs=2e-11)[0] for a,b in zip(cuts,cuts[1:]) if a<b)
        errors=[]
        for scale in (1e-16,1e16):
            scaled=(wi*scale,wh,wo*scale)
            alt,_=conditional_tail(scaled,m,known);errors.append(abs(alt-result))
        assert abs(result-reference)<2e-10,(known,result,reference)
        assert max(errors)<2e-10
        rows.append({'known':known,'m':m,'analytic':result,'reference':reference,
                     'abs_error':abs(result-reference),'scale_error':max(errors),'stats':stats})
        total_calls+=stats['polynomial_probability_calls']
    # Constant/duplicate readout: all first-index ties, including positive mass.
    tied=(model[0],model[1],np.ones((8,2)))
    for known in ([.2],[.2,-.4]):
        m=8-len(known)-3
        result,stats=conditional_tail(tied,m,known)
        lo=.5 if known[-1]<0 else 0.
        known_cdfs=list(__import__('scipy').special.ndtr(known))
        def direct_rank(v):
            nonlocal reference_rank_calls
            reference_rank_calls+=1
            return float(cdf_probability(m,0,[*known_cdfs,v]))
        ref=quad(direct_rank,lo,1.,points=sorted({x for x in [*known_cdfs,.5] if lo<x<1.}),epsabs=1e-12)[0]
        assert abs(ref-result)<1e-10
        rows.append({'case':'constant_tied_readout','known':known,'analytic':result,'reference':ref,'abs_error':abs(ref-result)})
    out={'rows':rows,'max_abs_error':max(r['abs_error'] for r in rows),
         'compute':{'cpu_seconds':time.process_time()-start,'conditional_tail_calls':len(rows)+12,
                    'reported_base_polynomial_calls':total_calls,'random_inputs':0,
                    'reference_quadrature_model_inputs':reference_model_calls,
                    'reference_rank_only_calls':reference_rank_calls}}
    Path(__file__).with_name('scalar_checks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
