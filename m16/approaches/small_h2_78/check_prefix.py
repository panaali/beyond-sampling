"""Bounded author checks; independent review remains required."""
import json,time,hashlib
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy.special import ndtri
from scipy.integrate import quad
from prefix import cdf_probability,normal_probability,last_reset_masses,count_integral,sign_rank_masses
from envelope import envelope,reset_certificate


def main():
    started=time.process_time();checks={};n_exact=0
    for m in range(7):
        for values in ([],[F(1,4)],[F(3,4)],[F(1,8),F(2,5)],[F(1,5),F(4,5)],
                       [F(3,5),F(7,8)],[F(1,7),F(1,3),F(2,3),F(6,7)]):
            probs=[cdf_probability(m,j,values) for j in range(m+2+len(values))]
            assert all(0<=p<=1 for p in probs)
            if values:assert sum(probs)==1,(m,values,probs)
            else:assert abs(sum(probs)-1)<1e-14
            n_exact+=1
    checks['rank_partition_cases']=n_exact
    # Independent integrand: explicit Bernoulli convolution rather than P0/P1.
    errors=[];integrals=0
    for a in range(7):
        for b in range(3):
            for k in (0,1):
                def direct(v):
                    law=np.array([1.])
                    for below in [v]*a+[min(1,2*v)]*b:
                        law=np.convolve(law,[below,1-below])
                    return law[k] if len(law)>k else 0.
                for lo,hi in ((.03,.49),(.2,.8),(.51,.97)):
                    reference=quad(direct,lo,hi,points=[.5] if lo<.5<hi else None,epsabs=1e-12)[0]
                    result=count_integral(lo,hi,a,b,k)
                    errors.append(abs(reference-result));integrals+=1
    assert max(errors)<1e-12
    checks['independent_polynomial_integrals']={'calls':integrals,'max_abs_error':max(errors)}
    # Independent conditional Gaussian draws; models are not evaluated here.
    rng=np.random.default_rng(49001);rows=[];N=65536
    for m,suffix in [(0,[-1.,.2]),(1,[.4,1.3]),(3,[-2.,-.5]),(4,[.2,-.4]),(6,[])]:
        x=np.concatenate([rng.standard_normal((N,m)),-np.abs(rng.standard_normal((N,2))),
                          np.tile(suffix,(N,1))],axis=1)
        target=np.argsort(x,axis=1)[:,-2]
        for j in range(x.shape[1]):
            exact=normal_probability(m,j,suffix);measured=float(np.mean(target==j))
            se=np.sqrt(exact*(1-exact)/N)
            z=(measured-exact)/se if se else 0.
            rows.append({'m':m,'suffix':suffix,'target':j,'analytic':exact,'mc':measured,'z':z})
    assert max(abs(row['z']) for row in rows)<5
    checks['conditional_validation']={'inputs':5*N,'rows':rows,'max_abs_z':max(abs(r['z']) for r in rows)}
    checks['last_reset_masses']=last_reset_masses(8)
    # Trained base and fixed shared-protocol noise law, one copy per sigma.
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    state=torch.load(path,map_location='cpu',weights_only=True)
    base=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    base=(base[0].reshape(-1),base[1],base[2]);model_rows=[]
    for sigma in (0.,.0025,.02):
        rng=np.random.default_rng(1)
        model=tuple(W+sigma*np.sqrt(np.mean(W**2))*rng.standard_normal(W.shape) for W in base)
        env=envelope(model[2]);cert=reset_certificate(model);assert cert['holds']
        for scale in (1e-16,1e16):
            scaled=(model[0]*scale,model[1],model[2]*scale)
            assert envelope(scaled[2])['possible_labels']==env['possible_labels']
            assert reset_certificate(scaled)['negative_negative_positive_winner']==cert['negative_negative_positive_winner']
        T=len(model[2]);K=set(env['possible_labels']);lo=F(0);hi=F(len(K),T)
        for req,label in [({T-2:0,T-1:0},0),({T-3:0,T-2:0,T-1:1},cert['negative_negative_positive_winner'])]:
            mass=sign_rank_masses(T,req);lo+=mass[label];hi-=sum(mass[j] for j in K if j!=label)
        # Full-network inputs only validate the reset, not accuracy estimation.
        rng=np.random.default_rng(49002);x=rng.standard_normal((N,T))
        wi,wh,wo=model;h=np.zeros((N,2))
        for t in range(T):h=np.maximum(h@wh.T+x[:,t,None]*wi,0)
        pred=np.argmax(h@wo.T,axis=1);last_two=np.all(x[:,-2:]<0,axis=1)
        last_three=(x[:,-3]<0)&(x[:,-2]<0)&(x[:,-1]>0)
        assert np.all(pred[last_two]==0)
        assert np.all(pred[last_three]==cert['negative_negative_positive_winner'])
        target=np.argsort(x,axis=1)[:,-2];acc=float(np.mean(pred==target))
        model_rows.append({'sigma':sigma,'labels':env['possible_labels'],'reset_certificate':cert,
                           'broad_interval':[float(lo),float(hi)],'exact_interval':[str(lo),str(hi)],
                           'diagnostic_accuracy':acc,'validation_inputs':N,
                           'reset_event_inputs':int(last_two.sum()+last_three.sum()),
                           'envelope_rational_arithmetic':env['rational_arithmetic_operations'],
                           'envelope_rational_comparisons':env['rational_comparisons']})
    checks['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    checks['models']=model_rows
    checks['compute']={'cpu_seconds':time.process_time()-started,'conditional_random_inputs':5*N,
                       'full_model_random_inputs':3*N,'full_model_recurrent_steps':3*N*8,
                       'analytic_model_inputs':0,'rational_scale_envelopes':9}
    out=Path(__file__).with_name('prefix_checks.json');out.write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps({'rank_partition_cases':n_exact,'max_integral_error':max(errors),
                     'max_mc_z':checks['conditional_validation']['max_abs_z'],
                     'models':[{k:r[k] for k in ('sigma','labels','broad_interval','diagnostic_accuracy')} for r in model_rows],
                     'compute':checks['compute']},indent=2))

if __name__=='__main__':main()
