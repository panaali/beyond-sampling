"""Independent rectangle order, convolution and deterministic event checks."""
import itertools,json,math,time
from pathlib import Path
from scipy.integrate import quad
from scipy.special import ndtr
from .rank import pair_rank,prefix_counts,phi
from .reset import masses

def direct(k,t):
    calls=0;out=[]
    cuts=sorted(set([-8.,0.]+[v for v in (t,-k*t) if math.isfinite(v) and -8<v<0]))
    for count in range(3):
        def f(y):
            nonlocal calls;calls+=1
            b=math.inf if k==0 else (0. if math.isinf(k) else -y/k)
            lower=float(ndtr(min(b,t)));upper=max(0.,float(ndtr(b))-float(ndtr(t)))
            return phi(y)*(lower if count==int(y>t) else upper if count==1+int(y>t) else 0.)
        out.append(sum(quad(f,a,b,epsabs=2e-13,epsrel=2e-13)[0] for a,b in zip(cuts,cuts[1:])))
    return out,calls

def main():
    start=time.process_time();rows=[];maxerr=0.;calls=0;eventchecks=0
    for k,t in itertools.product([0.,.1,1.,3.,math.inf],[-3.,-.2,0.,.2,3.]):
        g,s=pair_rank(k,t);ref,n=direct(k,t);calls+=n+s['callbacks'];error=max(abs(a-b) for a,b in zip(g,ref));maxerr=max(maxerr,error)
        assert error<2e-11,(k,t,g,ref)
        assert min(g)>=0 and abs(sum(g)-masses(k,2)['pair_reset_mass'])<2e-11
        for m in (0,1,4):
            got,_=prefix_counts(k,t,m);u=float(ndtr(t));v=float(ndtr(-t));ind=[0.]*(m+3)
            # Enumerate Bernoulli rank bits rather than reuse binomial formula.
            for bits in itertools.product((0,1),repeat=m):
                weight=math.prod(v if bit else u for bit in bits)
                for l in range(3):ind[sum(bits)+l]+=weight*g[l]
            assert max(abs(a-b) for a,b in zip(got,ind))<1e-14
        rows.append(dict(kappa=str(k),threshold=t,max_rectangle_difference=error,masses=g,production_stats=s))
    # Last pair ending3 in a four-input sequence: old input unrestricted;
    # conditioned E(x2,x3), survival of next pair iff x4>0.
    for k in (0.,.1,1.,3.):
        for old,x,y,future in itertools.product((-2.,-.1,.1,2.),repeat=4):
            e=y<=-k*max(x,0.);last=e and not(future<=-k*max(y,0.))
            assert last==(e and future>0)
            eventchecks+=1
    out=dict(complete=True,cases=rows,max_rectangle_difference=maxerr,scalar_callbacks=calls,event_logic_checks=eventchecks,random_model_inputs=0,cpu_seconds=time.process_time()-start,roundoff_certified=False)
    assert out['cpu_seconds']<18
    Path(__file__).with_name('rank_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))
if __name__=='__main__':main()
