"""Rank-conditioned backward collocation, with explicitly approximate q rules."""
from collections import defaultdict
import json
import pathlib
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import roots_genlaguerre,ndtr,gammaln,betainc
from numpy.polynomial.legendre import leggauss
from backward import load,invariant_box,possible_labels,theta_quantile,target_kernel_grid,make_interpolator,observe_interpolation

HERE=pathlib.Path(__file__).resolve().parent


class RankRules:
    def __init__(self,n,stats):
        self.n=n;self.stats=stats;self.rules={}

    def mass(self,b,k,m):
        """Return both K0(exactly one exceedance) and K1(no exceedances)."""
        b=np.asarray(b)
        if k not in self.rules:
            x,w=roots_genlaguerre(self.n,k/2-1)
            w=w/np.exp(gammaln(k/2))
            self.rules[k]=(np.sqrt(2*x),w)
            self.stats['rank_rules_generated']+=1
            self.stats['rank_rule_generation_nodes']+=self.n
        r,w=self.rules[k]
        z=b[...,None]*r
        cdf=ndtr(z);sf=ndtr(-z)
        q1=np.sum(cdf**m*w,axis=-1)
        q0=np.sum(m*cdf**(m-1)*sf*w,axis=-1) if m else np.zeros_like(q1)
        self.stats['rank_probability_queries']+=b.size
        self.stats['rank_quadrature_nodes']+=b.size*self.n
        return np.stack((q0,q1),axis=-1)


def conditional_rules(b,T,nangle,rank,stats):
    nodes,weights=leggauss(nangle);nodes=(nodes+1)/2;weights=weights/2
    result={}
    for k in range(1,T-2):
        threshold=.5+np.sign(b)*.5*betainc(.5,k/2,b*b/(1+b*b))
        branches=[]
        denominator=np.zeros((len(b),2))
        for exceeds,lo,hi in ((False,np.zeros_like(b),threshold),(True,threshold,np.ones_like(b))):
            probs=lo[:,None]+(hi-lo)[:,None]*nodes
            sine,cosine=theta_quantile(probs,k)
            qnext=rank.mass(b[:,None]*cosine,k+1,T-k-1)
            raw=np.zeros_like(qnext)
            if exceeds:
                raw[:,:,0]=qnext[:,:,1]
            else:
                raw=qnext.copy()
            raw*=((hi-lo)[:,None]*weights)[:,:,None]
            denominator+=raw.sum(axis=1)
            branches.append((exceeds,sine,cosine,raw))
        qprev=rank.mass(b,k,T-k)
        stats['max_unnormalized_rank_defect']=max(stats['max_unnormalized_rank_defect'],float(np.max(abs(denominator-qprev))))
        stats['max_relative_rank_defect']=max(stats['max_relative_rank_defect'],float(np.max(abs(denominator-qprev)/qprev)))
        if np.any(denominator<=0): raise ArithmeticError('Conditional rank event has zero numerical mass')
        normalized=[]
        total=np.zeros_like(denominator)
        for exceeds,sine,cosine,raw in branches:
            weight=raw/denominator[:,None,:]
            total+=weight.sum(axis=1)
            normalized.append((exceeds,sine,cosine,weight))
        stats['max_normalized_row_defect']=max(stats['max_normalized_row_defect'],float(np.max(abs(total-1))))
        result[k]=normalized
    return result


def transport(model,j,grid,terminal,rules,stats,method="linear"):
    wi,wh,wo=model;T=len(wo)
    states=np.column_stack([a.ravel() for a in np.meshgrid(*grid,indexing='ij')])
    repeats=len(grid[0])*len(grid[1]);shape=terminal.shape
    values=terminal
    npre=T-2 if j<T-2 else T-3
    for t in reversed(range(npre)):
        interp=make_interpolator(grid,values,method)
        if t==j:
            coords=np.column_stack((np.maximum(states[:,:2]@wh.T+states[:,2,None]*wi,0),states[:,2]))
            values=interp(coords).reshape(shape);stats['interpolation_queries']+=len(coords)
            observe_interpolation(values,stats)
            continue
        k=1+t-int(j<t)
        updated=np.zeros((len(states),2))
        for exceeds,sine,cosine,weights in rules[k]:
            sin=np.tile(sine,(repeats,1));cos=np.tile(cosine,(repeats,1))
            weight=np.tile(weights,(repeats,1,1))
            h=np.maximum((states[:,:2]@wh.T)[:,None,:]*cos[:,:,None]+wi*sin[:,:,None],0)
            coords=np.concatenate((h,(states[:,2,None]*cos)[:,:,None]),axis=2).reshape(-1,3)
            evaluated=interp(coords).reshape(len(states),sine.shape[1],2)
            observe_interpolation(evaluated,stats)
            if exceeds:
                updated[:,0]+=np.sum(evaluated[:,:,1]*weight[:,:,0],axis=1)
            else:
                updated+=np.sum(evaluated*weight,axis=1)
            stats['interpolation_queries']+=len(coords);stats['angle_nodes']+=len(coords)
        values=updated.reshape(shape)
    initial=make_interpolator(grid,values,method)(np.array([[0.,0.,-1.],[0.,0.,1.]]))[:,0]
    stats['interpolation_queries']+=2
    qminus=2*(T+1)/(T*2**T);qplus=2/T-qminus
    return float((initial[0]*qminus+initial[1]*qplus)/2),initial.tolist()


def main():
    start=time.process_time();deadline=start+110
    model=load(pathlib.Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth'))
    T=len(model[2]);bound=invariant_box(*model[:2]);labels=possible_labels(model[2])
    output={'classification':'deterministic rank-conditioned collocation; no certified error bound',
        'uses_previously_charged_terminal_caches':True,'rows':[],'model_forward_inputs':0}
    for n in (5,7):
        grid=(np.linspace(0,bound[0],n),np.linspace(0,bound[1],n),np.linspace(-1,1,n))
        data=np.load(HERE/f'terminal_grid{n}.npz')
        terminal={j:data[f'j{j}'] for j in labels}
        constant=(model[0],model[1],np.zeros_like(model[2]))
        control_stats=defaultdict(float)
        actual_constant=target_kernel_grid(constant,0,grid,control_stats,deadline,constant=True)
        for nq in (32,64):
            clock=time.process_time();stats=defaultdict(float);rank=RankRules(nq,stats)
            qterminal=rank.mass(grid[2],T-2,2)
            rules=conditional_rules(grid[2],T,16,rank,stats)
            contributions={};initials={};vmin=1.;vmax=0.
            for j in labels:
                W=terminal[j]/qterminal[None,None,:,:]
                vmin=min(vmin,float(W.min()));vmax=max(vmax,float(W.max()))
                contributions[j],initials[j]=transport(model,j,grid,W,rules,stats)
            ones=transport(model,0,grid,np.ones_like(terminal[0]),rules,stats)
            control=transport(model,0,grid,actual_constant/qterminal[None,None,:,:],rules,stats)
            row={'grid':n,'rank_quadrature_nodes':nq,'angles_per_rank_interval':16,
                'accuracy':sum(contributions.values()),'contributions':contributions,'initial_sign_values':initials,
                'terminal_W_min':vmin,'terminal_W_max':vmax,
                'constant_W_control':ones[0],'constant_W_error':ones[0]-1/T,
                'constant_readout_control':control[0],'constant_readout_error':control[0]-1/T,
                'work':dict(stats),'cpu_seconds':time.process_time()-clock,
                'extra_control_terminal_work':dict(control_stats),'extra_control_kernel_reused':nq==64}
            output['rows'].append(row)
        data.close()
    output['cpu_seconds_excluding_imports']=time.process_time()-start
    (HERE/'doob_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))

if __name__=='__main__':main()
