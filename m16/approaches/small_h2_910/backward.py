"""Deterministic backward collocation; numerical approximation, not a certificate."""
from collections import defaultdict
import hashlib
import json
import pathlib
import os
import sys
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve
from scipy.special import betainc,betaincinv
from numpy.polynomial.legendre import leggauss

HERE=pathlib.Path(__file__).resolve().parent
FUND=pathlib.Path(os.environ.get('FUNDAMENTALS_DIR',str(HERE.parent/'feynman_fundamentals')))
if not (FUND/'target_tail.py').exists():
    FUND=pathlib.Path('/workspace/clean-neurons/m16/approaches/feynman_fundamentals')
sys.path.insert(0,str(FUND))
from target_tail import target_probability


def load(path):
    import torch
    d=torch.load(path,map_location="cpu",weights_only=True)
    return tuple(d[key].numpy().astype(float).reshape(shape) for key,shape in (
        ("rnn.weight_ih_l0",(2,)),("rnn.weight_hh_l0",(2,2)),("linear.weight",(10,2))))


def possible_labels(wo):
    # Strict on BOTH coordinates makes domination valid even on hidden axes.
    # Origin goes to index0 regardless of domination.
    return [j for j,w in enumerate(wo) if j==0 or not any(np.all(v>w) for v in wo)]


def invariant_box(wi,wh):
    positive=np.maximum(wh,0)
    if max(abs(np.linalg.eigvals(positive)))>=1:
        raise ValueError('This box proof requires positive-part spectral radius<1')
    bound=np.linalg.solve(np.eye(2)-positive,np.abs(wi))
    if not np.all(bound>0): raise ValueError('Need strictly positive box sides')
    bound=bound*(1+1e-12)
    if np.max(positive@bound+abs(wi)-bound)>1e-13:
        raise ArithmeticError('Invariant box proof failed')
    return bound


def theta_quantile(v,k):
    v=np.asarray(v)
    sine=np.sign(v-.5)*np.sqrt(betaincinv(.5,k/2,abs(2*v-1)))
    return sine,np.sqrt(np.maximum(0.,1-sine*sine))


def target_kernel_grid(model,j,grid,stats,deadline,constant=False):
    a1,a2,b=grid
    shape=(len(a1),len(a2),len(b),2)
    values=np.zeros(shape)
    for ib,bval in enumerate(b):
        for i,x in enumerate(a1 if not constant else a1[:1]):
            for l,y in enumerate(a2 if not constant else a2[:1]):
                for count in (0,1):
                    if time.process_time()>deadline:
                        raise TimeoutError('Cycle CPU deadline before terminal call')
                    stats["terminal_attempts"]+=1
                    p,work=target_probability(model,np.array([x,y]),float(bval),count,j)
                    if not -1e-10<=p<=1+1e-10: raise ArithmeticError('Terminal probability range')
                    values[i,l,ib,count]=p
                    stats['terminal_calls']+=1
                    for key,val in work.items():
                        if isinstance(val,(int,float)) and key not in ('partition_mass','prefix_dimension','target_index'):
                            stats['terminal_'+key]+=val
                    stats['max_partition_error']=max(stats['max_partition_error'],abs(work.get('partition_mass',1)-1))
        if constant:
            values[:,:,ib,:]=values[0,0,ib,:]
    return values


def make_interpolator(grid,values,method):
    extra={"solver":spsolve} if method=="cubic" else {}
    return RegularGridInterpolator(grid,values,bounds_error=True,method=method,**extra)


def observe_interpolation(values,stats):
    stats["minimum_interpolated_value"]=min(stats["minimum_interpolated_value"],float(np.min(values)))
    stats["maximum_interpolated_value"]=max(stats["maximum_interpolated_value"],float(np.max(values)))
    stats["queries_below_minus_1e_10"]+=int(np.count_nonzero(values < -1e-10))
    stats["queries_above_one_plus_1e_10"]+=int(np.count_nonzero(values > 1+1e-10))


def integrate_backwards(model,j,grid,terminal,nangle,stats,method="linear"):
    wi,wh,wo=model;T=len(wo)
    axes=np.meshgrid(*grid,indexing='ij')
    states=np.column_stack([a.ravel() for a in axes])
    shape=terminal.shape
    npre=T-2 if j<T-2 else T-3
    nodes,weights=leggauss(nangle)
    nodes=(nodes+1)/2;weights=weights/2
    values=terminal
    for t in reversed(range(npre)):
        interpolator=make_interpolator(grid,values,method)
        if t==j:
            transformed=np.column_stack((np.maximum(states[:,:2]@wh.T+states[:,2,None]*wi,0),states[:,2]))
            values=interpolator(transformed).reshape(shape)
            observe_interpolation(values,stats)
            stats['interpolation_queries']+=len(transformed)
            continue
        k=1+t-int(j<t)
        b=states[:,2]
        threshold=np.sign(b)*.5*betainc(.5,k/2,b*b/(1+b*b))+.5
        # CDF split handles the capped-count discontinuity exactly.
        nextval=np.zeros((len(states),2))
        for exceeds,lo,hi in ((False,np.zeros_like(b),threshold),(True,threshold,np.ones_like(b))):
            probs=lo[:,None]+(hi-lo)[:,None]*nodes
            sine,cosine=theta_quantile(probs,k)
            hidden=np.maximum((states[:,:2]@wh.T)[:,None,:]*cosine[:,:,None]+wi*sine[:,:,None],0)
            coords=np.concatenate((hidden,(b[:,None]*cosine)[:,:,None]),axis=2).reshape(-1,3)
            # No clipping: the positive-part invariant-box proof must suffice.
            evaluated=interpolator(coords).reshape(len(states),nangle,2)
            observe_interpolation(evaluated,stats)
            integrated=np.einsum('nqk,q->nk',evaluated,weights)*(hi-lo)[:,None]
            if exceeds:
                nextval[:,0]+=integrated[:,1]
            else:
                nextval+=integrated
            stats['interpolation_queries']+=len(coords)
            stats['angle_nodes']+=len(coords)
        values=nextval.reshape(shape)
    initial=make_interpolator(grid,values,method)(np.array([[0.,0.,-1.],[0.,0.,1.]]))[:,0]
    stats['interpolation_queries']+=2
    return float(initial.mean()),initial.tolist()


def main():
    start=time.process_time();deadline=start+110
    model=load(pathlib.Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth'))
    bound=invariant_box(*model[:2]);labels=possible_labels(model[2])
    output={'classification':'deterministic numerical backward collocation; no error certificate',
        'box':bound.tolist(),'possible_labels':labels,'target_kernel_sha256':hashlib.sha256((FUND/'target_tail.py').read_bytes()).hexdigest(),
        'rows':[],'truncated':False}
    for n in (5,7):
        grid=(np.linspace(0,bound[0],n),np.linspace(0,bound[1],n),np.linspace(-1,1,n))
        cache={};kernel_work=defaultdict(float);clock=time.process_time()
        try:
            for j in labels:
                cache[j]=target_kernel_grid(model,j,grid,kernel_work,deadline)
            kernel_cpu=time.process_time()-clock
            np.savez(HERE/f'terminal_grid{n}.npz',**{f'j{j}':v for j,v in cache.items()})
            for nang in ((8,) if n==5 else (8,16)):
                stats=defaultdict(float);clock=time.process_time();contributions={};initials={}
                for j in labels:
                    contributions[j],initials[j]=integrate_backwards(model,j,grid,cache[j],nang,stats)
                output['rows'].append({'grid':n,'angles_per_rank_interval':nang,'accuracy':sum(contributions.values()),
                    'contributions':contributions,'initial_sign_values':initials,
                    'kernel_work':dict(kernel_work),'kernel_cpu':kernel_cpu,'transport_work':dict(stats),
                    'transport_cpu':time.process_time()-clock,'kernel_reused':n==7 and nang==16})
            # Constant readout: terminal is independent of hidden state, so
            # evaluate only b/count grid and broadcast, charging those calls.
            constant=(model[0],model[1],np.zeros_like(model[2]))
            sanitywork=defaultdict(float)
            values=target_kernel_grid(constant,0,grid,sanitywork,deadline,constant=True)
            p,signs=integrate_backwards(constant,0,grid,values,16,sanitywork)
            output['rows'].append({'sanity':'constant_class0','grid':n,'accuracy':p,
                'exact_accuracy':.1,'error':p-.1,'work':dict(sanitywork)})
        except (ArithmeticError,ValueError,TimeoutError) as error:
            output['rows'].append({'grid':n,'failure':str(error),'partial_kernel_work':dict(kernel_work)})
            output['truncated']=True
            break
        (HERE/'backward_results.json').write_text(json.dumps(output,indent=2)+'\n')
    output['cpu_seconds_excluding_imports']=time.process_time()-start
    (HERE/'backward_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))

if __name__=='__main__':main()
