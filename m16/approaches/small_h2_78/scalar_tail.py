"""One-input scalar elimination after a certified reset.

The only numerical quadrature here integrates known polynomials of bounded
integer degree, not model-input samples. Gate/readout geometry is float and is
not an interval-arithmetic certificate.
"""
from functools import lru_cache
import numpy as np
from scipy.special import ndtr
from prefix import cdf_probability
from envelope import envelope


@lru_cache(maxsize=None)
def polynomial_rule(degree):
    return np.polynomial.legendre.leggauss((degree+2)//2)


def rank_integral(m,target,known_cdfs,lo,hi):
    """Integral of prefix-marginalized correctness over last-input CDF [lo,hi]."""
    cuts=sorted({lo,hi,*[v for v in known_cdfs if lo<v<hi],*([.5] if lo<.5<hi else [])})
    nodes,weights=polynomial_rule(m+2)
    result=0.;calls=0
    for a,b in zip(cuts,cuts[1:]):
        if b<=a:continue
        vals=(nodes+1)*((b-a)/2)+a
        p=np.array([float(cdf_probability(m,target,[*known_cdfs,v])) for v in vals])
        result+=(b-a)/2*float(weights@p);calls+=len(nodes)
    return result,calls


def interior(lo,hi):
    if np.isneginf(lo) and np.isposinf(hi):return 0.
    if np.isneginf(lo):return hi-max(1.,abs(hi))
    if np.isposinf(hi):return lo+max(1.,abs(lo))
    return lo+(hi-lo)/2


@lru_cache(maxsize=32)
def readout_boundaries(key):
    output=np.asarray(key,dtype=float).reshape(-1,2)
    topology=envelope(output)
    winners=[c['winner'] for c in topology['cells']]
    pairs=tuple(sorted({(a,b) for a,b in zip(winners,winners[1:]) if a!=b}))
    return pairs,topology


def output_intervals(model,hidden,lower=-np.inf,upper=np.inf,use_envelope=True):
    wi,wh,wo=(np.asarray(w,dtype=float) for w in model)
    wi=wi.reshape(-1);offset=wh@hidden;T=len(wo)
    stats={'gate_cells':0,'logit_pair_roots':0,'readout_cells':0,
           'last_step_matrix_products':1,'scalar_gate_products':0}
    if use_envelope and len(wi)==2:
        before=readout_boundaries.cache_info().misses
        pairs,topology=readout_boundaries(tuple(wo.ravel()))
        stats['envelope_builds']=readout_boundaries.cache_info().misses-before
        stats['rational_arithmetic_operations']=stats['envelope_builds']*topology['rational_arithmetic_operations']
        stats['rational_comparisons']=stats['envelope_builds']*topology['rational_comparisons']
    else:
        pairs=tuple((i,j) for i in range(T) for j in range(i))
    roots=[-offset[i]/wi[i] for i in range(len(wi)) if wi[i]!=0]
    cuts=sorted({lower,upper,*[r for r in roots if lower<r<upper]})
    result=[]
    for a,b in zip(cuts,cuts[1:]):
        active=offset+wi*interior(a,b)>0
        alpha=wo@(offset*active);beta=wo@(wi*active)
        stats['gate_cells']+=1;stats['scalar_gate_products']+=len(wi)
        splits={a,b}
        for i,j in pairs:
            d=beta[i]-beta[j]
            if d!=0:
                r=(alpha[j]-alpha[i])/d
                if a<r<b:splits.add(r);stats['logit_pair_roots']+=1
        splits=sorted(splits)
        for c,d in zip(splits,splits[1:]):
            # First-index argmax handles positive-volume duplicate logits/dead states.
            winner=int(np.argmax(alpha+beta*interior(c,d)))
            result.append((c,d,winner));stats['readout_cells']+=1
    return result,stats


def conditional_tail(model,m,known_suffix):
    """Joint correctness and valid last-reset suffix extension probability.

    Last reset has m unrestricted inputs before its negative pair. known_suffix
    follows that pair and excludes the final input. Integrates the final input
    against its unconditioned standard Gaussian density. Nonempty known_suffix
    must begin positive and contain no consecutive negatives. Pair probability
    1/4 and known-suffix probability density are NOT included.
    """
    wi,wh,wo=(np.asarray(w,dtype=float) for w in model);wi=wi.reshape(-1)
    known=np.asarray(known_suffix,dtype=float)
    if len(set(known))!=len(known):
        raise ValueError('fixed suffix ties are outside the continuous-input conditioning convention')
    T=len(wo)
    if T!=m+2+len(known)+1:raise ValueError('inconsistent total sequence length')
    if len(known) and (known[0]<=0 or np.any((known[:-1]<0)&(known[1:]<0))):
        raise ValueError('invalid last-reset suffix')
    h=np.zeros(len(wi))
    for x in known:h=np.maximum(wh@h+wi*x,0)
    lower=0. if not len(known) or known[-1]<0 else -np.inf
    intervals,stats=output_intervals((wi,wh,wo),h,lower=lower)
    known_cdfs=list(ndtr(known));result=0.;calls=0
    for a,b,target in intervals:
        term,n=rank_integral(m,target,known_cdfs,float(ndtr(a)),float(ndtr(b)))
        result+=term;calls+=n
    stats.update(polynomial_probability_calls=calls,prefix_recurrent_steps=len(known),
                 valid_last_input_mass=float(1-ndtr(lower)))
    if result < -1e-12 or result > stats['valid_last_input_mass']+1e-12:
        raise ArithmeticError('probability outside event bound')
    return result,stats
