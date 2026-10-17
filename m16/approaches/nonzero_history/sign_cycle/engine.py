"""Two sign-conditioned mean histories with exact-degree scalar rank weights."""
from collections import defaultdict
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from ranks import scalar_integrals

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from affine import arrangement,ROOT2PI
sys.path.pop(0)


def means(model,stats):
    wi,wh,wo=model;m=len(wo)-2
    result={}
    for sign in (1,-1):
        c=wh@np.maximum(sign*wi,0);r=np.hypot(c,wi)
        mu=np.empty_like(c);positive=c>=0
        mu[positive]=(r[positive]+c[positive])/ROOT2PI
        mu[~positive]=wi[~positive]**2/((r[~positive]-c[~positive])*ROOT2PI)
        stats['mean_scalar_products']+=len(wi)**2
        stats['mean_hypot_calls']+=len(wi)
        for _ in range(2,m):
            a=wh@mu;sig=abs(wi);valid=sig>0
            mu=np.maximum(a,0);z=a[valid]/sig[valid]
            mu[valid]=sig[valid]*np.exp(-z*z/2)/ROOT2PI+a[valid]*ndtr(z)
            stats['mean_scalar_products']+=len(wi)**2
            stats['mean_cdf_calls']+=int(valid.sum());stats['mean_exp_calls']+=int(valid.sum())
        if not np.all(np.isfinite(mu)) or np.any(mu<0):raise ArithmeticError('conditional mean invalid')
        result[sign]=mu
    return result


def landmark_pieces(pieces,stats):
    refined=[]
    for yl,yr,low,high,label in pieces:
        cuts=[yl,yr]
        for level in range(-8,9):
            if yl<level<yr:cuts.append(float(level))
            for line in (low,high):
                stats['normal_landmark_intersection_tests']+=1
                if line[0]!=0:
                    root=(level-line[1])/line[0]
                    if yl<root<yr:cuts.append(float(root))
        cuts=np.unique(cuts);stats['sort_input_items']+=len(cuts)
        refined.extend((a,b,low,high,label) for a,b in zip(cuts[:-1],cuts[1:]))
    stats['normal_landmark_panels']+=len(refined)
    return refined


def calculate(model,shuffled=False,order=16,limit=6.,history='conditional'):
    start=time.process_time();stats=defaultdict(float);m=len(model[2])-2
    if m not in (2,3):raise ValueError('current exact-degree protocol is T4/T5')
    if history=='conditional':mu=means(model,stats)
    elif history=='zero':mu={s:np.zeros_like(model[0]) for s in (1,-1)}
    else:raise ValueError(history)
    nodes,weights=leggauss(order)
    stats['legendre_rules_constructed']+=1;stats['legendre_rule_order']=order
    signs=[]
    for sign in (1,-1):
        state=mu[-sign if shuffled else sign]
        pieces=landmark_pieces(arrangement(model,state,limit,stats),stats)
        value=0.;partition=0.
        for yl,yr,lower,upper,label in pieces:
            y=(yl+yr)/2+(yr-yl)/2*nodes
            lo=lower[0]*y+lower[1];hi=upper[0]*y+upper[1]
            mass,first,last,special,regular=scalar_integrals(y,lo,hi,m,sign,stats)
            truth=special if label==0 else first if label==m else last if label==m+1 else regular
            weight=weights*(yr-yl)/2*np.exp(-y*y/2)/ROOT2PI
            value+=float(weight@truth);partition+=float(weight@mass)
            stats['outer_normal_density_exponentials']+=len(y)
            stats['outer_quadrature_nodes']+=len(y)
        signs.append(dict(sign=sign,hidden_mean=state.tolist(),estimate=value,partition_mass=partition))
    value=sum(row['estimate'] for row in signs)/2
    return value,dict(order=order,shuffled=shuffled,history=history,signs=signs,
        omitted_y_mass_bound=float(2*ndtr(-limit)),cpu_seconds=time.process_time()-start,
        work=dict(stats),total_error_certified=False,portable_flops_certified=False,
        affine_source_sha256=hashlib.sha256((HERE.parent/'affine.py').read_bytes()).hexdigest())
