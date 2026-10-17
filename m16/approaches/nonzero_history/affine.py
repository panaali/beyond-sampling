"""Weight-only deterministic-history suffix and analytic rank integration."""
from collections import defaultdict
from itertools import product
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr

ROOT2PI=np.sqrt(2*np.pi)


def prefix_mean(model,mode,stats):
    wi,wh,wo=model;m=len(wo)-2
    if mode=='zero':return np.zeros_like(wi)
    if mode not in ('plugin','exact2'):raise ValueError(mode)
    mu=np.zeros_like(wi);start=0
    if mode=='exact2' and m>=2:
        terms=[]
        for sign in (1.,-1.):
            c=wh@np.maximum(sign*wi,0)
            r=np.hypot(c,wi)
            value=np.empty_like(c);positive=c>=0
            value[positive]=r[positive]+c[positive]
            negative=~positive
            value[negative]=wi[negative]**2/(r[negative]-c[negative])
            terms.append(value)
        mu=(terms[0]+terms[1])/(2*ROOT2PI);start=2
        stats['mean_dense_dot_products']+=2*len(wi)**2
        stats['mean_hypot_calls']+=2*len(wi)
        stats['mean_scalar_arithmetic']+=12*len(wi)
    for _ in range(start,m):
        a=wh@mu;sig=abs(wi);positive=sig>0
        mu=np.maximum(a,0)
        z=a[positive]/sig[positive]
        mu[positive]=sig[positive]*np.exp(-z*z/2)/ROOT2PI+a[positive]*ndtr(z)
        stats['mean_dense_dot_products']+=len(wi)**2
        stats['mean_cdf_calls']+=int(positive.sum())
        stats['mean_exp_calls']+=int(positive.sum())
        stats['mean_scalar_arithmetic']+=8*len(wi)
    if not np.all(np.isfinite(mu)) or np.any(mu<0):
        raise ArithmeticError('mean closure invalid')
    return mu


def arrangement(model,mu,limit,stats):
    """Disjoint vertical regions labelled by first-index affine readout."""
    wi,wh,wo=model;H=len(wi);base=wh@mu
    edges=[-limit,limit]
    for a,b in zip(wi,base):
        if a!=0:
            root=-b/a;stats['first_gate_roots']+=1
            if -limit<root<limit:edges.append(root)
    edges=np.unique(edges);pieces=[]
    for left,right in zip(edges[:-1],edges[1:]):
        active=base+wi*((left+right)/2)>0
        cy=wh@(wi*active);cc=wh@(base*active)
        stats['affine_dense_dot_products']+=2*H*H
        for mask_tuple in product((False,True),repeat=H):
            mask=np.array(mask_tuple)
            zero=(cy==0)&(wi==0)&(cc==0)
            if np.any(mask&zero):continue  # zero gate belongs to inactive cell
            logits=np.stack((wo@(cy*mask),wo@(wi*mask),wo@(cc*mask)),axis=1)
            stats['affine_dense_dot_products']+=3*len(wo)*H
            gates=[np.array([cy[i],wi[i],cc[i]])*(1 if mask[i] else -1)
                   for i in range(H) if not zero[i]]
            for label in range(len(wo)):
                constraints=list(gates);reject=False
                for other in range(len(wo)):
                    if other==label:continue
                    margin=logits[label]-logits[other]
                    stats['readout_margins']+=1
                    if not np.any(margin):
                        if other<label:reject=True;break
                    else:constraints.append(margin)
                if reject:continue
                lo,hi=float(left),float(right);lower=[];upper=[]
                for raw in constraints:
                    norm=max(abs(raw));stats['plane_normalizations']+=1
                    if norm==0:continue
                    a,b,c=raw/norm
                    if b>0:lower.append(np.array([-a/b,-c/b]))
                    elif b<0:upper.append(np.array([-a/b,-c/b]))
                    elif a>0:lo=max(lo,-c/a)
                    elif a<0:hi=min(hi,-c/a)
                    elif c<0:reject=True;break
                if reject or hi<=lo:continue
                lines=lower+upper+[np.array([1.,0.])]
                knots=[lo,hi]
                for i in range(len(lines)):
                    for j in range(i):
                        stats['line_intersection_tests']+=1
                        slope=lines[i][0]-lines[j][0]
                        if slope!=0:
                            root=(lines[j][1]-lines[i][1])/slope
                            if lo<root<hi:knots.append(root)
                knots=np.unique(knots);stats['sort_input_items']+=len(knots)
                for yl,yr in zip(knots[:-1],knots[1:]):
                    mid=(yl+yr)/2
                    low=max(lower,key=lambda z:z[0]*mid+z[1]) if lower else np.array([0.,-np.inf])
                    high=min(upper,key=lambda z:z[0]*mid+z[1]) if upper else np.array([0.,np.inf])
                    stats['envelope_line_evaluations']+=len(lower)+len(upper)
                    if high[0]*mid+high[1]<=low[0]*mid+low[1]:continue
                    pieces.append((float(yl),float(yr),low,high,label))
    stats['vertical_pieces']+=len(pieces)
    return pieces


def cdf_mass(lo,hi,stats):
    # np.where eagerly evaluates both branches: four CDF arrays, not two.
    stats['normal_cdf_values']+=4*np.size(lo)
    stats['cdf_difference_arrays']+=np.size(lo)
    positive=lo>0
    return np.where(positive,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo))


def power_diff(qlo,qhi,mass,power,stats):
    """qhi^power-qlo^power from stable qhi-qlo (small integer power)."""
    total=np.zeros_like(mass)
    for k in range(power):total+=qhi**(power-1-k)*qlo**k
    stats['integer_power_products']+=2*power*np.size(mass)
    return mass*total


def rank_integrals(y,lo,hi,m,stats):
    """Mass, true first-suffix mass, true last-suffix mass over x interval."""
    mass=cdf_mass(lo,hi,stats)
    qy=ndtr(y);stats['normal_cdf_values']+=len(y)
    first=np.zeros_like(y);last=np.zeros_like(y)
    for below in (True,False):
        a=lo if below else np.maximum(lo,y)
        b=np.minimum(hi,y) if below else hi
        valid=b>a
        if not np.any(valid):continue
        aa,bb=a[valid],b[valid];mm=cdf_mass(aa,bb,stats)
        qa,qb=ndtr(aa),ndtr(bb);stats['normal_cdf_values']+=2*len(aa)
        top=power_diff(qa,qb,mm,m+1,stats)/(m+1)
        if below:
            first[valid]+=m*(1-qy[valid])*qy[valid]**(m-1)*mm
            last[valid]+=top
        else:
            first[valid]+=qy[valid]**m*mm
            last[valid]+=power_diff(qa,qb,mm,m,stats)-m*top
    earlier=(mass-first-last)/m
    stats['minimum_rank_interval_mass']=min(stats['minimum_rank_interval_mass'],float(min(first.min(),last.min(),earlier.min())))
    if min(first.min(),last.min(),earlier.min()) < -1e-12:
        raise ArithmeticError('negative scalar rank probability')
    return mass,first,last,earlier


def calculate(model,mode='exact2',order=16,limit=6.,normal_landmarks=False):
    start=time.process_time();stats=defaultdict(float)
    wi,wh,wo=model;T=len(wo);m=T-2
    if m<1:raise ValueError('at least three inputs required')
    mu=prefix_mean(model,mode,stats)
    pieces=arrangement(model,mu,limit,stats)
    if normal_landmarks:
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
        pieces=refined
        stats['normal_landmark_panels']=len(pieces)
    nodes,weights=leggauss(order);value=0.;partition=0.
    stats['legendre_rules_constructed']+=1
    stats['legendre_rule_order']=order
    for yl,yr,low,high,label in pieces:
        y=(yl+yr)/2+(yr-yl)/2*nodes
        lo=low[0]*y+low[1];hi=high[0]*y+high[1]
        mass,first,last,earlier=rank_integrals(y,lo,hi,m,stats)
        truth=first if label==m else last if label==m+1 else earlier
        density=(yr-yl)/2*weights*np.exp(-y*y/2)/ROOT2PI
        value+=float(density@truth);partition+=float(density@mass)
        stats['quadrature_nodes']+=len(y)
        stats['normal_density_exponentials']+=len(y)
    omission=float(2*ndtr(-limit))
    stats.update(mode=mode,order=order,normal_landmarks=normal_landmarks,prefix_mean=mu.tolist(),estimate=value,
                 partition_mass=partition,partition_defect=partition-(1-omission),
                 omitted_y_mass_bound=omission,cpu_seconds=time.process_time()-start,
                 total_error_certified=False)
    return value,dict(stats)
