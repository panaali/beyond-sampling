"""Exact ratio regions and deterministic two-suffix Gaussian rank integral."""
import math,time
from fractions import Fraction as F
import numpy as np
from scipy.special import ndtr,owens_t,roots_laguerre,roots_legendre
from .reset import coefficient
from .rank import pair_rank

def regions(wi,wh,wo):
    q=lambda x:F.from_float(float(x));wi=list(map(q,wi));wh=[list(map(q,row)) for row in wh];wo=[list(map(q,row)) for row in wo]
    k=coefficient(wi,wh)['kappa'];a=[sum(wh[i][j]*wi[j] for j in (0,1)) for i in (0,1)];lo=-k
    cuts=sorted(set([lo]+[x for x in (F(0),F(1))+tuple(-a[i]/wi[i] for i in (0,1)) if x>lo]));out=[];comparisons=0
    for left,right in zip(cuts,cuts[1:]+[None]):
        mid=(left+right)/2 if right is not None else left+1
        gates=tuple(a[i]+wi[i]*mid>0 for i in (0,1));lines=[(sum(row[i]*a[i] for i in (0,1) if gates[i]),sum(row[i]*wi[i] for i in (0,1) if gates[i])) for row in wo]
        roots=[left]
        for i in range(len(wo)):
            for j in range(i):
                comparisons+=1;u,v=lines[i][0]-lines[j][0],lines[i][1]-lines[j][1]
                if v:
                    root=-u/v
                    if root>left and (right is None or root<right):roots.append(root)
        roots=sorted(set(roots))
        for l,r in zip(roots,roots[1:]+[right]):
            z=(l+r)/2 if r is not None else l+1;values=[u+v*z for u,v in lines];label=max(range(len(wo)),key=lambda j:values[j])
            hidden=[max(F(0),a[i]+wi[i]*z) for i in (0,1)];direct=[sum(row[i]*hidden[i] for i in (0,1)) for row in wo];assert direct==values
            assert label==max(range(len(wo)),key=lambda j:direct[j])
            out.append(dict(lo=l,hi=r,label=label,gates=gates,other_suffix_above=(z>1 if label==len(wo)-2 else z<1)))
    return out,dict(pair_root_checks=comparisons,regions=len(out),rational_arithmetic=True)

def integrate(wi,wh,wo,order):
    start=time.process_time();cells,stats=regions(wi,wh,wo);T=len(wo);N=T-4;k=float(coefficient(wi,wh)['kappa']);q=math.atan(k)/(2*math.pi);r=.5-q
    if N<1:raise ValueError('This primitive requires T>=5')
    a,aw=roots_legendre(order);s,sw=roots_laguerre(order);rad=np.sqrt(2*s);value=0.;nodes=0;negative_clamps=0;mass=0.
    for cell in cells:
        if cell['label'] not in (T-2,T-1):raise ValueError('Non-suffix prediction in current two-suffix integral')
        lo=math.atan(float(cell['lo']));hi=math.pi/2 if cell['hi'] is None else math.atan(float(cell['hi']));angles=(lo+hi)/2+(hi-lo)/2*a
        if cell['label']==T-1 and lo<0:raise ValueError('Nonpositive selected suffix input unsupported')
        t=(np.cos(angles) if cell['label']==T-2 else np.sin(angles))[:,None]*rad[None,:];u=ndtr(t);sf=ndtr(-t);ow=owens_t(t,k);g0=.5*u+ow-q;g1=.5*sf-ow
        if float(np.min(g1)) < -1e-14:raise ArithmeticError('Materially negative Owen-T rank mass')
        negative_clamps+=int(np.sum(g1<0));g1=np.maximum(g1,0.)
        f=u**N*g0 if cell['other_suffix_above'] else N*sf*u**(N-1)*g0+u**N*g1
        value+=float(aw@f@sw)*(hi-lo)/(4*math.pi);mass+=float(np.sum(aw)*np.sum(sw))*r*(hi-lo)/(4*math.pi);nodes+=order*order
    stats.update(order=order,nodes=nodes,owens_t_calls=nodes,normal_cdf_calls=2*nodes,negative_clamps=negative_clamps,cpu_seconds=time.process_time()-start,event_mass_integrated=mass,event_mass_analytic=r*(.25+q),roundoff_certified=False,quadrature_bound=None)
    return value,stats
