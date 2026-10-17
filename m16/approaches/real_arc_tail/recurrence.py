"""Isolated real even-power variant of T55 matching-parity moments."""
from pathlib import Path
import importlib.util
import math
import sys
import numpy as np
from scipy.special import beta,betainc,hyp2f1

SOURCE=Path('/workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals')
sys.path.insert(0,str(SOURCE))
spec=importlib.util.spec_from_file_location('arc_tail_baseline',SOURCE/'moments.py')
baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)

def add(stats,**values):
    if stats is not None:
        for k,v in values.items(): stats[k]=stats.get(k,0)+v

def even_arc(a,b,theta,degree,stats=None):
    if degree<0 or degree%2: raise ValueError('nonnegative even degree required')
    out=[float(theta)]
    add(stats,arc_chains=1,arc_values=degree//2+1)
    if not degree:return out
    s,c=math.sin(theta),math.cos(theta)
    f=a*c+b*s;g=-a*s+b*c;r2=a*a+b*b
    fpower=f;apower=a;f2=f*f;a2=a*a
    add(stats,arc_trig=2,arc_scalar_arithmetic=12)
    for n in range(2,degree+1,2):
        endpoint=fpower*g-apower*b
        out.append(((n-1)*r2*out[-1]-endpoint)/n)
        add(stats,arc_scalar_arithmetic=7,arc_recurrence_steps=1)
        if n<degree:
            fpower*=f2;apower*=a2
            add(stats,arc_scalar_arithmetic=2)
    return out

def stable_arc(a,b,theta,degree,stats=None):
    if degree<0 or degree%2:raise ValueError('nonnegative even degree required')
    out=[float(theta)]
    add(stats,arc_chains=1,arc_values=degree//2+1)
    if not degree:return out
    s,c=math.sin(theta/2),math.cos(theta/2)
    df=2*s*(-a*s+b*c);dg=-2*s*(a*c+b*s)
    f=a+df;r2=a*a+b*b
    add(stats,arc_trig=2,arc_scalar_arithmetic=17)
    for n in range(2,degree+1,2):
        power_sum=sum(f**(n-2-j)*a**j for j in range(n-1))
        endpoint=f**(n-1)*dg+b*df*power_sum
        out.append(((n-1)*r2*out[-1]-endpoint)/n)
        add(stats,arc_integer_power_calls=2*(n-1)+1,arc_recurrence_steps=1,arc_scalar_arithmetic=2*(n-1)+8)
    return out

def positive_arc(a,b,theta,degree,stats=None,endpoint=False):
    if degree<0 or degree%2 or not 0<=theta<=math.pi:raise ValueError('even degree and minor arc required')
    out=[float(theta)]
    add(stats,arc_chains=1,arc_values=degree//2+1)
    if not degree:return out
    s,c=math.sin(theta/2),math.cos(theta/2)
    A=a*c+b*s;B=-a*s+b*c;z=s*s
    add(stats,arc_trig=2,arc_scalar_arithmetic=8)
    for n in range(2,degree+1,2):
        value=0.
        for j in range(0,n+1,2):
            p=(j+1)/2;q=(n-j+1)/2
            if endpoint and theta<=math.pi/2:
                mass=s**(j+1)/p*float(hyp2f1(p,1-q,p+1,z))
                add(stats,arc_hyp2f1_calls=1,arc_integer_power_calls=1,arc_scalar_arithmetic=4)
            elif endpoint:
                mass=float(beta(p,q)*(1-betainc(q,p,c*c)))
                add(stats,arc_beta_calls=1,arc_betainc_calls=1,arc_scalar_arithmetic=3)
            else:
                mass=float(beta(p,q)*betainc(p,q,z))
                add(stats,arc_beta_calls=1,arc_betainc_calls=1,arc_scalar_arithmetic=1)
            value+=math.comb(n,j)*A**(n-j)*B**j*mass
            add(stats,arc_integer_power_calls=2,arc_binomial_calls=1,arc_scalar_arithmetic=8)
        out.append(value)
    return out

def polygon_moment(vertices,degree,stats=None,arc=even_arc):
    if degree<1 or degree%2!=1: raise ValueError('positive odd degree required')
    vertices=np.array([baseline.unit(v) for v in vertices])
    if len(vertices)<3 or np.min(vertices[:,0]) < -1e-14: raise ValueError('invalid positive-hemisphere polygon')
    center=vertices.sum(axis=0);boundary=np.zeros((degree+1)//2)
    for u,v in zip(vertices,np.roll(vertices,-1,axis=0)):
        cross=np.cross(u,v);length=np.linalg.norm(cross)
        if not length:raise ValueError('coincident or antipodal consecutive rays')
        inward=cross/length;side=inward@center
        if not side:raise ValueError('degenerate polygon')
        inward*=np.sign(side);outward0=-inward[0]
        theta=np.arctan2(length,u@v);tangent=np.cross(cross/length,u)
        arcs=arc(float(u[0]),float(tangent[0]),float(theta),degree-1,stats)
        for i,value in enumerate(arcs):boundary[i]+=outward0*value
    value=0.
    for i,p in enumerate(range(1,degree+1,2)):
        previous=(p-1)*value if p>1 else 0.
        value=(previous-boundary[i])/(p+1)
    add(stats,polygon_calls=1,polygon_edges=len(vertices),polygon_recurrence_steps=(degree+1)//2)
    return value

# Copy only the module namespace, preserving the original implementation and
# guard/fallback functions while binding their global moment lookup locally.
variant_spec=importlib.util.spec_from_file_location('arc_tail_variant',SOURCE/'moments.py')
variant=importlib.util.module_from_spec(variant_spec);variant_spec.loader.exec_module(variant)
variant.polygon_moment=polygon_moment

def module_for(arc):
    sp=importlib.util.spec_from_file_location('arc_tail_'+arc.__name__,SOURCE/'moments.py')
    mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
    mod.polygon_moment=lambda vs,d,stats=None:polygon_moment(vs,d,stats,arc=arc)
    return mod

stable=module_for(stable_arc)
positive=module_for(positive_arc)

def endpoint_arc(a,b,theta,degree,stats=None):
    return positive_arc(a,b,theta,degree,stats,endpoint=True)

endpoint=module_for(endpoint_arc)
