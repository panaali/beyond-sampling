"""Isolated real even-power variant of T55 matching-parity moments."""
from pathlib import Path
import importlib.util
import math
import sys
import numpy as np

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

def polygon_moment(vertices,degree,stats=None):
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
        arcs=even_arc(float(u[0]),float(tangent[0]),float(theta),degree-1,stats)
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
