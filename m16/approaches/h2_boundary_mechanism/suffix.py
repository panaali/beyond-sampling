"""Two-input H2 boundary families, exact radial rank weights, no fitted advice."""
import math,time,sys
import numpy as np
from scipy.special import ndtr
from inventory import envelope
sys.path.insert(0,'/workspace/clean-neurons')
from m16.approaches.small_ladder_review.portable_cost.accounting import Audit,gauss_rule

def radial(a,n,audit):
    """Return F_n and F_(n-1), n=2 or3, exactly in real arithmetic."""
    aa=a*a;ratio=a/np.sqrt(1+aa);den=1+2*aa;angle=np.arcsin(a/np.sqrt(den))
    f1=.5+.5*ratio;f2=.25+2*ratio*(.25+angle/(2*math.pi))
    if n==2:
        audit.add('radial_closed_form',90*a.size)
        return f2,f1
    f3=.125+3*ratio*(.125+(2*angle+np.arcsin(aa/den))/(4*math.pi))
    audit.add('radial_closed_form',125*a.size)
    return f3,f2

def sectors(model,audit):
    wi,wh,wo=model;env=envelope(wo);angles=[0.,math.pi/4,math.pi/2,5*math.pi/4,3*math.pi/2,2*math.pi]
    t=len(wo);pairs=t*(t-1)//2
    audit.add('envelope_setup',20*pairs+(pairs+1)*(4*t+10)+(pairs+1)**2+128)
    maps={}
    for sign in (1,-1):
        first=wi*(wi*sign>0);b=np.stack([wh@first,wi],axis=1);maps[sign]=b
        normals=list(b)
        normals.extend((wo[left['label']]-wo[right['label']])@b for left,right in zip(env[:-1],env[1:]))
        audit.add('boundary_affine_setup',32+16*len(env))
        for normal in normals:
            if not np.any(normal):continue
            theta=math.atan2(-float(normal[0]),float(normal[1]))%(2*math.pi)
            for candidate in (theta,(theta+math.pi)%(2*math.pi)):
                if sign*math.cos(candidate)>=0:angles.append(candidate)
            audit.add('boundary_roots',96)
    angles=sorted(set(angles));pieces=[]
    audit.add('boundary_sort',len(angles)**2)
    for lo,hi in zip(angles[:-1],angles[1:]):
        if hi==lo:continue
        middle=(lo+hi)/2;direction=np.array([math.cos(middle),math.sin(middle)])
        b=maps[1 if direction[0]>=0 else -1];pre=b@direction;h=np.maximum(pre,0)
        label=int(np.argmax(wo@h));pieces.append((lo,hi,label,b))
        audit.add('sector_readout',80+6*t)
    return pieces

def calculate(model,order=16):
    start=time.process_time();model=tuple(np.asarray(w,float) for w in model);wi,wh,wo=model;t=len(wo);m=t-2
    if wi.shape!=(2,) or wh.shape!=(2,2) or wo.shape!=(t,2) or t not in (4,5):raise ValueError('H2T4/T5 required')
    audit=Audit();audit.add('input_validation',128);pieces=sectors(model,audit);x,w=gauss_rule(order,audit)
    a=abs(wh);history=a@a@abs(wi);ebar=history.copy()
    for _ in range(1,m):history=a@history;ebar+=history
    ebar*=4.;original_ebar=ebar.copy()
    plus=np.maximum(wh,0);minus=np.maximum(-wh,0);term=abs(wi);prefix=term.copy()
    for _ in range(1,m):term=plus@term;prefix+=term
    prefix*=4.;first_tail=np.maximum(minus@prefix,plus@prefix)
    ebar=np.minimum(ebar,a@first_tail);box=float((2*ndtr(4.)-1)**m)
    audit.add('history_bound_setup',360+32*m)
    total=agreement=0.;family_masses={};nodes=0
    for lo,hi,label,b in pieces:
        theta=lo+(hi-lo)*x;weights=(hi-lo)*w/(2*math.pi);cos=np.cos(theta);sin=np.sin(theta)
        upper=np.maximum(cos,sin);lower=np.minimum(cos,sin)
        fu,fu_prev=radial(upper,m,audit);fl,_=radial(lower,m,audit)
        if label<t-2:truth=1/m-fu_prev+fu-fl/m
        else:
            is_larger=(cos>=sin) if label==t-2 else (sin>=cos)
            truth=np.where(is_larger,m*(fu_prev-fu),fl)
        total+=float(weights@truth);family_masses[str(label)]=family_masses.get(str(label),0.)+float(weights.sum())
        pre=np.stack([cos,sin],axis=1)@b.T;hidden=np.maximum(pre,0);active=pre>0
        thresholds=np.zeros(order);count=active.sum(axis=1)
        simple=count<=1
        if simple.any():
            magnitude=abs(pre[simple]);ratios=np.divide(ebar,magnitude,out=np.full_like(magnitude,np.inf),where=magnitude!=0)
            ratios[(magnitude==0)&(ebar==0)]=0.;thresholds[simple]=ratios.max(axis=1)
        both=~simple
        if both.any():
            for other in range(t):
                if other==label:continue
                margin=wo[label]-wo[other];value=hidden[both]@margin;drift=float(abs(margin)@ebar)
                if np.all(margin==0) and label<other:continue
                ratio=np.divide(drift,value,out=np.full_like(value,np.inf),where=value>0)
                thresholds[both]=np.maximum(thresholds[both],ratio)
        agreement+=float(weights@np.exp(-.5*thresholds**2))*box
        audit.add('angular_rank_and_mass',80*order)  # sin/cos40, interpolation, rank weights, sums
        audit.add('history_certificate_nodes',(100+24*t)*order)
        audit.add('history_certificate_setup',32*t)
        nodes+=order
    return total,dict(cpu_seconds=time.process_time()-start,order=order,angular_sectors=len(pieces),angular_nodes=nodes,
        output_family_masses=family_masses,history_component_bound=ebar.tolist(),original_history_component_bound=original_ebar.tolist(),prefix_box_mass=box,
        numerical_agreement_lower_diagnostic=agreement,quadrature_certified=False,
        numerical_history_error_upper_diagnostic=1-agreement,cost=audit.report(),
        forward_equivalents=audit.known/(13*t),model_inputs_sampled=0)

def est(model,budget,rng):
    value,stats=calculate(model)
    return value,float('inf')  # finite charge awaits independent review/accuracy validation
