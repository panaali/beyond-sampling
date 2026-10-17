"""Positive two-input conditional hidden law with exact-mean angular cells."""
from collections import defaultdict
import math
import numpy as np

ROOT2PI=math.sqrt(2*math.pi)

def rayleigh(n,stats):
    if n not in (2,4):raise ValueError('frozen radial orders2/4 only')
    moments=np.array([2.**(k/2)*math.gamma(1+k/2) for k in range(2*n)])
    H=np.array([[moments[i+j] for j in range(n)] for i in range(n)])
    H1=np.array([[moments[i+j+1] for j in range(n)] for i in range(n)])
    L=np.linalg.cholesky(H)
    J=np.linalg.solve(L,np.linalg.solve(L,H1).T).T
    if np.max(abs(J-J.T))>1e-10:raise ArithmeticError('moment Jacobi symmetry')
    nodes,vectors=np.linalg.eigh((J+J.T)/2)
    weights=vectors[0]**2
    stats['radial_moment_gamma_calls']+=2*n
    stats['radial_hankel_entries']+=2*n*n
    stats['radial_cholesky_calls']+=1;stats['radial_linear_solve_calls']+=2;stats['radial_eigh_calls']+=1
    if not(np.all(nodes>0) and np.all(weights>0)):raise ArithmeticError('nonpositive radial rule')
    errors=np.array([weights@nodes**k-moments[k] for k in range(2*n)])
    if np.max(abs(errors)/np.maximum(1.,abs(moments)))>1e-11:raise ArithmeticError('radial moments')
    return nodes,weights,dict(nodes=nodes.tolist(),weights=weights.tolist(),moment_errors=errors.tolist(),hankel_condition=float(np.linalg.cond(H)))

def analytic_mean(wi,wh,sign):
    c=wh@np.maximum(sign*wi,0);r=np.hypot(c,wi);mu=np.zeros_like(c)
    pos=c>=0;mu[pos]=(r[pos]+c[pos])/ROOT2PI
    neg=~pos;mu[neg]=wi[neg]**2/((r[neg]-c[neg])*ROOT2PI)
    return mu

def conditional(wi,wh,sign,subdivisions,radii,radial_weights,stats,rank_aligned=False):
    if sign not in (-1,1) or subdivisions not in (2,4):raise ValueError('frozen sign/subdivisions')
    c=wh@np.maximum(sign*wi,0);stats['prefix_matrix_scalar_products']+=len(wi)**2
    cuts=[-math.pi/2,math.pi/2]
    if rank_aligned:cuts.append(sign*math.pi/4)
    for a,b in zip(c,wi):
        if a==0 and b==0:continue
        angle=math.atan2(-a,b);stats['gate_atan2_calls']+=1
        for k in (-1,0,1):
            root=angle+k*math.pi;stats['gate_root_boundary_tests']+=2
            if -math.pi/2<root<math.pi/2:cuts.append(root)
    cuts=sorted(set(cuts));stats['gate_angle_sort_items']+=len(cuts)
    states=[];weights=[];support=[];inputs=[];second=np.zeros((len(wi),len(wi)));strata={};panel_records=[]
    for left,right in zip(cuts[:-1],cuts[1:]):
        width=right-left;mid=left+width/2
        mask=c*math.cos(mid)+wi*math.sin(mid)>0
        a=c*mask;b=wi*mask;tag=''.join('1' if x else '0' for x in mask)
        strata[tag]=strata.get(tag,0.)+width/math.pi
        sinc=math.sin(width)/width
        cc=(1+math.cos(2*mid)*sinc)/2;ss=(1-math.cos(2*mid)*sinc)/2;cs=math.sin(2*mid)*sinc/2
        second+=2*width/math.pi*(cc*np.outer(a,a)+ss*np.outer(b,b)+cs*(np.outer(a,b)+np.outer(b,a)))
        for index in range(subdivisions):
            lo=left+width*index/subdivisions;hi=left+width*(index+1)/subdivisions
            span=hi-lo;center=lo+span/2;half=span/2
            factor=math.sin(half)/half
            direction=a*(math.cos(center)*factor)+b*(math.sin(center)*factor)
            if np.any(direction<0) or not np.all(np.isfinite(direction)):raise ArithmeticError('negative/nonfinite cell-mean state')
            for radius,rweight in zip(radii,radial_weights):
                states.append(radius*direction);weights.append(span/math.pi*rweight);support.append(tag)
                inputs.append([sign*radius*math.cos(center)*factor,radius*math.sin(center)*factor])
            stats['angular_panels']+=1;stats['angular_trig_calls']+=3
            stats['state_atom_coordinates']+=len(radii)*len(wi)
            panel_records.append(dict(lo=lo,hi=hi,mask=tag,direction_mean=direction.tolist()))
    states=np.array(states);weights=np.array(weights)
    mean=weights@states;represented_second=states.T@(weights[:,None]*states)
    loss=second-represented_second;analytic=analytic_mean(wi,wh,sign)
    if abs(weights.sum()-1)>1e-12:raise ArithmeticError('conditional mass')
    if np.max(abs(mean-analytic))>1e-11*max(1.,np.max(abs(analytic))):raise ArithmeticError('conditional mean')
    eig=np.linalg.eigvalsh(loss)
    if min(eig)<-1e-11*max(1.,np.linalg.norm(second)):raise ArithmeticError('covariance loss not PSD')
    report=dict(sign=sign,atoms=len(weights),mass=float(weights.sum()),mean=mean.tolist(),analytic_mean=analytic.tolist(),mean_error=float(np.max(abs(mean-analytic))),second_moment=second.tolist(),represented_second_moment=represented_second.tolist(),covariance_loss=loss.tolist(),covariance_loss_eigenvalues=eig.tolist(),stratum_masses=strata,panels=panel_records,prefix_inputs=inputs,rank_aligned=rank_aligned)
    stats['atom_mean_validation_products']+=states.size
    stats['atom_second_validation_products']+=len(weights)*len(wi)**2
    stats['covariance_validation_eigh_calls']+=1
    return states,weights,report

def atoms(model,subdivisions=2,radial_order=2,rank_aligned=False):
    wi,wh,wo=(np.asarray(x,dtype=float) for x in model)
    if len(wo)!=4:raise ValueError('two-input prefix requires T4')
    stats=defaultdict(float);r,w,radial=rayleigh(radial_order,stats);result={};reports={}
    for sign in (1,-1):
        states,weights,report=conditional(wi,wh,sign,subdivisions,r,w,stats,rank_aligned)
        result[sign]=(states,weights);reports[sign]=report
    return result,dict(subdivisions=subdivisions,radial_order=radial_order,radial=radial,conditional=reports,work=dict(stats))
