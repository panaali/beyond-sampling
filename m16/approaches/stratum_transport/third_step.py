"""Forward push of two-input strata through one fresh Gaussian input.

Uses elementary Rayleigh ratio integrals, with numerical angle boundaries.
"""
from fractions import Fraction as F
import math,time
import numpy as np
from two_input import Law


def primitive(theta,form,side=None):
    if form is None:return theta if side=='upper' else 0.
    a,b=map(float,form)
    return theta/2+math.asin((a*math.sin(theta)-b*math.cos(theta))/math.sqrt(1+a*a+b*b))/2


def roots_in(normal,lo,hi):
    if not any(normal):return []
    x=math.atan2(-float(normal[0]),float(normal[1]))
    return [x+k*math.pi for k in (-2,-1,0,1,2) if lo<x+k*math.pi<hi]


def push(model,deadline=math.inf):
    law=Law(model);wi=[F(float(x)) for x in law.wi]
    wh=[[F(float(x)) for x in row] for row in law.wh]
    out=np.zeros(len(law.wo));masses={k:0. for k in ('origin','axis0','axis1','input_ray','area')}
    stats={'source_sectors':len(law.cells),'threshold_forms':0,'angular_intervals':0,'destination_bands':0,
        'skipped_angular_mass_bound':0.,'minimum_band_mass':0.}
    def dot(a,b):return law.add(law.mul(a[0],b[0]),law.mul(a[1],b[1]))
    for cell in law.cells:
        B=cell['B'];C=[[dot(row,[B[0][k],B[1][k]]) for k in (0,1)] for row in wh]
        forms=set();angular=[];gates={}
        for i in (0,1):
            if wi[i]:forms.add(tuple(law.div(-c,wi[i]) for c in C[i]))
            else:angular.append(tuple(C[i]))
        for mask in range(4):
            M=[[*(C[i] if mask&(1<<i) else [F(0),F(0)]),wi[i] if mask&(1<<i) else F(0)] for i in (0,1)]
            lines=[tuple(dot(row,[M[0][k],M[1][k]]) for k in range(3)) for row in law.woq]
            for i in range(len(lines)):
                for j in range(i):
                    diff=[law.add(lines[i][k],-lines[j][k]) for k in range(3)]
                    if diff[2]:forms.add(tuple(law.div(-d,diff[2]) for d in diff[:2]))
                    else:angular.append(tuple(diff[:2]))
            rank=0 if not any(any(row) for row in M) else 1
            for k in range(3):
                for l in range(k):
                    if law.add(law.mul(M[0][k],M[1][l]),-law.mul(M[0][l],M[1][k])):rank=2
            kind='area' if rank==2 else ('origin' if not rank else ('axis0' if not any(M[1]) else ('axis1' if not any(M[0]) else 'input_ray')))
            reps=[next(j for j in range(i+1) if lines[j]==line) for i,line in enumerate(lines)]
            gates[mask]=(kind,np.array(lines,dtype=float),reps)
        forms=list(forms);stats['threshold_forms']+=len(forms)
        lo=-math.pi/2 if cell['lo'] is None else math.atan(float(cell['lo']))
        hi=math.pi/2 if cell['hi'] is None else math.atan(float(cell['hi']))
        angles={lo,hi}
        for v in angular:angles.update(roots_in(v,lo,hi))
        for i in range(len(forms)):
            for j in range(i):
                v=[law.add(forms[i][k],-forms[j][k]) for k in (0,1)]
                angles.update(roots_in(v,lo,hi))
        angles=sorted(angles);Cf=np.array(C,dtype=float);wf=np.array(wi,dtype=float)
        for a,b in zip(angles,angles[1:]):
            if time.process_time()>deadline:raise TimeoutError('forward push deadline')
            if b-a<1e-14:
                stats['skipped_angular_mass_bound']+=(b-a)/(2*math.pi);continue
            stats['angular_intervals']+=1
            theta=(a+b)/2;u=np.array([math.cos(theta),math.sin(theta)])
            ordered=sorted(((float(np.array(v,dtype=float)@u),v) for v in forms),key=lambda item:item[0])
            bounds=[(-math.inf,None),*ordered,(math.inf,None)]
            for (vl,fl),(vh,fh) in zip(bounds,bounds[1:]):
                if vl==vh:continue
                probe=(vl+vh)/2 if np.isfinite(vl) and np.isfinite(vh) else (vh-1 if np.isneginf(vl) else vl+1)
                if not np.isfinite(probe):probe=0.
                active=Cf@u+wf*probe>0;mask=int(active[0])+2*int(active[1])
                kind,lines,reps=gates[mask]
                winner=reps[int(np.argmax(lines@np.array([*u,probe])))]
                mass=((primitive(b,fh,'upper')-primitive(a,fh,'upper'))-
                      (primitive(b,fl,'lower')-primitive(a,fl,'lower')))/(2*math.pi)
                stats['minimum_band_mass']=min(stats['minimum_band_mass'],mass)
                if mass < -1e-12:raise ArithmeticError(f'negative band mass {mass}')
                masses[kind]+=mass;out[winner]+=mass;stats['destination_bands']+=1
    return {'stratum_masses':masses,'output_masses':out.tolist(),'work':stats,
        'rational_and_angle_work':dict(law.work),'classification':'exact-form sector push with floating angular arithmetic'}
