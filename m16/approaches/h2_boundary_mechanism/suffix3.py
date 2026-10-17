"""Fixed three-input H2 boundary arrangement and exact omitted-rank weights.

No original-T region enumeration, model-input sampling, or reference imports.
Operation charges are proposed upper bounds awaiting independent review.
"""
import itertools, math, time
import numpy as np
from suffix import Audit, gauss_rule
from inventory import envelope


def normalize(v, audit):
    size=float(np.max(np.abs(v)))
    audit.add('normalize',38)
    if size==0:return v.copy()
    v=v/size
    return v/math.sqrt(float(v@v))


def clip(poly, normal, audit):
    """Both closed spherical half-polygons; no near-zero deletion threshold."""
    normal=normalize(normal,audit)
    if not np.any(normal):return [poly]
    values=poly@normal
    audit.add('polygon_sides',12*len(poly)+12)
    if np.all(values>=0) or np.all(values<=0):return [poly]
    answer=[]
    for sign in (1,-1):
        verts=[]
        for i in range(len(poly)):
            j=(i+1)%len(poly);a=sign*values[i];b=sign*values[j]
            if a>=0:verts.append(poly[i])
            if (a<0 and b>0) or (a>0 and b<0):
                verts.append(normalize(abs(b)*poly[i]+abs(a)*poly[j],audit))
                audit.add('clip_intersection',16)
            audit.add('clip_control',20)
        # A zero endpoint is included once by its own edge, not by intersection.
        if len(verts)>=3:answer.append(np.asarray(verts))
    return answer


def boundaries(model,audit):
    wi,wh,wo=model;t=len(wo);env=envelope(wo);pairs=t*(t-1)//2
    audit.add('envelope_setup',20*pairs+(pairs+1)*(4*t+10)+(pairs+1)**2+128)
    pieces=[]
    for signs in itertools.product((-1,1),repeat=3):
        poly=np.diag(signs).astype(float)
        h=np.zeros((2,3));h[:,0]=wi*(wi*signs[0]>0)
        pieces.append((poly,h));audit.add('initial_charts',32)
    counts=[len(pieces)]
    for step in (1,2):
        children=[]
        for poly,h in pieces:
            pre=wh@h;pre[:,step]+=wi;audit.add('recurrent_map',30)
            polygons=[poly]
            for normal in pre:
                polygons=[child for p in polygons for child in clip(p,normal,audit)]
            for p in polygons:
                center=p.sum(axis=0);active=pre@center>0
                children.append((p,pre*active[:,None]));audit.add('gate_map_classification',6*len(p)+30)
        pieces=children;counts.append(len(pieces))
    outputs=[]
    for poly,h in pieces:
        normals=[(wo[a['label']]-wo[b['label']])@h for a,b in zip(env[:-1],env[1:])]
        # Three order planes make top-two suffix indices constant per piece.
        normals.extend(np.eye(3)[i]-np.eye(3)[j] for i,j in ((0,1),(0,2),(1,2)))
        audit.add('output_and_rank_planes',20*len(normals))
        polygons=[poly]
        for normal in normals:polygons=[child for p in polygons for child in clip(p,normal,audit)]
        for p in polygons:
            center=p.sum(axis=0);hidden=h@center;label=int(np.argmax(wo@hidden));rank=np.argsort(center)
            outputs.append((p,label,int(rank[-1]),int(rank[-2])))
            audit.add('piece_readout_and_rank',6*len(p)+12*t+50)
    triangles=[];midpoint_mismatches=0
    for poly,label,largest,second in outputs:
        for i in range(1,len(poly)-1):
            v=np.stack((poly[0],poly[i],poly[i+1]))
            # Explicit determinant, not an opaque numerical kernel.
            cross=np.cross(v[1],v[2]);det=abs(float(v[0]@cross));audit.add('triangulation_determinant',40)
            if det==0:continue
            triangles.append((v,det,label,largest,second))
    return triangles,dict(gate_piece_counts=counts,output_polygons=len(outputs),triangles=len(triangles))


def radial(a,n,audit):
    """chi3 Fn and F(n-1), n=1,2, including negative a."""
    aa=a*a;den=1+aa;angle=np.arctan(a)
    f1=.5+(angle+a/den)/math.pi
    if n==1:
        audit.add('chi3_radial',32*a.size)
        return f1,np.ones_like(a)
    f2=.25+(2*angle+np.arcsin(aa/den)+2*a/den+2*aa/(den*np.sqrt(1+2*aa)))/(2*math.pi)
    audit.add('chi3_radial',88*a.size)
    return f2,f1


def calculate(model,order=8,validation=False):
    start=time.process_time();wi,wh,wo=(np.asarray(a,float) for a in model);model=(wi,wh,wo);t=len(wo);m=t-3
    if wi.shape!=(2,) or wh.shape!=(2,2) or wo.shape!=(t,2) or m not in (1,2):raise ValueError('H2T4/T5 only')
    audit=Audit();audit.add('input_validation',128)
    triangles,stats=boundaries(model,audit);x,w=gauss_rule(order,audit)
    u,v=np.meshgrid(x,x,indexing='ij');wu,wv=np.meshgrid(w,w,indexing='ij')
    bary=np.stack((u,(1-u)*v,(1-u)*(1-v)),axis=-1).reshape(-1,3)
    weights=(wu*wv*(1-u)).ravel();audit.add('triangle_rule',15*order*order)
    value=mass=rankmass=0.;families={};mismatches=0;mismatch_mass=mismatch_mass_upper=0.
    for rays,det,label,largest,second in triangles:
        z=bary@rays;q=np.sum(z*z,axis=1);r=np.sqrt(q);direction=z/r[:,None]
        angular=weights*det/(4*math.pi*q*r)
        fu,fp=radial(direction[:,largest],m,audit);fs,_=radial(direction[:,second],m,audit)
        if label<t-3:truth=1/m-fp+fu-fs/m
        elif label==t-3+largest:truth=m*(fp-fu)
        elif label==t-3+second:truth=fs
        else:truth=np.zeros_like(fu)
        value+=float(angular@truth);piece_mass=float(angular.sum());mass+=piece_mass
        rankmass+=float(angular@(m*(1/m-fp+fu-fs/m)+m*(fp-fu)+fs))
        families[str(label)]=families.get(str(label),0.)+piece_mass
        audit.add('triangle_nodes',90*len(weights));audit.add('triangle_setup',30)
        if validation:
            c=rays.sum(axis=0);h=np.zeros(2)
            for a in c:h=np.maximum(wh@h+wi*a,0)
            mismatch=int(np.argmax(wo@h)!=label);mismatches+=mismatch
            if mismatch:
                mismatch_mass+=piece_mass
                # All vertices are unit and in one signed orthant: norm²>=1/3.
                mismatch_mass_upper+=det*3**1.5/(8*math.pi)
    stats.update(estimate=value,angular_mass=mass,rank_weight_mass=rankmass,output_family_masses=families,
        angular_order=order,angular_nodes=order*order*len(triangles),cost=audit.report(),forward_equivalents=audit.known/(13*t),
        cpu_seconds=time.process_time()-start,midpoint_label_mismatches=mismatches if validation else None,
        midpoint_mismatch_mass=mismatch_mass if validation else None,
        midpoint_mismatch_mass_upper=mismatch_mass_upper if validation else None,
        quadrature_certified=False,history_error_certified=False,model_inputs_sampled=0)
    return float(value),stats


def est(model,budget,rng):
    return calculate(model)[0],float('inf')
