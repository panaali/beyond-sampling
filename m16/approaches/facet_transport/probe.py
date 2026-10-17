"""Tiny numerical spherical clipping and moving-decision-facet diagnostic."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import itertools,json,time
from pathlib import Path
import numpy as np

COUNTS={'clips':0,'gate_affine_products':0,'validation_forwards':0}
EPS=1e-11

def norm(x):
    scale=np.max(np.abs(x))
    return float(scale*np.linalg.norm(x/scale)) if scale else 0.

def unit(x):
    scale=np.max(np.abs(x))
    if scale==0:raise ValueError('cannot normalize a zero vector')
    y=x/scale
    return y/np.linalg.norm(y)

def area(p):
    if len(p)<3:return 0.
    a=p[0];out=0.
    for b,c in zip(p[1:-1],p[2:]):
        out+=2*np.arctan2(abs(np.linalg.det(np.stack([a,b,c]))),1+a@b+b@c+c@a)
    return float(out)

def clip(p,n):
    COUNTS['clips']+=1
    if len(p)<3:return np.empty((0,3))
    if not np.any(n):return p.copy()
    n=unit(n);vals=p@n
    if vals.min()>=-EPS:return p.copy()
    if vals.max()<=EPS:return np.empty((0,3))
    out=[]
    for u,v,du,dv in zip(p,np.roll(p,-1,axis=0),vals,np.roll(vals,-1)):
        if du>=-EPS:out.append(u)
        if (du>EPS and dv < -EPS) or (du < -EPS and dv>EPS):
            out.append(unit(u+(v-u)*du/(du-dv)))
    clean=[]
    for v in out:
        if not clean or np.linalg.norm(v-clean[-1])>1e-10:clean.append(v)
    if len(clean)>1 and np.linalg.norm(clean[0]-clean[-1])<1e-10:clean.pop()
    q=np.array(clean).reshape(-1,3)
    return q if area(q)>1e-14 else np.empty((0,3))

def octants():
    return [np.diag(s).astype(float) for s in itertools.product([-1,1],repeat=3)]

def gate_complex(model):
    wi,wh,wo=model;H=len(wi)
    regions=[(p,np.zeros((H,3))) for p in octants()]
    stage_counts=[]
    for t in range(3):
        nxt=[]
        for p,B in regions:
            Z=wh@B;Z[:,t]+=wi;COUNTS['gate_affine_products']+=1
            active=[(p,np.zeros((H,3)))]
            for j in range(H):
                if not np.any(Z[j]):continue
                children=[]
                for q,A in active:
                    pos=clip(q,Z[j]);neg=clip(q,-Z[j])
                    if len(pos):
                        C=A.copy();C[j]=Z[j];children.append((pos,C))
                    if len(neg):children.append((neg,A.copy()))
                active=children
            nxt.extend(active)
        regions=nxt;stage_counts.append(len(regions))
        if len(regions)>20000:raise RuntimeError('gate cell cap exceeded')
    return regions,stage_counts

def rank_complex(regions):
    planes=[np.eye(3)[a]-np.eye(3)[b] for a,b in [(0,1),(0,2),(1,2)]]
    out=[]
    for p,B in regions:
        pieces=[p]
        for n in planes:
            pieces=[q for p in pieces for q in (clip(p,n),clip(p,-n)) if len(q)]
        for p in pieces:
            target=int(np.argsort(p.sum(axis=0))[-2]);out.append((p,B,target))
    if len(out)>50000:raise RuntimeError('rank cell cap exceeded')
    return out

def margin_clip(p,n,j,k):
    if not np.any(n):
        return np.empty((0,3)) if j<k else p
    return clip(p,n)

def accuracy(regions,wo):
    total=0.
    for p,B,k in regions:
        q=p
        for j in range(3):
            if j!=k:q=margin_clip(q,(wo[k]-wo[j])@B,j,k)
        total+=area(q)
    return total/(4*np.pi)

def plane_arc(p,n):
    if len(p)<3 or not np.any(n):return None
    n=unit(n);v=p@n;points=[]
    for x,y,a,b in zip(p,np.roll(p,-1,axis=0),v,np.roll(v,-1)):
        if abs(a)<1e-9:points.append(x)
        if a*b<0:points.append(unit(x+(y-x)*a/(a-b)))
    unique=[]
    for x in points:
        if all(np.linalg.norm(x-y)>1e-8 for y in unique):unique.append(x)
    if len(unique)<2:return None
    u,w=max(itertools.combinations(unique,2),key=lambda z:np.linalg.norm(z[0]-z[1]))
    angle=np.arccos(np.clip(u@w,-1,1))
    return (u,w,float(angle)) if angle>1e-8 else None

def merge_count(arcs):
    groups={}
    for n,u,v in arcs:
        n=unit(n)
        if n[np.flatnonzero(abs(n)>1e-8)[0]]<0:n=-n
        key=tuple(np.round(n,9));groups.setdefault(key,[]).append((u,v))
    count=0
    for key,vals in groups.items():
        n=unit(np.array(key));axis=np.eye(3)[np.argmin(abs(n))];e=unit(np.cross(n,axis));f=np.cross(n,e)
        intervals=[]
        for u,v in vals:
            a=np.arctan2(u@f,u@e)%(2*np.pi);b=np.arctan2(v@f,v@e)%(2*np.pi)
            gap=(b-a+np.pi)%(2*np.pi)-np.pi
            if gap<0:a,b=b,a;gap=-gap
            b=a+gap
            if b>2*np.pi:intervals.extend([(a,2*np.pi),(0,b-2*np.pi)])
            else:intervals.append((a,b))
        merged=[]
        for a,b in sorted(intervals):
            if merged and a<=merged[-1][1]+1e-8:merged[-1][1]=max(merged[-1][1],b)
            else:merged.append([a,b])
        ncomp=len(merged)
        if ncomp>1 and merged[0][0]<1e-8 and merged[-1][1]>2*np.pi-1e-8:ncomp-=1
        count+=ncomp
    return len(groups),count

def flux(regions,wo,delta):
    # Margin-zero arcs are candidates: boundary_audit removes internal ties.
    result=0.;arcs=[];degenerate=0
    for p,B,k in regions:
        for j in range(3):
            if j==k:continue
            n=(wo[k]-wo[j])@B;dn=(delta[k]-delta[j])@B
            if not np.any(n):
                if np.any(dn):degenerate+=1
                continue
            q=p
            for other in range(3):
                if other!=k and other!=j:q=margin_clip(q,(wo[k]-wo[other])@B,other,k)
            edge=plane_arc(q,n)
            if edge is None:continue
            u,v,angle=edge
            result+=(dn@(u+v))*np.tan(angle/2)/norm(n)
            arcs.append((n,u,v))
    normals,merged=merge_count(arcs)
    return result/(4*np.pi),dict(decision_arcs=len(arcs),unique_planes=normals,merged_arcs=merged,
        candidate_arcs_require_boundary_audit=True,
        transport_merge_performed=False,degenerate_margins=degenerate)

def validate(regions,model):
    wi,wh,wo=model;worst=0.
    for p,B in regions:
        x=unit(p.sum(axis=0));h=np.zeros(len(wi))
        for value in x:h=np.maximum(wh@h+wi*value,0)
        COUNTS['validation_forwards']+=1
        worst=max(worst,float(np.max(abs(h-B@x))))
    return worst

def main():
    wall=time.monotonic();cpu=time.process_time()
    assert abs(sum(map(area,octants()))-4*np.pi)<1e-12
    assert abs(sum(area(clip(p,np.array([1.,2.,3.]))) for p in octants())-2*np.pi)<1e-12
    rng=np.random.default_rng(31001)
    synth=(rng.standard_normal(3),rng.standard_normal((3,3)),rng.standard_normal((3,3)))
    from alg_zoo import zoo_2nd_argmax
    m=zoo_2nd_argmax(4,3,seed=0)
    base=tuple(w.detach().numpy().astype(float) for w in (m.rnn.weight_ih_l0,m.rnn.weight_hh_l0,m.linear.weight))
    base=(base[0][:,0],base[1],base[2])
    cases=[('synthetic',synth)]
    for sigma in [.0025,.02]:
        rng=np.random.default_rng(1)
        model=tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
        cases.append((f'dev_H4_T3_s{sigma}',model))
    report=dict(protocol_commit='fcdde11',cases=[])
    for name,model in cases:
        t0=time.process_time();before=COUNTS.copy()
        regions,stages=gate_complex(model);ranked=rank_complex(regions)
        affine_error=validate(regions,model)
        wo=model[2];delta=np.random.default_rng(31002).standard_normal(wo.shape)*.1*np.sqrt(np.mean(wo*wo))
        derivative,geometry=flux(ranked,wo,delta)
        differences=[]
        for step in [1e-3,5e-4,2.5e-4]:
            plus=accuracy(ranked,wo+step*delta);minus=accuracy(ranked,wo-step*delta)
            fd=(plus-minus)/(2*step)
            differences.append(dict(step=step,plus=plus,minus=minus,finite_difference=fd,absolute_error=abs(fd-derivative)))
        record=dict(name=name,accuracy=accuracy(ranked,wo),gate_cells=len(regions),stage_counts=stages,
            rank_cells=len(ranked),gate_area_error=abs(sum(area(p) for p,B in regions)-4*np.pi),
            rank_area_error=abs(sum(area(p) for p,B,k in ranked)-4*np.pi),
            max_affine_validation_error=affine_error,flux=derivative,finite_differences=differences,
            gate_to_merged_ratio=len(regions)/max(geometry['merged_arcs'],1),
            counts={k:COUNTS[k]-before[k] for k in COUNTS},cpu_seconds=time.process_time()-t0,**geometry)
        print(json.dumps(record),flush=True);report['cases'].append(record)
        if time.process_time()-cpu>300:raise RuntimeError('CPU cap exceeded')
    report.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,counts=COUNTS,
        classification='tiny exact-geometry diagnostic in float64, not a global estimator or formal certificate')
    Path(__file__).with_name('results.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
