"""Post-review audit: distinguish genuine correctness boundaries from internal ties."""
import json,time
from pathlib import Path
import numpy as np
import torch
import probe as P


def raw_arcs(regions,wo,delta):
    arcs=[]
    for poly,B,k in regions:
        for j in range(3):
            if j==k:continue
            n=(wo[k]-wo[j])@B;dn=(delta[k]-delta[j])@B
            if not np.any(n):continue
            q=poly
            for other in range(3):
                if other not in (j,k):q=P.margin_clip(q,(wo[k]-wo[other])@B,other,k)
            edge=P.plane_arc(q,n)
            if edge is not None:
                u,v,angle=edge;arcs.append((n,dn,u,v,angle))
    return arcs


def refine(arcs,regions):
    points=np.array([x for p,B,k in regions for x in p]+[x for n,dn,u,v,a in arcs for x in (u,v)])
    out=[]
    for index,(n,dn,u,v,angle) in enumerate(arcs):
        nh=P.unit(n);tangent=P.unit(v-(u@v)*u)
        params=np.arctan2(points@tangent,points@u)
        mask=(abs(points@nh)<1e-8)&(params>1e-9)&(params<angle-1e-9)
        values=sorted([0.,angle]+params[mask].tolist())
        knots=[values[0]]
        for t in values[1:]:
            if t-knots[-1]>1e-9:knots.append(t)
        for a,b in zip(knots[:-1],knots[1:]):
            x=np.cos(a)*u+np.sin(a)*tangent;y=np.cos(b)*u+np.sin(b)*tangent
            out.append((index,n,dn,x,y,b-a))
    return out


def classify(model,piece):
    _,n,dn,u,v,angle=piece;nh=P.unit(n);tan=P.unit(v-(u@v)*u)
    delta=min(1e-4,angle/10)
    scales=delta*10.**(-np.arange(4))
    if scales[-1]<1e-10:return 'uncertain_short',None,0
    pts=np.array([np.cos(t*angle)*u+np.sin(t*angle)*tan for t in [1/3,1/2,2/3]])
    x=np.array([np.cos(d)*x+sign*np.sin(d)*nh for x in pts for d in scales for sign in [-1,1]])
    wi,wh,wo=model;h=np.zeros((len(x),len(wi)))
    for t in range(3):h=np.maximum(h@wh.T+x[:,t,None]*wi,0)
    logits=h@wo.T
    target=np.argsort(x,axis=1)[:,-2];pred=logits.argmax(axis=1)
    status=(target==pred).reshape(3,4,2)
    last=status[:,1:,:]
    # Require agreement across all three interior positions and three finest scales.
    stable=np.all(last==last[0,0,:])
    sorted_logits=np.sort(logits,axis=1)
    gap=sorted_logits[:,-1]-sorted_logits[:,-2]
    # Exact all-zero hidden states have defined argmax ties; other near ties are uncertain.
    near=(gap<=1e-12*(1+abs(logits).max(axis=1)))&np.any(h!=0,axis=1)
    if np.any(near.reshape(3,4,2)[:,1:,:]):return 'uncertain_numeric',status.tolist(),len(x)
    if not stable:return 'uncertain_transition',status.tolist(),len(x)
    return ('genuine' if last[0,0,0]!=last[0,0,1] else 'internal'),status.tolist(),len(x)


def main():
    started=time.process_time();wall=time.monotonic()
    r=np.random.default_rng(31001);synthetic=(r.standard_normal(3),r.standard_normal((3,3)),r.standard_normal((3,3)))
    state=torch.load('/workspace/work/checkpoints/H4_T3_seed0.pth',map_location='cpu',weights_only=True)
    w=[state[k].numpy().astype(float) for k in ['rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight']]
    base=(w[0][:,0],w[1],w[2]);cases=[('synthetic',synthetic)]
    for sigma in [.0025,.02]:
        r=np.random.default_rng(1);cases.append((f'dev_H4_T3_s{sigma}',tuple(w+sigma*np.sqrt(np.mean(w*w))*r.standard_normal(w.shape) for w in base)))
    report={'cases':[],'note':'Post-review correction audit on unchanged models; no new candidate exploration.'}
    for name,model in cases:
        gates,_=P.gate_complex(model);regions=P.rank_complex(gates)
        wo=model[2];delta=np.random.default_rng(31002).standard_normal(wo.shape)*.1*np.sqrt(np.mean(wo*wo))
        raw=raw_arcs(regions,wo,delta);pieces=refine(raw,regions)
        records=[];genuine=[];uncertain=[];counter={};fluxes={};evals=0
        for index,n,dn,u,v,a in pieces:
            label,sides,cost=classify(model,(index,n,dn,u,v,a));evals+=cost
            flux=float(dn@(u+v)*np.tan(a/2)/P.norm(n)/(4*np.pi))
            counter[label]=counter.get(label,0)+1;fluxes[label]=fluxes.get(label,0)+flux
            if label=='genuine':genuine.append((n,u,v))
            elif label!='internal':uncertain.append((n,u,v))
            records.append(dict(raw_arc=index,label=label,angle=a,flux=flux,side_correctness=sides,
                normal=n.tolist(),velocity=dn.tolist(),start=u.tolist(),end=v.tolist()))
        ng,mg=P.merge_count(genuine) if genuine else (0,0)
        na,ma=P.merge_count(genuine+uncertain) if genuine or uncertain else (0,0)
        row=dict(name=name,gate_cells=len(gates),raw_arcs=len(raw),refined_pieces=len(pieces),counts=counter,
            signed_flux_by_label=fluxes,side_validation_forwards=evals,genuine_unique_planes=ng,
            genuine_geometric_merged=mg,genuine_plus_uncertain_geometric_merged=ma,
            optimistic_compression_from_genuine=len(gates)/max(mg,1),
            compression_including_uncertain=len(gates)/max(ma,1),pieces=records)
        report['cases'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='pieces'}),flush=True)
        if time.process_time()-started>300:raise RuntimeError('CPU cap')
    report.update(cpu_seconds=time.process_time()-started,wall_seconds=time.monotonic()-wall,
        geometry_counts=P.COUNTS,total_side_validation_forwards=sum(r['side_validation_forwards'] for r in report['cases']))
    Path(__file__).with_name('boundary_audit.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
