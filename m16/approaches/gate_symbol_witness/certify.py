"""Exact binary-weight recurrence certificates, independent of LP maps."""
from fractions import Fraction as F
from pathlib import Path
import json,time,hashlib
import torch
HERE=Path(__file__).parent

def matrix(a):return [[F(float(v)) for v in row] for row in a]
def exact_forward(x,Wi,Wh,Wo):
    h=[F(0)]*len(Wi);pre=[];gates=[]
    for v in x:
        z=[Wi[i][0]*v+sum(w*q for w,q in zip(Wh[i],h)) for i in range(len(Wi))]
        pre.append(z);gates.append([q>0 for q in z]);h=[max(q,F(0)) for q in z]
    return [sum(w*q for w,q in zip(row,h)) for row in Wo],pre,gates

def global_lipschitz(T,Wi,Wh,Wo):
    l=[F(0)]*len(Wi);ls=[]
    for _ in range(T):
        l=[abs(Wi[i][0])+sum(abs(w)*q for w,q in zip(Wh[i],l)) for i in range(len(Wi))];ls.append(l)
    return ls

def controls():
    Wi=[[F(1)],[F(1)]];Wh=[[F(1),F(0)],[F(0),F(0)]];Wo=[[F(1),F(-13)],[F(0),F(0)],[-F(1),F(0)],[-F(1),F(0)]]
    points=[[F(3),F(2),F(1),u] for u in (F(1,4),F(3,4))];outs=[exact_forward(x,Wi,Wh,Wo) for x in points]
    assert outs[0][0][0]==3 and outs[1][0][0]==-3
    assert outs[0][2]==outs[1][2] and all(all(z>0 for z in step) for out in outs for step in out[1])
    assert [max(range(4),key=lambda j:out[0][j]) for out in outs]==[0,1]
    Wh2=[[F(1),F(0)],[F(0),F(1)]];Wo2=[[F(1),F(0)],[F(0),F(1)],[F(0),F(0)],[F(0),F(0)]]
    null=[exact_forward(x,Wi,Wh2,Wo2)[0] for x in points];assert all(y[0]==y[1]>y[2]==y[3] for y in null)
    return dict(constructive_output0=['3','-3'],constructive_winners=[0,1],constant_tie_winners=[0,0],diagnostic_forward_passes=4)

def check_point(x,rec,Wi,Wh,Wo,expected_correct,ls):
    outputs,pre,gates=exact_forward(x,Wi,Wh,Wo);assert gates==rec['gates'],'gate word changed'
    assert all(q for z in pre for q in z),'nonstructural zero gate'
    order=rec['order'];rank_slacks=[x[a]-x[b] for a,b in zip(order,order[1:])];assert min(rank_slacks)>0
    winner=max(range(len(outputs)),key=lambda j:outputs[j]);margins=[outputs[winner]-v for j,v in enumerate(outputs) if j!=winner];assert min(margins)>0,'nonunique winning output'
    truth=rec['top'][1];assert (winner==truth)==expected_correct
    limits=[v/2 for v in rank_slacks]
    for z,L in zip(pre,ls):limits.extend(abs(q)/b for q,b in zip(z,L) if b)
    for j in range(len(outputs)):
        if j!=winner:
            denom=sum(abs(a-b)*l for a,b,l in zip(Wo[winner],Wo[j],ls[-1]));assert denom>0;limits.append((outputs[winner]-outputs[j])/denom)
    radius=min(limits)/4
    return dict(x=list(map(str,x)),winner=winner,correct=winner==truth,rank_min=str(min(rank_slacks)),gate_abs_min=str(min(abs(q) for z in pre for q in z)),winner_margin_min=str(min(margins)),outputs=list(map(str,outputs)),open_box_radius=str(radius),open_box_radius_float=float(radius),gate_constraints=sum(map(len,pre)))

def main():
    start=time.process_time();ctrl=controls();print('analytic controls passed',flush=True)
    raw=json.loads((HERE/'search_results.json').read_text());p=Path('/workspace/work/t39_sources/example_checkpoint.pth');sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==raw['model_sha256'];sd=torch.load(p,map_location='cpu',weights_only=True);Wi,Wh,Wo=[matrix(sd[k].double().numpy()) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')];ls=global_lipschitz(10,Wi,Wh,Wo)
    attempts=[];certificate=None
    for idx in raw['positive_candidates'][:8]:
        rec=raw['rows'][idx];top=rec['top'];pair=[]
        try:
            for key,correct in [('correct',True),('best_wrong',False)]:
                x=[F(round(v*2**30),2**30) for v in rec[key]['x']]
                for j in top:x[j]=F(rec['anchor'][j])
                pair.append(check_point(x,rec,Wi,Wh,Wo,correct,ls))
            assert all(pair[0]['x'][j]==pair[1]['x'][j] for j in top)
            certificate=dict(index=idx,top=top,rank_order=rec['order'],gates=rec['gates'],pair=pair,common_radius=str(min(F(v['open_box_radius']) for v in pair)));attempts.append(dict(index=idx,passed=True));break
        except AssertionError as e:attempts.append(dict(index=idx,passed=False,failure=str(e)))
    out=dict(model_sha256=sha,controls=ctrl,attempts=attempts,certificate=certificate,cpu_seconds=time.process_time()-start,scope='Exact dyadic input-space witness only. Joint open neighborhoods preserve common top3; no probability or accuracy estimate.')
    (HERE/'certificate.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(attempts=attempts,cpu_seconds=out['cpu_seconds'],radii=[p['open_box_radius_float'] for p in certificate['pair']] if certificate else None)))
if __name__=='__main__':main()
