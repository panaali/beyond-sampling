"""Frozen finite LP proposals; no numerical result is an exact certificate."""
from pathlib import Path
import json,hashlib,time,warnings
import numpy as np
from scipy.optimize import linprog,OptimizeWarning
import torch
HERE=Path(__file__).parent
MODEL=Path('/workspace/work/t39_sources/example_checkpoint.pth')
SHA='498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'
def weights():
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==SHA
    sd=torch.load(MODEL,map_location='cpu',weights_only=True)
    return tuple(sd[k].double().numpy() for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
def maps(x,Wi,Wh,Wo):
    H=len(Wi);T=len(x);A=np.zeros((H,T));rows=[];gates=[]
    for t in range(T):
        Z=Wh@A;Z[:,t]+=Wi[:,0];z=Z@x;g=z>0;gates.append(g.tolist())
        for row,on in zip(Z,g):
            n=np.linalg.norm(row)
            if n:rows.append(row*(1 if on else -1)/n)
        A=Z*g[:,None]
    return np.array(rows),Wo@A,gates
CALLS=0
def propose(x,top,order,G,L,truth,rival=None):
    global CALLS
    T=len(x);rows=list(G)
    for a,b in zip(order,order[1:]):
        r=np.zeros(T);r[a]=1;r[b]=-1;rows.append(r/np.sqrt(2))
    margins=[L[truth]-L[j] for j in range(T) if j!=truth] if rival is None else [L[rival]-L[truth]]
    for r in margins:
        norm=np.linalg.norm(r)
        if not norm:return dict(success=False,status='identical_output_margin',t=None)
        rows.append(r/norm)
    C=np.array(rows);A=np.column_stack((-C,np.ones(len(C))))
    bounds=[(-1.,x[top[2]]-1/128) for _ in range(T)]+[(0,None)]
    for j in top:bounds[j]=(x[j],x[j])
    objective=np.zeros(T+1);objective[-1]=-1;CALLS+=1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',OptimizeWarning)
        z=linprog(objective,A_ub=A,b_ub=np.zeros(len(C)),bounds=bounds,method='highs',options={'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    return dict(success=bool(z.success),status=int(z.status),t=float(z.x[-1]) if z.success else None,x=z.x[:-1].tolist() if z.success else None)
def anchors():
    for r in range(8):
        top=[r,(r+3)%10,(r+6)%10]
        for d in (1/64,1/16):
            for lo,hi in ((-.5,0),(0,.5),(.5,.75),(.75,.875)):
                x=np.zeros(10);x[top]=[1,1-d,1-2*d]
                for j,i in enumerate([i for i in range(10) if i not in top],1):x[i]=lo+(hi-lo)*j/8
                yield x,top

def main():
    start=time.process_time();Wi,Wh,Wo=weights();results=[];pairs=[]
    for idx,(x,top) in enumerate(anchors()):
        order=np.argsort(-x).tolist();assert order[:3]==top and len(set(x))==10
        G,L,gates=maps(x,Wi,Wh,Wo);truth=top[1];correct=propose(x,top,order,G,L,truth)
        wrongs=[(j,propose(x,top,order,G,L,truth,j)) for j in range(10) if j!=truth]
        eligible=[(j,z) for j,z in wrongs if z['success']]
        best=max(eligible,key=lambda jz:jz[1]['t'],default=None)
        hit=bool(correct['success'] and correct['t']>1e-7 and best and best[1]['t']>1e-7)
        rec=dict(index=idx,anchor=x.tolist(),top=top,order=order,gates=gates,correct=correct,wrong_statuses=[dict(rival=j,status=z['status'],t=z['t']) for j,z in wrongs],best_wrong=dict(rival=best[0],**best[1]) if best else None,positive_candidate=hit)
        results.append(rec)
        if hit:pairs.append(idx)
    out=dict(model_sha256=SHA,anchors=64,lp_calls=CALLS,positive_candidates=pairs,rows=results,cpu_seconds=time.process_time()-start,scope='Floating proposals only; no probability, accuracy or exact certificate.')
    (HERE/'search_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':main()
