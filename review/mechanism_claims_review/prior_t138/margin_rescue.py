"""Exact retained affine coordinate on one certified reachable gate cell."""
from fractions import Fraction as F
from pathlib import Path
import json,time,hashlib
import torch
from certify import matrix,exact_forward
HERE=Path(__file__).parent

def prescribed(x,Wi,Wh,Wo,gates):
    h=[F(0)]*len(Wi)
    for value,gate in zip(x,gates):
        h=[(Wi[i][0]*value+sum(a*b for a,b in zip(Wh[i],h))) if gate[i] else F(0) for i in range(len(Wi))]
    return [sum(a*b for a,b in zip(row,h)) for row in Wo]

def main():
    start=time.process_time();cert=json.loads((HERE/'certificate.json').read_text());c=cert['certificate'];p=Path('/workspace/work/t39_sources/example_checkpoint.pth');assert hashlib.sha256(p.read_bytes()).hexdigest()==cert['model_sha256'];sd=torch.load(p,map_location='cpu',weights_only=True);Wi,Wh,Wo=[matrix(sd[k].double().numpy()) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')]
    T=len(Wo);columns=[]
    for i in range(T):
        x=[F(int(i==j)) for j in range(T)];columns.append(prescribed(x,Wi,Wh,Wo,c['gates']))
    first,second=c['pair'];a,b=first['winner'],second['winner'];x0=list(map(F,first['x']));x1=list(map(F,second['x']));coeff=[col[a]-col[b] for col in columns]
    dot=lambda x:sum(q*v for q,v in zip(coeff,x));m0,m1=dot(x0),dot(x1);assert m0>0>m1
    lam=m0/(m0-m1);assert 0<lam<1;xstar=[v+lam*(w-v) for v,w in zip(x0,x1)];ystar,zstar,gstar=exact_forward(xstar,Wi,Wh,Wo);assert gstar==c['gates'] and all(q for row in zstar for q in row)
    assert ystar[a]==ystar[b];other=[ystar[a]-v for j,v in enumerate(ystar) if j not in (a,b)];assert min(other)>0
    for x,stored in ((x0,first),(x1,second)):
        predicted=[sum(col[j]*v for col,v in zip(columns,x)) for j in range(T)];assert predicted==list(map(F,stored['outputs']))
    low=[i for i in range(T) if i not in c['top']];fixed=sum(coeff[i]*x0[i] for i in c['top']);assert all(coeff[i] for i in low)
    out=dict(anchor_index=c['index'],truth=a,competitor=b,top=c['top'],low_indices=low,margin_coefficients=list(map(str,coeff)),margin_coefficients_float=list(map(float,coeff)),fixed_top_contribution=str(fixed),endpoint_margins=[str(m0),str(m1)],crossing_parameter=str(lam),crossing_parameter_float=float(lam),crossing_input=list(map(str,xstar)),other_output_min_margin=str(min(other)),other_output_min_margin_float=float(min(other)),gate_constraints=160,rank_constraints=9,prescribed_basis_evaluations=10,actual_crossing_evaluations=1,cpu_seconds=time.process_time()-start,scope='Exact fixed-cell affine margin; one crossing on this segment by convexity/linearity, no global state compression or accuracy integration.')
    (HERE/'margin_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('low_indices','margin_coefficients_float','crossing_parameter_float','other_output_min_margin_float','cpu_seconds')}))
if __name__=='__main__':main()
