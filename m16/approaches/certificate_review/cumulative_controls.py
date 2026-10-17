from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json,time
from cumulative import allowance,decision,final_cell
from simultaneous import serial

def trace(model,x,at=None,g=None):
    wi,wh,wo=model;h=[F(0)]*len(wi);zs=[]
    for t,u in enumerate(x):
        z=[sum(a*b for a,b in zip(row,h))+v*u for row,v in zip(wh,wi)];zs.append(z)
        h=[v*int(on) for v,on in zip(z,g)] if t==at else [max(F(0),v) for v in z]
    out=[sum(a*b for a,b in zip(row,h)) for row in wo]
    return zs,out,max(range(len(out)),key=lambda i:out[i])

def main():
    start=time.process_time();wi=(F(1),F(-1,2),F(1,3));wh=((F(1,2),F(-1,4),F(3,4)),(F(-1,2),F(1,4),F(1,8)),(F(1,4),F(-3,4),F(1,2)));wo=((F(1),F(-1),F(1,2)),(F(-1,3),F(2),F(0)),(F(0),F(-1,2),F(1)),(F(1,4),F(0),F(-1)));model=(wi,wh,wo)
    inputs=[tuple(map(F,x)) for x in [(1,2,3,4),(-1,2,-3,4),(1,-2,3,-4),(-2,-1,1,2),(2,1,-1,-2),(F(1,3),F(2,3),F(-1,3),F(4,3))]]
    scale=(F(2),F(1,3),F(5));scaled=(tuple(a*b for a,b in zip(wi,scale)),tuple(tuple(wh[i][j]*scale[i]/scale[j] for j in range(3)) for i in range(3)),tuple(tuple(v/c for v,c in zip(row,scale)) for row in wo));counts={k:0 for k in ('correct','wrong','unknown')};checks=0
    for x in inputs:
        zs,out,winner=trace(model,x);zs2,out2,winner2=trace(scaled,x);assert out2==out and winner2==winner
        for t in range(4):
            for g in product((0,1),repeat=3):
                _,frozen,_=trace(model,x,t,g);_,frozen2,_=trace(scaled,x,t,g);assert frozen2==frozen
                for truth in range(4):
                    errors=allowance(wh,wo,zs[t],g,truth,3-t);other=allowance(scaled[1],scaled[2],zs2[t],g,truth,3-t);assert errors==other
                    fm=[frozen[truth]-v for v in frozen];actual=[out[truth]-v for v in out];assert all(abs(a-b)<=e for a,b,e in zip(actual,fm,errors))
                    status=decision(fm,errors,truth);counts[status]+=1;checks+=1
                    if status=='correct':assert winner==truth
                    elif status=='wrong':assert winner!=truth
    tie=[([0,0],[0,0],0,'correct'),([0,0],[0,0],1,'wrong'),([0,0],[1,0],1,'unknown'),([0,-1],[0,1],0,'unknown'),([0,-2],[0,1],0,'wrong')]
    for margins,errors,label,want in tie:assert decision(margins,errors,label)==want
    original=json.loads((Path(__file__).parent/'simultaneous.json').read_text());cells=[]
    for case in original['cases'][:2]:
        A=[list(map(F,r)) for r in case['final_forms']];readout=[list(map(F,r)) for r in case['model'][2]];rays=[list(map(F,r)) for r in case['rays']];result=final_cell(A,readout,(0,0,1),2,rays);want='unknown' if case['name']=='two_gates' else 'correct';assert result['status']==want;cells.append(dict(name=case['name'],result=result))
    family=json.loads((Path(__file__).parent/'family_control.json').read_text());result=final_cell([list(map(F,r)) for r in family['final_forms']],[list(map(F,r)) for r in family['model'][2]],(0,0,1),2,[list(map(F,r)) for r in family['rays']]);assert result['status']=='unknown';cells.append(dict(name='one_initial_family',result=result))
    # Different competitors can beat truth at different vertices while truth wins inside.
    A=((F(1),F(0)),(F(0),F(1)));readout=((F(0),F(0)),(F(1),F(-3,4)),(F(-1),F(1,4)));rays=((F(0),F(1)),(F(1),F(1)));result=final_cell(A,readout,(1,1),0,rays);assert result['status']=='unknown';cells.append(dict(name='wrong_competitor_switch',result=result))
    out=dict(pointwise_label_checks=checks,status_counts=counts,scaling_checks=checks,exact_true_traces=12,exact_frozen_traces=384,tie_controls=tie,final_cell_controls=cells,cpu_seconds=time.process_time()-start,scope='Cumulative allowance rigorous pointwise for all remaining depths; vertex extension only final affine readout and a fixed wrong competitor.')
    Path(__file__).with_suffix('.json').write_text(json.dumps(serial(out),indent=2)+'\n');print(json.dumps(dict(checks=checks,counts=counts,cpu_seconds=out['cpu_seconds'])))
if __name__=='__main__':main()
