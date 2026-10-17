"""A frozen downstream ReLU margin need not be concave on an input cell."""
from fractions import Fraction as F
from pathlib import Path
import json,time
from simultaneous import serial

def margin(v,w):
    hidden=[max(F(0),v-w/2),max(F(0),w/2-v),max(F(0),w)]
    return hidden[0]+hidden[1]-hidden[2]/5

def main():
    start=time.process_time();points=[(F(0),F(1)),(F(1),F(1)),(F(1,2),F(1))];values=[margin(*p) for p in points];assert values==[F(3,10),F(3,10),F(-1,5)]
    allowance=F(1,20);lower=[v-allowance for v in values];assert min(lower[:2])>0>lower[2]
    assert values[2]<(values[0]+values[1])/2
    # Independently checked interior witness in the initial-half control.
    data=json.loads((Path(__file__).parent/'family_control.json').read_text());rays=[list(map(F,r)) for r in data['rays']];weights=(1,100,1);x=tuple(sum(w*r[j] for w,r in zip(weights,rays)) for j in range(3))
    from simultaneous import recurrence
    model=tuple(tuple(map(F,part)) if i==0 else tuple(tuple(map(F,r)) for r in part) for i,part in enumerate(data['model']));actual=recurrence(model,x);frozen=recurrence(model,x,(0,0,1));a=actual['preactivations'][-1][:2]
    assert actual['margin']==F(-4347,500) and frozen['margin']==F(68977,2000) and a==[F(391,20),F(9453,400)]
    assert all(v>0 for v in a) and max(a)<frozen['margin']<sum(a)
    out=dict(downstream_map='ReLU(v−w/2)+ReLU(w/2−v)−ReLU(w)/5',points=points,margins=values,constant_homogeneous_allowance=allowance,lower_margin_bounds=lower,scope='Analytic downstream-map nonconcavity only; not asserted reachable in the same synthetic recurrent model.',family_interior=dict(weights=weights,input=x,actual=actual,frozen=frozen,individual_allowances=a),cpu_seconds=time.process_time()-start)
    Path(__file__).with_suffix('.json').write_text(json.dumps(serial(out),indent=2)+'\n');print(out['cpu_seconds'])
if __name__=='__main__':main()
