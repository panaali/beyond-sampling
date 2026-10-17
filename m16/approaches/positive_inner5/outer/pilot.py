"""Fixed weight normalization and one identity cone, reused in full phase."""
from pathlib import Path
import json,time,math
from fractions import Fraction as F
from common import load,rules,fixtures,one,HERE,coordinate_rays,json_safe

def main():
    dest=HERE/'pilot.json';progress=HERE/'pilot_progress.jsonl'
    assert not dest.exists() and not progress.exists()
    def record(row):
        with progress.open('a') as fh:fh.write(json.dumps(json_safe(row),allow_nan=False)+'\n')
    record({'stage':'source_and_mapping_attempt'})
    p=load();rs,setup=rules(p);checks=[]
    manifest=fixtures(p);mapping=[]
    # Deliberately nonsymmetric saved rays: exact dyadic input linear map.
    lam=[F(1,4),F(3,8),F(3,16),F(3,64),F(9,64)]
    for item in manifest:
        raw=[[F(float.fromhex(x)) for x in row] for row in item['rays_hex']]
        R=coordinate_rays(raw)
        candidate=[sum(R[j][i]*lam[i] for i in range(5)) for j in range(5)]
        original_lam=[F(0)]*5
        for i,k in enumerate([2,3,4,0,1]):original_lam[k]=lam[i]
        expected=[sum(original_lam[i]*raw[i][j] for i in range(5)) for j in range(5)]
        mapping.append({'id':item['id'],'pass':candidate==expected,
                        'nonsymmetric':raw!=list(map(list,zip(*raw)))})
        record({'stage':'mapping_result','result':mapping[-1]})
    assert all(r['pass'] for r in mapping) and any(r['nonsymmetric'] for r in mapping)
    record({'stage':'normalization_attempt'})
    for n,(x,w) in rs.items():
        # Three independent 1D factors of the exact Duffy Jacobian integral.
        value=math.prod(math.fsum(wi*(1-xi)**degree for xi,wi in zip(x,w)) for degree in [3,2,1])
        checks.append({'order':n,'jacobian_integral':value,'expected':1/24,'pass':abs(value-1/24)<=1e-14})
        record({'stage':'normalization_result','result':checks[-1]})
    assert all(c['pass'] for c in checks)
    item=fixtures(p)[0];assert item['id']=='original/positive5orthant'
    record({'stage':'attempt','id':item['id'],'order':16})
    row=one(item,16,rs[16])
    result={'exact_generator_mapping':mapping,'normalization':checks,'setup':setup,'row':row,
            'pass':row['complete'] and row.get('finite_positive',False) and abs(row['value']-1/32)<=1e-10,
            'normalization_check_cost':'diagnostic-only, separate from candidate named ledger',
            'scope':'single analytic pilot; source floating and native setup remain uncertified'}
    dest.write_text(json.dumps(json_safe(result),indent=2,allow_nan=False)+'\n')
    with progress.open('a') as f:f.write(json.dumps(json_safe({'stage':'result','result':result}),allow_nan=False)+'\n')
    print(json.dumps({'pass':result['pass'],'value':row.get('value'),'cpu':row['cpu']}))
    assert result['pass']
if __name__=='__main__':main()
