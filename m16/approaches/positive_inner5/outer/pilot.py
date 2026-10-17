"""Fixed weight normalization and one identity cone, reused in full phase."""
from pathlib import Path
import json,time,math
from common import load,rules,fixtures,one,HERE

def main():
    dest=HERE/'pilot.json';progress=HERE/'pilot_progress.jsonl'
    assert not dest.exists() and not progress.exists()
    p=load();rs,setup=rules(p);checks=[]
    for n,(x,w) in rs.items():
        # Three independent 1D factors of the exact Duffy Jacobian integral.
        value=math.prod(math.fsum(wi*(1-xi)**degree for xi,wi in zip(x,w)) for degree in [3,2,1])
        checks.append({'order':n,'jacobian_integral':value,'expected':1/24,'pass':abs(value-1/24)<=1e-14})
    assert all(c['pass'] for c in checks)
    item=fixtures(p)[0];assert item['id']=='original/positive5orthant'
    progress.write_text(json.dumps({'stage':'attempt','id':item['id'],'order':16})+'\n')
    row=one(item,16,rs[16])
    result={'normalization':checks,'setup':setup,'row':row,
            'pass':row['complete'] and row.get('finite_positive',False) and abs(row['value']-1/32)<=1e-10,
            'normalization_check_cost':'diagnostic-only, separate from candidate named ledger',
            'scope':'single analytic pilot; source floating and native setup remain uncertified'}
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with progress.open('a') as f:f.write(json.dumps({'stage':'result','result':result},allow_nan=False)+'\n')
    print(json.dumps({'pass':result['pass'],'value':row.get('value'),'cpu':row['cpu']}))
    assert result['pass']
if __name__=='__main__':main()
