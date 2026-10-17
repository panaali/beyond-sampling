"""Exact saved split-child constant bounds; no quadrature or model imports."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import sys,json,hashlib,time
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent))
from bound import bound,pack,Ledger

def det_exact(R,A):
    B=[r[:] for r in R];v=F(1)
    for k in range(5):
        j=next((j for j in range(k,5) if B[j][k]),None)
        if j is None:return F(0)
        if j!=k:B[k],B[j]=B[j],B[k];v=-v;A.count['sign_negation']+=1
        pivot=B[k][k];v=A.op('mul',v,pivot)
        for i in range(k+1,5):
            ratio=A.op('div',B[i][k],pivot)
            for j in range(k+1,5):B[i][j]=A.op('sub',B[i][j],A.op('mul',ratio,B[k][j]))
    return v

def main():
    dest=H/'results.json';progress=H/'progress.jsonl';assert not dest.exists() and not progress.exists()
    P=json.loads((H/'protocol.json').read_text())
    for p,h in P['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    original=json.loads(Path(P['constant_source']).read_text());saved=json.loads(Path(P['children_source']).read_text())
    assert original['complete'] and saved['complete'] and len(saved['children'])==22
    constants=original['constants'];assert [c['order'] for c in constants]==[6,10,16]
    def rat(v):return F(int(v[0]),int(v[1]))
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(pack(r),allow_nan=False)+'\n')
    start=time.process_time();rows=[];setups=[];keys=set()
    for child in saved['children']:
        key=(child['fixture'],child['side']);assert key not in keys;keys.add(key)
        record(dict(stage='child_attempt',fixture=key[0],side=key[1]))
        R=[[F.from_float(float.fromhex(x)) for x in r] for r in child['rays_hex']];A=Ledger()
        G=[[sum((A.op('mul',R[i][k],R[j][k]) for k in range(5)),F(0)) for j in range(5)] for i in range(5)]
        A.count['add']+=25*5 # conservative explicit zero-start sum allowance
        c=child['certificate'];assert c['mode']=='acute'
        assert G==[[F(x) for x in r] for r in c['gram']]
        assert all(x>=0 for row in G for x in row)
        mn=min(G[i][i] for i in range(5));M2=max(G[i][i] for i in range(5));a=A.op('div',mn,F(5))
        assert a==F(c['a']) and a>0 and M2==F(c['b'])
        det=det_exact(R,A);assert det==F(c['det']) and det!=0
        setups.append(dict(fixture=key[0],side=key[1],a=a,M2=M2,det=det,counts=dict(A.count),peak_bits=A.peak))
        for cn in constants:
            record(dict(stage='bound_attempt',fixture=key[0],side=key[1],order=cn['order']))
            got=bound(a,M2,det,rat(cn['epsilon']),rat(cn['source_weight_sum']),rat(cn['delta_L1']))
            row=dict(fixture=key[0],side=key[1],order=cn['order'],result=got);rows.append(row);record(row)
    assert len(rows)==66
    groups=[]
    for fixture in P['fixtures']:
        for n in [6,10,16]:
            rs=[r for r in rows if r['fixture']==fixture and r['order']==n];assert sorted(r['side'] for r in rs)==[0,1]
            good=all(r['result']['status']=='bounded' for r in rs)
            upper=sum((r['result']['probability_upper'] for r in rs),F(0)) if good else None
            old=next(r for r in original['rows'] if r['fixture']==fixture and r['order']==n)
            g=dict(fixture=fixture,order=n,status='bounded' if good else 'unknown',upper=upper,below_existing_1e_10=good and upper<=F(1,10**10),original_upper=old['result']['probability_upper'])
            groups.append(g);record(dict(stage='parent',row=g))
    result=dict(complete=True,rows=rows,groups=groups,exact_setup=setups,constants_reused=True,source_constants_new_mapping_ops=0,exact_parent_sum_additions_allowance=66,cpu_seconds=time.process_time()-start,tensor_mass_model_geometry_MC_calls=0,scope=P['scope'],cost_qualification='Named exact categories, not native float units. Validation comparisons, pivot search, Fraction/GCD/isqrt/runtime not fully priced; inherited constant/certificate construction remains separate. Gram sum allowance conservatively includes initial zero addition.')
    dest.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(rows=len(rows),groups=len(groups),bounded=sum(g['status']=='bounded' for g in groups),small=sum(g['below_existing_1e_10'] for g in groups),cpu_seconds=result['cpu_seconds'])))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
