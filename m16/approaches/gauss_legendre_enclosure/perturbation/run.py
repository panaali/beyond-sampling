"""Saved-only original11×3 constant perturbation contraction; no tensor calls."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
from bound import bound,pack,Ledger
HERE=Path(__file__).resolve().parent

def fraction(z):return F(int(z[0]),int(z[1]))

def constants(row):
    """Original binary affine mapping, exact distance to ideal mapped constants."""
    A=Ledger();epsilon=F(0);delta=F(0);s=F(0);items=[]
    assert len(row['source_comparison'])==row['n']
    for c,xh,wh in zip(row['source_comparison'],row['source_nodes_hex'],row['source_weights_hex']):
        # Literal source map; positive multiplication by0.5 is exact here,
        # but the source node's preceding add may round and is included.
        sx=F.from_float((float.fromhex(xh)+1.0)/2.0);sw=F.from_float(float.fromhex(wh)/2.0)
        lo,hi=map(fraction,c['ideal_node']);wl,wu=map(fraction,c['ideal_weight'])
        il=A.op('div',A.op('add',lo,F(1)),F(2));iu=A.op('div',A.op('add',hi,F(1)),F(2))
        wl=A.op('div',wl,F(2));wu=A.op('div',wu,F(2))
        assert F(0)<=sx<=F(1) and F(0)<=il<=iu<=F(1) and F(0)<=sw and F(0)<wl<=wu
        de=max(abs(A.op('sub',sx,il)),abs(A.op('sub',sx,iu)))
        dw=max(abs(A.op('sub',sw,wl)),abs(A.op('sub',sw,wu)))
        A.count.update(rational_abs=4,extremum_comparison=3,input_float_to_exact=2)
        epsilon=max(epsilon,de);delta=A.op('add',delta,dw);s=A.op('add',s,sw)
        items.append(dict(index=c['index'],source_node=sx,source_weight=sw,ideal_node=(il,iu),ideal_weight=(wl,wu),node_error_upper=de,weight_error_upper=dw))
    assert abs(s-F(1))<=delta
    return dict(order=row['n'],epsilon=epsilon,delta_L1=delta,source_weight_sum=s,items=items,counts=dict(A.count),peak_bits=A.peak,source_mapping_arithmetic=3*row['n'],ideal_weight_sum=F(1),ideal_sum_premise='Exact normalized Gaussian-rule identity from reviewed node/weight theorem, not rounded interval summation.')

def main():
    out=HERE/'results.json';progress=HERE/'progress.jsonl';assert not out.exists() and not progress.exists()
    P=json.loads((HERE/'run_protocol.json').read_text())
    for f,h in P['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    control=json.loads((HERE/'controls.json').read_text());assert control['cases']==16 and not control['failures']
    nodes=json.loads(Path(P['nodes']).read_text());assert nodes['all_bounded']
    cert=json.loads(Path(P['certificates']).read_text());original=json.loads(Path(P['original']).read_text())
    assert cert['complete'] and original['complete']
    assert [c['fixture'] for c in cert['certificates']]==P['fixture_ids']==[r['name'] for r in original['fixtures']]
    def record(x):
        with progress.open('a') as f:f.write(json.dumps(pack(x),allow_nan=False)+'\n')
    start=time.process_time();record(dict(stage='constants_attempt'))
    const=[constants(r) for r in nodes['orders']];assert [c['order'] for c in const]==[6,10,16]
    record(dict(stage='constants_complete',rows=const));rows=[];setup=[]
    for c,orig in zip(cert['certificates'],original['fixtures']):
        record(dict(stage='fixture_attempt',fixture=c['fixture']))
        R=[[F.from_float(float.fromhex(x)) for x in row] for row in orig['rays_hex']]
        A=Ledger();G=[]
        for i in range(5):
            line=[]
            for j in range(5):
                value=F(0)
                for k in range(5):value=A.op('add',value,A.op('mul',R[i][k],R[j][k]))
                line.append(value)
            G.append(line)
        assert G==[[F(x) for x in r] for r in c['gram']]
        M2=max(G[i][i] for i in range(5));a=F(c['a']);det=F(c['det'])
        assert F(0)<a<=M2 and abs(det)==abs(F(orig['exact_stored_determinant']))
        setup.append(dict(fixture=c['fixture'],M2=M2,a=a,det=det,mode=c['mode'],counts=dict(A.count),peak_bits=A.peak,certificate_review=P['certificate_review']))
        for cn in const:
            record(dict(stage='bound_attempt',fixture=c['fixture'],order=cn['order']))
            got=bound(a,M2,det,cn['epsilon'],cn['source_weight_sum'],cn['delta_L1'])
            row=dict(fixture=c['fixture'],order=cn['order'],a=a,M2=M2,det=det,result=got)
            row['below_existing_1e_10_diagnostic']=got['status']=='bounded' and got['probability_upper']<=F(1,10**10)
            rows.append(row);record(row)
    result=dict(complete=True,cases=len(rows),constants=const,exact_ray_setup=setup,rows=rows,cpu_seconds=time.process_time()-start,source_mapping_arithmetic=sum(c['source_mapping_arithmetic'] for c in const),tensor_mass_model_geometry_MC_calls=0,scope=P['scope'],inherited_costs='Exact q/Gram/determinant certificates and ideal GL enclosures reused; their construction costs remain in source artifacts. Current exact mapping/Gram/bound categories retained, validation/GCD/control/runtime not completely priced.')
    out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(cases=len(rows),bounded=sum(r['result']['status']=='bounded' for r in rows),small=sum(r['below_existing_1e_10_diagnostic'] for r in rows),cpu_seconds=result['cpu_seconds'])))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (HERE/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(phase='target',type=type(e).__name__,message=str(e)))+'\n')
        raise
