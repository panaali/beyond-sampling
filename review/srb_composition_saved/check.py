"""Saved composition audit. No author imports, estimator/Q or RNG calls."""
from pathlib import Path
from fractions import Fraction as F
import ast,hashlib,json,math,struct,sys,time,traceback
H=Path(__file__).resolve().parent
A=Path('/workspace/work/recover-small_h2_34/review/srb_cost_review/composition')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(b):return hashlib.sha256(b).hexdigest()
def pack(vals,kind):return struct.pack('<'+('d' if kind=='float64' else 'q')*len(vals),*vals)
def arr(vals,shape,kind):
    out=dict(shape=shape,dtype=kind,sha256=digest(pack(vals,kind)))
    if len(vals)<=10000:out['values']=[float(v).hex() for v in vals] if kind=='float64' else vals
    return out
def decode(d):
    assert d['dtype'] in ['float64','int64'] and len(d['values'])==math.prod(d['shape'])
    vals=[float.fromhex(v) for v in d['values']] if d['dtype']=='float64' else d['values']
    assert all(math.isfinite(v) for v in vals)
    assert arr(vals,d['shape'],d['dtype'])==d
    return vals
def main():
    start=time.process_time();assert sys.byteorder=='little'
    pins=read(H/'pins.json')
    for p,h in pins.items():assert sha(p)==h,p
    proto=read(A/'protocol.json');gate=read(A/'gate.json');raw=read(A/'results.json');trees=read(A/'trees.json')
    for p,h in {**proto['pins'],**gate['pins']}.items():assert sha(p)==h,p
    assert raw['complete'] and gate['complete'] and raw['protocol_sha256']==gate['protocol_sha256']==sha(A/'protocol.json')
    assert raw['gate_sha256']==sha(A/'gate.json')
    source=ast.parse(Path(proto['cv_source']).read_text());fn={'pstratum','pc','draw_all','make'};constants={'T','EDGES','NS','_pc','COST_DRAW'}
    nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in fn or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)]
    assert ast.dump(ast.Module(body=nodes,type_ignores=[]))==ast.dump(ast.parse(trees['composition']))
    qs=ast.parse(Path(proto['q_source']).read_text());assert ast.dump(ast.Module(body=[n for n in qs.body if isinstance(n,ast.FunctionDef)],type_ignores=[]))==ast.dump(ast.parse(trees['Q']))
    hs=ast.parse(Path(proto['harness_source']).read_text());charged=next(n for n in hs.body if isinstance(n,ast.FunctionDef) and n.name=='_charged')
    assert ast.dump(ast.Module(body=[charged],type_ignores=[]))==ast.dump(ast.parse(trees['charged']))
    over=next(n.value for n in hs.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OVERRIDES' for t in n.targets))
    v=next(v for k,v in zip(over.keys,over.values,strict=True) if isinstance(k,ast.Constant) and k.value=='control_variate')
    assert ast.dump(v)==ast.dump(ast.parse(trees['override'],mode='eval').body)
    import numpy as np  # saved NPZ reading only; no author/evaluator imports
    with np.load(proto['precompute'],allow_pickle=False) as z:
        savedP=z['P'];assert savedP.shape==(7,) and savedP.dtype==np.float64;P=savedP.tolist();pc_keys=sorted(z.files)
    allocation=read(proto['allocation'])['plain'];assert len(allocation)==7
    setup=read(A/'fixtures.json');prototypes=setup['prototypes'];assert len(prototypes)==7 and all(len(x)==9 for x in prototypes)
    # Strict prototype strata are determined from exact binary input values; no candidate source call.
    edges=list(map(F,[0.,.01,.025,.05,.1,.2]));labels=[]
    for x in prototypes:
        top=sorted(map(F,x),reverse=True)[:3]
        g=min((top[0]-top[1])/abs(top[0]),(top[1]-top[2])/max(abs(top[1]),F(1e-12)))
        labels.append(0 if top[1]<=0 else sum(e<=g for e in edges))
    assert labels==raw['prototype_labels']==list(range(7))
    cases=setup['cases'];assert [(c['budget'],c['delayed']) for c in cases]==[(0,False),(1000,False),(1000,True),(17,False)]
    assert len(raw['rows'])==4
    progress=[json.loads(x) for x in (A/'progress.jsonl').read_text().splitlines()];expected=[dict(stage='started')];summaries=[]
    c=float(F(1.6)+F(.1));factor=float(F(3.3)/F(1.6));eps=F(1,2**53)
    for fixture,row in zip(cases,raw['rows'],strict=True):
        name=fixture['name'];assert all(row[k]==fixture[k] for k in ['name','budget','delayed']) and row['parity']
        n=math.floor(F(fixture['budget'])/F(c));nh=[max(4,math.floor(float(F(a)*n))) for a in allocation];M=sum(nh);cycles=[6,7] if fixture['delayed'] else [7];D=50000*len(cycles)
        assert row['n']==n and row['nh']==nh and row['kept']==M and row['drawn']==D
        cost=math.ceil(float(F(M)*F(c)));charged_float=float(F(cost)*F(factor));hyp=math.ceil(float(F(M)*F(float(F(3.3)+F(.1)))))
        assert row['raw_cost']==cost and row['harness_charged_hex']==charged_float.hex() and row['override_factor_hex']==factor.hex()
        assert row['hypothetical_Q_only_charge']==hyp and row['literal_extra_candidate_rule']==str(F(D-M,25)) and row['reported_draw_allowance']==str(F(M,10))
        requests=[];stream_records=[]
        for cycle in cycles:
            count,tail=divmod(50000,cycle);pattern=b''.join(pack(x,'float64') for x in prototypes[:cycle]);xb=pattern*count+b''.join(pack(x,'float64') for x in prototypes[:tail]);ib=pack(list(range(cycle)),'int64')*count+pack(list(range(tail)),'int64')
            requests.append(dict(shape=[50000,9],cycle=cycle,sha256=digest(xb),candidate_count=50000))
            stream_records.append(dict(function='pstratum',input=dict(shape=[50000,9],dtype='float64',sha256=digest(xb)),output=dict(shape=[50000],dtype='int64',sha256=digest(ib)),strata_counts=[count+int(j<tail) if j<cycle else 0 for j in range(7)]))
        a,b=row['arms'];assert a['mode']=='original' and b['mode']=='observed' and a['records']==[]
        for k in ['estimate_hex','cost','warnings','requests','cacheP']:assert a[k]==b[k],(name,k)
        assert b['requests']==requests and b['cacheP']==arr(P,[7],'float64') and b['cost']==cost
        rec=b['records'];assert len(rec)==len(cycles)+3
        assert rec[0]==dict(function='pc',keys=pc_keys,P=arr(P,[7],'float64'))
        assert rec[1:1+len(cycles)]==stream_records
        drawn=rec[-2];assert drawn==dict(function='draw_all',nh=arr(nh,[7],'int64'),got=arr(nh,[7],'int64'),drawn=D,returned=[arr(prototypes[j]*nh[j],[nh[j],9],'float64') for j in range(7)],returned_drawn=D)
        e=rec[-1];assert e['function']=='est' and e['c_hex']==c.hex() and e['n']==n and e['nh']==arr(nh,[7],'int64')
        x=[v for j in range(7) for v in prototypes[j]*nh[j]];h=[j for j in range(7) for _ in range(nh[j])];w=[float(F(P[j])/nh[j]) for j in h]
        assert e['x']==arr(x,[M,9],'float64') and e['h']==arr(h,[M],'int64') and e['w']==arr(w,[M],'float64')
        q=decode(e['q']);assert e['q']['shape']==[M] and all(0<=v<=1 for v in q)
        score=float.fromhex(e['estimate_hex']);exact=sum((F(u)*F(v) for u,v in zip(w,q,strict=True)),F(0));abstotal=sum((abs(F(u)*F(v)) for u,v in zip(w,q,strict=True)),F(0));gamma=(2*M*eps)/(1-2*M*eps)
        assert abs(F(score)-exact)<=gamma*abstotal+F(2*M,2**1074)
        assert e['estimate_hex']==b['estimate_hex'] and e['cost']==cost
        expected.append(dict(stage='case_started',name=name))
        for mode in ['original','observed']:expected.extend([dict(stage='arm_started',name=name,mode=mode),dict(stage='arm_completed',name=name,mode=mode)])
        expected.append(dict(stage='case_completed',name=name,M=M,D=D))
        summaries.append(dict(name=name,n=n,quotas=nh,kept=M,drawn=D,raw_cost=cost,harness_charged_hex=charged_float.hex(),hypothetical_Q_only=hyp,nominal_extra_rule=str(F(D-M,25)),nominal_reported_draw=str(F(M,10)),score_hex=score.hex()))
    expected.append(dict(stage='completed',sha256=sha(A/'results.json')));assert progress==expected
    assert raw['estimator_calls']==8 and raw['prototype_stratum_control_calls']==1 and raw['new_random_Gaussian_trained_MC_calls']==0 and raw['full_cost'] is False and raw['probabilistic_sampling_validation'] is False
    assert math.isfinite(raw['cpu_seconds']) and raw['cpu_seconds']>=0 and raw['maxrss_kib']<2097152
    return dict(complete=True,pins=pins,cases=summaries,progress_records=len(progress),source_AST_parity=True,author_estimator_calls=8,author_prototype_calls=1,new_author_estimator_Q_RNG_calls=0,author_cpu=raw['cpu_seconds'],review_cpu=time.process_time()-start,scope='Fixed deterministic API-stream parity/allocation/charge composition only. No Gaussian-law, Q accuracy, complete cost or historical ranking inference.',score_check='Independent exact saved w*q sum with standard finite binary64 dot rounding enclosure; no Q replay.')
if __name__=='__main__':
    assert not (H/'results.json').exists() and not (H/'failure.txt').exists()
    try:r=main()
    except BaseException:
        with (H/'failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
    with (H/'results.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ['pins','cases']}))
