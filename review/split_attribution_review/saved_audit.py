"""Independent scalar-loop contraction of frozen T195 arrays; no author imports."""
import hashlib,json,math,time,traceback
from pathlib import Path
from collections import defaultdict
from fractions import Fraction as F
import numpy as np
P=Path(__file__).resolve().parent; D=P/'frozen'
KEYS={'t','k','frozen','sigma','truth','nrays','dE','absdE','npieces','m_cell','m_min','s_cls','s_err','perm','a','con','mag'}
SUM=math.fsum

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def log(x):
    with (P/'saved_progress.jsonl').open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def close(a,b,path=''):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:close(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(bool,int)) or a is None:assert a==b,(path,a,b)
    else:assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(path,a,b)
def frac(rows,pred):return sum(bool(pred(r)) for r in rows)/len(rows) if rows else 0.
def total(rows,key):return SUM(float(r[key]) for r in rows)
def grouped(rows):return dict(n=len(rows),n_cls=sum(r['absdE']>1e-13 for r in rows),absdE=total(rows,'absdE'),dE=total(rows,'dE'),m_min=total(rows,'m_min'),s_err=sum(r['s_err']==1 for r in rows))
def archive(path,T,guard=False):
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)==KEYS|{'meta'}|({'guard'} if guard else set()),str(path)
        meta=json.loads(str(z['meta']));n=len(z['t']); out=[]
        assert n==meta['resolved']
        for k in KEYS:
            a=z[k];assert a.shape==((n,T) if k in {'perm','a','con','mag'} else (n,)),(path,k,a.shape)
            assert np.isnan(a).all() if guard and k in {'m_cell','m_min'} else np.isfinite(a).all(),(path,k)
        for j in range(n):
            r={k:z[k][j].tolist() for k in KEYS};perm=r['perm'];assert sorted(perm)==list(range(T))
            for k in ['t','k','frozen','sigma','truth','nrays','npieces','s_cls','s_err']:assert r[k]==int(r[k]),(path,k)
            assert 1<=r['t']<T and 0<=r['k']<8 and 0<=r['truth']<T and r['truth']==perm[1]
            ranks={input_index:rank for rank,input_index in enumerate(perm,1)}
            driver=max(range(T),key=lambda k:r['con'][k]);grad=max(range(T),key=lambda k:r['mag'][k])
            r['rank']=ranks[driver];r['gradrank']=ranks[grad];r['newest']=int(driver==r['t'])
            positive=[max(0.,x) for x in r['con']];den=max(SUM(positive),1e-300)
            r['weights']={str(q):positive[perm[q-1]]/den for q in range(1,T+1)}
            r['base']=sum(ranks[k]>=4 for k in range(int(r['t'])+1))/(int(r['t'])+1)
            if guard:assert r['dE']==r['absdE']==r['npieces']==0 and r['s_cls']==r['s_err']==-1
            out.append(r)
        gs=z['guard'].tolist() if guard else None
        if guard:assert z['guard'].ndim==2 and z['guard'].shape[1]==5 and np.isfinite(z['guard']).all()
    return meta,out,gs

def summary(meta,rows,T,ref):
    a=total(rows,'absdE');den=max(a,1e-300);cls=[r for r in rows if r['absdE']>1e-13]
    d={k:meta[k] for k in ['acc','ops','cells','sec']}
    d.update(n_splits=len(rows),abs_mass=a,net_mass=total(rows,'dE'),n_cls=len(cls),s_cls=sum(r['s_cls']==1 for r in rows),s_err=sum(r['s_err']==1 for r in rows),m_min=total(rows,'m_min'))
    by={}
    for key,rkey in [('rank','rank'),('gradrank','gradrank'),('step','t'),('neuron','k'),('newest','newest')]:
        groups=defaultdict(list)
        for r in rows:groups[str(int(r[rkey]))].append(r)
        by[key]={v:grouped(sub) for v,sub in groups.items()}
    by['rank_frac']={str(q):{'n':SUM(r['weights'][str(q)] for r in rows),'absdE':SUM(r['weights'][str(q)]*r['absdE'] for r in rows),'m_min':SUM(r['weights'][str(q)]*r['m_min'] for r in rows)} for q in range(1,T+1)}
    by['step_rank']={str(t):{str(q):{'n':sum(r['t']==t and r['rank']==q for r in rows),'absdE':SUM(r['absdE'] for r in rows if r['t']==t and r['rank']==q)} for q in range(1,T+1)} for t in sorted({int(r['t']) for r in rows})}
    d['by']=by
    d['low_rank']=dict(share_splits=frac(rows,lambda r:r['rank']>=4),share_cls_splits=frac(cls,lambda r:r['rank']>=4),share_absdE=SUM(r['absdE'] for r in rows if r['rank']>=4)/den,share_absdE_frac=SUM(v['absdE'] for q,v in by['rank_frac'].items() if int(q)>=4)/den,share_absdE_gradrank=SUM(r['absdE'] for r in rows if r['gradrank']>=4)/den,uniform_baseline=SUM(r['base'] for r in rows)/len(rows),uniform_baseline_absdE=SUM(r['base']*r['absdE'] for r in rows)/den)
    d['prunable']={f'{delta:g}':{'share':frac(rows,lambda r:r['absdE']<delta),'share_rank4':frac([r for r in rows if r['absdE']<delta],lambda r:r['rank']>=4),'sum_absdE':SUM(r['absdE'] for r in rows if r['absdE']<delta)} for delta in [1e-10,1e-8,1e-6,1e-4]}
    d['identical']=all(meta[k]==ref[k] for k in ['acc','ops','cells','resolved']);assert d['identical']
    return d

def run():
    gate=load(P/'saved_gate.json')
    for rel,h in gate['pins'].items():assert sha(P/rel)==h,rel
    assert not (P/'saved_results.json').exists() and not (P/'saved_progress.jsonl').exists()
    log({'state':'started','gate':sha(P/'saved_gate.json')});start=time.process_time()
    try:
        manifest=load(P/'manifest.json')
        for item in manifest['records']:assert sha(P.parents[1]/item['frozen'])==item['sha256'],item['frozen']
        assert (D/'estT.py').read_bytes()==(D/'dependencies/m16/approaches/estT_fix/estT.py').read_bytes()
        names=[f'H8_T{T}_s{s}' for T in [3,4] for s in range(5)]
        assert {p.stem for p in (D/'out').glob('*.npz')}==set(names)|{x+'_guard' for x in names}
        analysis=load(D/'analysis.json');assert set(analysis)==set(names)
        summaries={};pooled={3:[],4:[]};guards={}
        for name in names:
            T=int(name[4]);meta,rows,_=archive(D/'out'/f'{name}.npz',T)
            assert meta['name']==name and meta['cf'] is True
            ref=load(D/'ref'/f'{name}.json');s=summary(meta,rows,T,ref);close(s,analysis[name],name)
            gm,gr,gs=archive(D/'out'/f'{name}_guard.npz',T,True)
            assert gm['cf'] is False
            assert all(gm[k]==meta[k] for k in ['acc','ops','cells','cut','resolved','simplices'])
            for a,b in zip(rows,gr,strict=True):
                for k in KEYS-{'dE','absdE','npieces','m_cell','m_min','s_cls','s_err'}:assert a[k]==b[k],(name,k)
            guards[name]={'n':len(gs),'cum_above_one':sum(r[2]>1 for r in gs),'individual_above_one':sum(r[3]>1 for r in gs),'cumulative_only_above_one':sum(r[2]>1 and r[3]<=1 for r in gs),'max_cumulative':max((r[2] for r in gs),default=0),'scope':'sampled candidate-point ratios, not count of additional actual splits'}
            summaries[name]=s;pooled[T].extend(rows);log({'state':'model_completed','name':name,'n':len(rows)})
        aggregates={}
        report=(D/'report.md').read_text()
        for T,rows in pooled.items():
            a=total(rows,'absdE');n=len(rows)
            rank={str(q):grouped([r for r in rows if r['rank']==q]) for q in range(1,T+1)}
            step={str(t):grouped([r for r in rows if r['t']==t]) for t in range(1,T)}
            thin=[r for r in rows if 0<=r['m_cell']<1e-4]
            aggregates[str(T)]={'n':n,'absdE':a,'rank':rank,'step':step,'rank4_effect_share':SUM(r['absdE'] for r in rows if r['rank']>=4)/a,'thin_split_share':len(thin)/n,'thin_effect_share':total(thin,'absdE')/a,'own_effect_below_1e6':frac(rows,lambda r:r['absdE']<1e-6),'nonpositive_cell_masses':sum(r['m_cell']<=0 for r in rows)}
            assert f'{n} resolved splits, sum absdE {a:.4g}' in report
            assert f"splits with own absdE < 1e-6: {aggregates[str(T)]['own_effect_below_1e6']:.3f}" in report
            for q in range(1,T+1):
                sub=[r for r in rows if r['rank']==q]
                line=f"| {q} | {len(sub)} | {len(sub)/n:.3f} | {frac(sub,lambda r:r['absdE']>1e-13):.2f} | {total(sub,'absdE')/a:.3f} | {total(sub,'m_min')/total(rows,'m_min'):.3f} | {frac(sub,lambda r:r['s_err']==1):.2f} |"
                assert line in report,line
        expected={(3,s,method,tau) for s in range(5) for method in ['cap','mass'] for tau in [1e-4,1e-5,1e-6,1e-7,1e-8]}|{(3,0,'cap',0.),(4,0,'cap',1e-6)}
        pruning=[];seen=set();truth3=load(D/'ref/exact_H8_T3.json')
        for p in sorted((D/'prune').glob('*.json')):
            r=load(p);key=(r['T'],r['seed'],r['test'],r['tau']);assert key in expected and key not in seen;seen.add(key)
            assert r['H']==8 and r['flops']==2*r['ops']
            H,T=r['H'],r['T'];per=F(6000)*(T*(2*H*H+2*H)+2*H*T)/F(10*(2*16*16+2*16)+2*16*10)
            N=F(2*r['ops'])/per;close(float(N),r['N'],str(p))
            assert sum(v[0] for v in r['by_step'].values())==r['pruned'];close(SUM(v[1] for v in r['by_step'].values()),r['bound'],str(p))
            assert 0<=r['wrong_centroid']<=r['pruned']
            ref=load(D/'ref'/f'H8_T{T}_s{r["seed"]}.json');entry={'file':p.name,'record':r,'continuous_budget_exact':str(N),'recipe_delta':r['acc']-ref['acc'],'legacy_cost_ratio':r['ops']/ref['ops'],'oracle_delta':r.get('pruned_err',0)-r.get('oracle_err',0)}
            if T==3:
                truth=F(truth3[str(r['seed'])]);variance=truth*(1-truth)/N
                entry.update(reference=float(truth),sampling_variance=float(variance),point_squared_error_ratio=float((F(r['acc'])-truth)**2/variance),reported_B_squared_ratio=float(F(r['bound'])**2/variance))
            pruning.append(entry)
        assert seen==expected
        result={'gate_sha256':sha(P/'saved_gate.json'),'analysis':summaries,'pooled':aggregates,'guards':guards,'pruning':pruning,'manifest_files':len(manifest['records']),'missing_T5_attribution':True,'cpu_seconds':time.process_time()-start,'scope':'saved arithmetic and four-field estimator/reference parity; no exact geometry/mass/truth/cost or causal certification'}
        with (P/'saved_results.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
        log({'state':'completed','result_sha256':sha(P/'saved_results.json')})
    except BaseException:log({'state':'failed','traceback':traceback.format_exc(),'cpu_seconds':time.process_time()-start});raise
if __name__=='__main__':run()
