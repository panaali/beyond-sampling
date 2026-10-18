"""Independent saved partition audit via exhaustive feasible activation masks."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,combinations
from collections import Counter
import json,hashlib,resource,time,traceback,math
P=Path(__file__).resolve().parent;START=time.process_time()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pins(v):
    assert isinstance(v,dict) and v
    for p,h in v.items():assert sha(p)==h,p
def decode(x):
    if isinstance(x,dict):
        if set(x)=={'q'}:
            a,b=x['q'];assert type(a)is int and type(b)is int and b>0
            v=F(a,b);assert [v.numerator,v.denominator]==[a,b];return v
        return {k:decode(v) for k,v in x.items()}
    if isinstance(x,list):return [decode(v) for v in x]
    return x
def encode(x):
    if isinstance(x,F):return {"q":[x.numerator,x.denominator]}
    if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
    if isinstance(x,list):return [encode(v) for v in x]
    return x
def literal(a,b):return json.dumps(a,sort_keys=True)==json.dumps(encode(b),sort_keys=True)
def save(r):
    r['review_cpu']=time.process_time()-START;r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t=P/'results.tmp';t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(P/'results.json')
def cap():assert time.process_time()-START<19

def fixtures():
    I=lambda n:[[int(i==j) for j in range(n)] for i in range(n)]
    def model(wi,wo,h,rank,wh=None):return dict(H=len(h),T=len(wo),Wi=wi,Wh=I(len(h)) if wh is None else wh,Wo=wo,h=h,rank=rank)
    z=model([1],[[0],[0],[0]],[0],[2,1,0,1],[[0]])
    constant=lambda h: model([0,0],[[1,0],[0,1],[0,0]],h,[2,-1,0,1])
    xs=[('zero_readout',z,1,[(2,-1)]),('constant_second',constant([0,1]),1,[(1,-1)]),('constant_last',model([0,0],[[0,0],[0,0],[1,0]],[1,0],[2,-1,0,1]),-1,[(1,1),(2,1)]),('identical_logits',model([1],[[1],[1],[1]],[1],[2,1,0,1]),1,[(2,-1)]),('memoryless_signed',model([1,-1],[[1,0],[0,1],[0,0]],[0,0],[1,-1,0,1],[[0,0],[0,0]]),2,[(1,-2)]),('reachable_interior_roots',model([1,-1,0,0],[[1,0,0,0],[0,1,0,0],[F(3,4),F(3,4),0,0],[-1,-1,0,0]],[1,1,0,0],[2,0,0,2]),F(1,2),[(F(1,2),1),(2,-1)])]
    for name,h in [('weak_equal_first',[1,0]),('weak_equal_second',[0,1])]:
        m=constant(h);m['rank']=[2,-2,0,1];xs.append((name,m,1,[(2,-1)]))
    for name,h,root in [('jump_left',[1,2],1),('jump_right',[2,1],2)]:xs.append((name,constant(h),1,[(root,-1)]))
    return [dict(label=n,model=m,expected=dict(constant=F(c),terms=[[F(x),k] for x,k in ts]),reachability='certified synthetic prefix' if i==5 else 'extension-only',prefix=[2,-1,0] if i==5 else None) for i,(n,m,c,ts) in enumerate(xs)]
def guards():
    from copy import deepcopy
    out=[]
    for name in ['bad_shape','negative_hidden','equal_ranks','equal_indices','last_index','boolean_weight']:
        m=deepcopy(fixtures()[0]['model'])
        if name=='bad_shape':m['Wh']=[[]]
        if name=='negative_hidden':m['h']=[-1]
        if name=='equal_ranks':m['rank']=[1,1,0,1]
        if name=='equal_indices':m['rank']=[2,1,0,0]
        if name=='last_index':m['rank']=[2,1,0,2]
        if name=='boolean_weight':m['Wi']=[True]
        out.append(dict(label=name,model=m))
    return out

def point(m,h,y):
    state=[max(F(0),F(w)*y+sum((F(v)*u for v,u in zip(row,h,strict=True)),F(0))) for w,row in zip(m['Wi'],m['Wh'],strict=True)]
    logits=[sum((F(v)*u for v,u in zip(row,state,strict=True)),F(0)) for row in m['Wo']]
    winner=next(i for i,v in enumerate(logits) if v==max(logits))
    a,b,ia,ib=m['rank'];order=sorted([(F(a),ia),(F(b),ib),(y,m['T']-1)],key=lambda v:(-v[0],v[1]))
    return state,logits,winner,order[1][1]
def interior(x,l,u):return (l is None or l<x) and (u is None or x<u)
def midpoint(l,u):return (l+u)/2 if l is not None and u is not None else u-1 if u is not None else l+1 if l is not None else F(0)
def symbolic(intervals):
    # First telescope all signed endpoints, then apply symmetry to the aggregate.
    ends=Counter();constant=F(0)
    for lo,hi in intervals:
        if hi is None:constant+=1
        else:ends[hi]+=1
        if lo is not None:ends[lo]-=1
    positive=Counter()
    for x,n in ends.items():
        if x==0:constant+=F(n,2)
        elif x<0:constant+=n;positive[-x]-=n
        else:positive[x]+=n
    return dict(constant=constant,terms=[[x,n] for x,n in sorted(positive.items()) if n])

def audit_case(f,row,r):
    m=f['model'];H=m['H'];T=m['T'];a=row['answer'];c=[sum((F(v)*u for v,u in zip(w,m['h'],strict=True)),F(0)) for w in m['Wh']]
    assert a['c']==c
    # Enumerate every mask and intersect its strict sign inequalities. Identically
    # zero forms must be OFF. This independently recovers every nonempty gate cell.
    pieces=[]
    for mask in product([0,1],repeat=H):
        lo=hi=None;feasible=True
        for w,z,on in zip(m['Wi'],c,mask,strict=True):
            if w==0:
                if on!=int(z>0):feasible=False;break
            else:
                root=-z/F(w)
                lower=(w>0)==bool(on)
                if lower:lo=root if lo is None else max(lo,root)
                else:hi=root if hi is None else min(hi,root)
        if not feasible or (lo is not None and hi is not None and lo>=hi):continue
        intercept=[sum((F(v)*on*z for v,on,z in zip(w,mask,c,strict=True)),F(0)) for w in m['Wo']]
        slope=[sum((F(v)*on*wi for v,on,wi in zip(w,mask,m['Wi'],strict=True)),F(0)) for w in m['Wo']]
        roots=[];ties=[]
        for j,k in combinations(range(T),2):
            dc=intercept[j]-intercept[k];ds=slope[j]-slope[k]
            if ds and interior(-dc/ds,lo,hi):roots.append(-dc/ds)
            elif not dc and not ds:ties.append([j,k])
        pieces.append((dict(l=lo,u=hi,mask=list(mask),intercept=intercept,slope=slope,roots=sorted(set(roots)),permanent_ties=ties),len(roots)))
    pieces.sort(key=lambda z:(z[0]['l'] is not None,z[0]['l'] or F(0)))
    assert [p for p,n in pieces]==a['gate_pieces']
    gr=sorted({-z/F(w) for w,z in zip(m['Wi'],c,strict=True) if w});assert gr==a['gate_roots']
    roots=sorted(set(gr+[F(v) for v in m['rank'][:2]]+[x for p,n in pieces for x in p['roots']]))
    assert roots==a['roots'] and len(roots)<=H+2+(H+1)*T*(T-1)//2
    expected=[];correct=[];refs=[];bounds=[None]+roots+[None]
    for lo,hi in zip(bounds,bounds[1:]):
        y=midpoint(lo,hi);hidden,logits,pred,truth=point(m,m['h'],y)
        expected.append(dict(l=lo,u=hi,y=y,logits=logits,prediction=pred,truth=truth,correct=pred==truth))
        if pred==truth:correct.append([lo,hi])
        refs.append(dict(kind='interval',y=y,hidden=hidden,logits=logits,prediction=pred,truth=truth))
    for y in roots:
        hidden,logits,pred,truth=point(m,m['h'],y);refs.append(dict(kind='boundary',y=y,hidden=hidden,logits=logits,prediction=pred,truth=truth))
        for p,n in pieces:
            if (p['l'] is None or p['l']<=y) and (p['u'] is None or y<=p['u']):assert [z+w*y for z,w in zip(p['intercept'],p['slope'],strict=True)]==logits
    assert a['cells']==expected and a['correct_intervals']==correct and row['references']==refs
    expr=symbolic(correct);assert expr==a['symbolic']==f['expected']
    counts=dict(gate_cells=len(pieces),pair_root_occurrences=sum(n for p,n in pieces),finite_breakpoints=len(roots),intervals=len(expected),constant_setup_dot_terms=H*H,affine_readout_dot_terms=2*len(pieces)*T*H,pair_checks=len(pieces)*T*(T-1)//2)
    assert a['counts']==counts
    pr=[]
    if f['prefix'] is not None:
        h=[F(0)]*H
        for y in f['prefix']:
            h,logits,pred,_=point(m,h,F(y));pr.append(dict(y=y,hidden=h,logits=logits,prediction=pred))
        assert h==m['h'];indices=sorted(range(len(f['prefix'])),key=lambda j:(-f['prefix'][j],j));j,k=indices[:2];assert m['rank']==[f['prefix'][j],f['prefix'][k],j,k]
    assert row['prefix_references']==pr
    r['point_checks']+=len(refs);r['prefix_checks']+=len(pr);r['enumerated_masks']+=2**H
    return dict(label=f['label'],gate_cells=len(pieces),roots=len(roots),intervals=len(expected),direct_points=len(refs),prefix_steps=len(pr),reachability=f['reachability'],symbolic_constant=str(expr['constant']),symbolic_terms=[[str(v),n] for v,n in expr['terms']])

def run(r):
    s=read(P/'scope.json');g=read(P/'gate.json');d=read(P/'completed_inputs.json');rev=read(g['source_review'])
    assert g['ready'] is True and g['review_publication'] and g['execution_publication'] and d['raw_commit'] and d['publication']
    own={str(P/n) for n in ['check.py','scope.json','README.md']}
    assert rev['complete'] is True and rev['pass_all'] is True and rev['execution_authorized'] is True and own<=set(rev['pins'])
    assert own|{str(P/'completed_inputs.json'),g['source_review']}<=set(g['pins'])
    for v in [s['pins'],g['pins'],d['pins'],rev['pins']]:pins(v)
    assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,)*2
    a=Path(s['author']);ap=read(a/'protocol.json');ag=read(a/'gate.json');ar=read(ap['source_review']);wr=read(ap['written_review'])
    required={str(a/n) for n in ['primitive.py','fixtures.py','run.py','launch.sh','PROTOCOL.md','README.md','protocol.json','gate.json','output/results.json','output/progress.jsonl','run1.log']}|{ap['source_review'],ap['written_review']}
    assert required<=set(d['pins']) and ag['ready'] is True
    assert ar['complete'] and ar['pass_all'] and ar['execution_authorized'] and wr['complete'] and wr['pass_all'] and wr['execution_allowed'] is False
    for v in [ap['pins'],ag['pins'],ar['pins'],wr['pins']]:pins(v)
    raw=read(a/'output/results.json');x=decode(raw);assert x['complete'] is True and x['pass_all'] is True
    assert x['pins']==ag['pins'] and x['protocol_pins']==ap['pins'] and x['limits']==dict(cpu=[16,20],address_space=[2**31]*2)
    assert math.isfinite(x['cpu_seconds']) and 0<=x['cpu_seconds']<15 and type(x['maxrss_kib'])is int and x['maxrss_kib']>0
    assert [x[k] for k in ['partition_attempts','partition_returns','guard_attempts','prefix_steps','cdf_calls','trained_calls','rng_calls']]==[10,10,6,3,0,0,0]
    assert len(x['rows'])==16 and x['events']==34
    r.update(rows=[],point_checks=0,prefix_checks=0,enumerated_masks=0);events=[dict(kind='started')]
    for i,(f,row) in enumerate(zip(fixtures(),x['rows'][:10],strict=True)):
        assert literal(raw['rows'][i]['fixture'],f)
        cap();assert row['label']==f['label'] and row['fixture']==f and row['status']=='passed'
        r['rows'].append(audit_case(f,row,r));events += [dict(kind=k,label=f['label']) for k in ['started_case','completed_case']];save(r)
    reasons=['matrix shape','negative hidden state','strict ranks/distinct prefix indices','strict ranks/distinct prefix indices','strict ranks/distinct prefix indices','exact int/Fraction required']
    for i,(f,row,reason) in enumerate(zip(guards(),x['rows'][10:],reasons,strict=True),10):
        assert literal(raw['rows'][i]['fixture'],f)
        assert row['label']==f['label'] and row['fixture']==f and row['status']=='passed' and row['rejection']==reason
        events += [dict(kind=k,label=f['label']) for k in ['started_guard','completed_guard']];r['rows'].append(dict(label=f['label'],malformed_condition_verified=True,rejection=reason));save(r)
    events.append(dict(kind='completed'));assert [json.loads(v) for v in (a/'output/progress.jsonl').read_text().splitlines()]==events and len(events)==34
    assert r['point_checks']==x['point_forwards']<=400 and r['prefix_checks']==3
    assert x['rows'][6]['answer']['symbolic']==x['rows'][7]['answer']['symbolic']
    assert x['rows'][8]['answer']['symbolic']['terms']==[[F(1),-1]] and x['rows'][9]['answer']['symbolic']['terms']==[[F(2),-1]]
    cap();r.update(complete=True,pass_all=True,total_rows=16,events=34,author_cpu=x['cpu_seconds'],author_calls=0,cdf_trained_rng_calls=0,scope_sha256=sha(P/'scope.json'),input_pins=d['pins'],limitations='Ten fixed rational extension states; only one synthetic prefix certified reachable. Symbolic equality sufficient, unequal forms not generally probability inequality. Jump1/2 strict separation additionally uses Gaussian CDF monotonicity. Guard records checked against malformed fixtures without author rejection calls; selected count formulas only, not full price. No trained merge, numerical CDF, probability bound or compression result.')
if __name__=='__main__':
    assert not(P/'results.json').exists() and not(P/'results.tmp').exists()
    r=dict(complete=False,pass_all=False,rows=[]);save(r)
    try:run(r)
    except BaseException as e:r.update(complete=False,pass_all=False,error=repr(e),traceback=traceback.format_exc());raise
    finally:save(r)
