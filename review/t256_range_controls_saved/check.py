"""Independent exact determinant/vertex audit; no author code execution."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations,permutations
from collections import Counter
from math import comb,isfinite
import hashlib,json,resource,time,traceback
P=Path(__file__).resolve().parent
START=time.process_time()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pins(v):
    assert v
    for p,h in v.items():assert sha(p)==h,p
def q(x):
    assert type(x)in(int,str),('unexpected saved rational type',type(x))
    return F(x)
def vector(xs):return list(map(q,xs))
def save(r):
    r['review_cpu']=time.process_time()-START;r['review_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t=P/'results.tmp';t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(P/'results.json')
def cap():assert time.process_time()-START<19,'19 CPU reviewer cap'
def fixtures():
    cases=[
    dict(name='separated',box=[[3,4],[1,2],[-2,-1]],coeff=[1,2,-1],constant=1,leader=0,second=1,gap=1,expected=[7,11]),
    dict(name='overlap_rank',box=[[0,3]]*3,coeff=[-1,2,1],constant=0,leader=0,second=1,gap=1,expected=[-1,2]),
    dict(name='interior_max_kink',box=[[1,4],[0,3],[-2,0]],coeff=[0,-1,2],constant=0,leader=0,second=1,gap=1,expected=[-7,-1],extremum_t=['max',1]),
    dict(name='interior_min_kink',box=[[2,5],[-1,3],[-3,-1]],coeff=[2,-1,0],constant=0,leader=0,second=1,gap=1,expected=[3,11],extremum_t=['min',1]),
    dict(name='mixed_four',box=[[2,6],[-1,4],[-4,0],[-3,1]],coeff=[-2,1,3,-1],constant=2,leader=0,second=1,gap='1/2'),
    dict(name='zero_objective',box=[[2,6],[-1,4],[-4,0],[-3,1]],coeff=[0,0,0,0],constant=7,leader=0,second=1,gap='1/2',expected=[7,7]),
    dict(name='negative_coordinates',box=[[-1,2],[-3,0],[-5,-2]],coeff=[1,-2,1],constant=-3,leader=0,second=1,gap='1/2'),
    dict(name='rational_coefficients',box=[['1/3','7/3'],['-2/3','5/3'],['-5/3','1/3']],coeff=['-3/2','2/5','7/3'],constant='4/7',leader=0,second=1,gap='1/3'),
    dict(name='empty_domain',box=[[0,1]]*3,coeff=[1,-1,2],constant=0,leader=0,second=1,gap=1,expected=None),
    dict(name='touching_point',box=[[0,1]]*3,coeff=[2,-1,3],constant=1,leader=0,second=1,gap='1/2',expected=['5/2','5/2']),
    dict(name='zero_gap_rank_tie',box=[[1,1],[1,1],[0,0]],coeff=[1,2,3],constant=0,leader=0,second=1,gap=0,expected=[3,3]),
    dict(name='permuted_four',box=[[-3,1],[2,6],[-1,4],[-4,0]],coeff=[-1,-2,1,3],constant=2,leader=1,second=2,gap='1/2')]
    base=dict(box=[[0,3],[0,3],[0,3]],coeff=[1,1,1],constant=0,leader=0,second=1,gap=0)
    changes=[('two_inputs',dict(box=[[0,1],[0,1]],coeff=[1,1])),('reversed',dict(box=[[1,0],[0,1],[0,1]])),('negative_gap',dict(gap=-1)),('same_label',dict(second=0)),('outside_label',dict(leader=3)),('float_coefficient',dict(coeff=[1.0,1,1])),('short_coefficients',dict(coeff=[1,1])),('boolean_label',dict(second=True))]
    return cases,[(name,dict(base,**change)) for name,change in changes]

def det(rows):
    n=len(rows);total=F(0)
    for perm in permutations(range(n)):
        term=F((-1)**sum(perm[i]>perm[j] for i in range(n) for j in range(i+1,n)))
        for i,j in enumerate(perm):term*=rows[i][j]
        total+=term
    return total

def independent_vertices(box,i,j,g):
    n=len(box);planes=[]
    # Use >= rows, opposite to the author's <= constraint representation.
    for k,(lo,hi) in enumerate(box):
        a=[F(0)]*n;a[k]=1;planes.append((tuple(a),lo))
        a=[F(0)]*n;a[k]=-1;planes.append((tuple(a),-hi))
    a=[F(0)]*n;a[i]=1;a[j]=-1;planes.append((tuple(a),g))
    for k in range(n):
        if k in(i,j):continue
        a=[F(0)]*n;a[j]=1;a[k]=-1;planes.append((tuple(a),g))
    vertices=set();nonzero=0;systems=0
    for ids in combinations(range(len(planes)),n):
        cap();systems+=1;A=[planes[k][0] for k in ids];b=[planes[k][1] for k in ids];d=det(A)
        if not d:continue
        nonzero+=1
        x=tuple(det([[b[r] if c==col else A[r][c] for c in range(n)] for r in range(n)])/d for col in range(n))
        if all(sum(a*v for a,v in zip(normal,x))>=rhs for normal,rhs in planes):vertices.add(x)
    return planes,sorted(vertices),systems,nonzero

def domain(case,row):
    assert row['kind']=='domain' and row['name']==case['name'] and row['input']==case and row['status']=='passed' and row['active_call'] is None
    box=[vector(v) for v in case['box']];coeff=vector(case['coeff']);c=q(case['constant']);g=q(case['gap']);i=case['leader'];j=case['second'];n=len(box)
    planes,vs,systems,nonzero=independent_vertices(box,i,j,g)
    ref=row['vertex_reference'];assert ref['systems']==systems==comb(3*n-1,n) and ref['nonsingular']==nonzero
    # Compare the complete constraint multiset after reversing the source signs.
    saved_planes=Counter((tuple(-z for z in vector(a)),-q(b)) for a,b in ref['constraints'])
    assert saved_planes==Counter(planes)
    assert [tuple(vector(v)) for v in ref['vertices']]==vs
    values=[c+sum(a*x for a,x in zip(coeff,v)) for v in vs]
    assert vector(ref['values'])==values
    bounds=None if not values else [min(values),max(values)]
    assert (None if ref['range'] is None else vector(ref['range']))==bounds
    raw=row['physical'];other=row['v_domain'];adapt=row['saved_v_adapter'];assert adapt==raw
    assert raw['status']==other['status']==('feasible' if vs else 'empty')
    assert (None if raw['range'] is None else vector(raw['range']))==bounds
    assert other['range']==raw['range']
    if 'expected' in case:assert bounds==(None if case['expected'] is None else vector(case['expected']))
    # Projection interval from necessary lower/upper bounds on the second input.
    lo=max([box[j][0]]+[box[k][0]+g for k in range(n) if k not in(i,j)])
    hi=min(box[j][1],box[i][1]-g)
    assert vector(raw['second_interval'])==[lo,hi]
    assert [z/2 for z in vector(other['second_interval'])]==[lo,hi]
    if not vs:
        assert lo>hi and raw['candidates']==other['candidates']==[] and raw['witnesses'] is other['witnesses'] is None
        K=[]
    else:
        K=sorted({lo,hi}|{t for t in [box[i][0]-g]+[box[k][1]+g for k in range(n) if k not in(i,j)] if lo<=t<=hi})
        assert len(K)<=n+1 and [q(v['t']) for v in raw['candidates']]==K
        assert len(other['candidates'])==len(K)
        candidates=[]
        for t,record,vrecord in zip(K,raw['candidates'],other['candidates'],strict=True):
            # Feasible sliced-coordinate intervals and objective monotonicity.
            intervals=[]
            for k in range(n):
                a,b=box[k]
                if k==j:a=b=t
                elif k==i:a=max(a,t+g)
                else:b=min(b,t-g)
                assert a<=b;intervals.append((a,b))
            low=[b if coeff[k]<0 else a for k,(a,b) in enumerate(intervals)]
            high=[b if coeff[k]>0 else a for k,(a,b) in enumerate(intervals)]
            objective=[c+sum(a*x for a,x in zip(coeff,v)) for v in (low,high)]
            assert vector(record['low'])==low and vector(record['high'])==high and vector(record['values'])==objective
            for v,z in zip((low,high),objective):
                assert all(sum(a*x for a,x in zip(normal,v))>=rhs for normal,rhs in planes)
                assert bounds[0]<=z<=bounds[1]
            assert q(vrecord['t'])==2*t and vrecord['values']==record['values']
            for key in ('low','high'):assert vector(vrecord[key])==[2*z for z in vector(record[key])]
            candidates.append((low,high,objective))
        minimum=min(range(len(K)),key=lambda k:candidates[k][2][0]);maximum=max(range(len(K)),key=lambda k:candidates[k][2][1])
        witnesses=[candidates[minimum][0],candidates[maximum][1]]
        assert [vector(v) for v in raw['witnesses']]==witnesses
        assert [vector(v) for v in other['witnesses']]==[[2*z for z in v] for v in witnesses]
        assert [candidates[minimum][2][0],candidates[maximum][2][1]]==bounds
        if 'extremum_t' in case:
            side,t=case['extremum_t'];pos=0 if side=='min' else 1;t=q(t)
            assert lo<t<hi and any(t==K[k] and candidates[k][2][pos]==bounds[pos] for k in range(len(K)))
    work=dict(input_rational_values=3*n+2,ordered_pairs=1,candidate_values=len(K),endpoint_completions=2*len(K),objective_products=2*n*len(K),objective_additions=2*(n+1)*len(K))
    assert raw['work']==other['work']==adapt['work']==work
    assert raw['cost_scope']==other['cost_scope']==adapt['cost_scope'] and 'not fully priced' in raw['cost_scope']
    return dict(name=case['name'],dimension=n,systems=systems,nonsingular=nonzero,vertices=len(vs),range=None if bounds is None else list(map(str,bounds)),candidate_points=len(K),saved_work=work)

def guard(name,args,row):
    assert row['kind']=='guard' and row['name']==name and row['status']=='passed' and row['expected']=='ValueError' and row['attempted'] is True
    assert json.dumps(row['input'],sort_keys=True)==json.dumps(args,sort_keys=True)
    assert isinstance(row['error'],str) and row['error'] and 'unexpected_result' not in row
    if name=='two_inputs':assert len(args['box'])<3
    elif name=='reversed':assert any(a>b for a,b in args['box'])
    elif name=='negative_gap':assert args['gap']<0
    elif name=='same_label':assert args['leader']==args['second']
    elif name=='outside_label':assert args['leader']>=len(args['box'])
    elif name=='float_coefficient':assert any(type(a)is float for a in args['coeff'])
    elif name=='short_coefficients':assert len(args['coeff'])!=len(args['box'])
    elif name=='boolean_label':assert type(args['second'])is bool
    else:raise AssertionError(name)
    return dict(name=name,rejection_verified=True)

def run(r):
    scope=read(P/'scope.json');gate=read(P/'gate.json')
    assert gate['ready'] is True and gate['review_publication'] and gate['execution_publication']
    review=read(gate['source_review']);assert review['complete'] is True and review['pass_all'] is True and review['execution_authorized'] is True
    need={str(P/'check.py'),str(P/'scope.json')}
    assert need<=set(review['pins']) and need|{gate['source_review'],str(P/'completed_inputs.json')}<=set(gate['pins'])
    for v in [scope['pins'],gate['pins'],review['pins']]:pins(v)
    completed=read(P/'completed_inputs.json');pins(completed['pins']);assert completed['raw_commit'] and completed['publication']
    a=Path(scope['author']);required={str(a/n) for n in ['primitive.py','fixtures.py','reference.py','run.py','launch.sh','PROTOCOL.md','protocol.json','gate.json','output/results.json','output/progress.jsonl','run1.log']}
    assert required<=set(completed['pins'])
    p=read(a/'protocol.json');g=read(a/'gate.json');out=read(a/'output/results.json')
    assert g['ready'] is True and out['complete'] is True and out['pass_all'] is True and out['pins']==g['pins'] and out['protocol_pins']==p['pins']
    ar=read(p['source_review']);theorem=read(p['theorem_review'])
    assert ar['complete'] is True and ar['pass_all'] is True and ar['execution_authorized'] is True
    assert theorem['complete'] is True and theorem['pass_all'] is True
    assert p['source_review'] in completed['pins'] and p['theorem_review'] in completed['pins']
    for v in [p['pins'],g['pins'],ar['pins']]:pins(v)
    assert out['limits']==dict(cpu=[16,20],address_space=[2*1024**3]*2)
    assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2*1024**3,)*2
    assert isfinite(out['cpu_seconds']) and out['cpu_seconds']>=0 and out['maxrss_kib']>0
    assert out['model_calls']==out['cdf_calls']==out['rng_calls']==0
    assert (out['candidate_attempts'],out['valid_returns'],out['reference_attempts'],out['reference_returns'],out['guard_rejections'])==(44,36,12,12,8)
    cases,guards=fixtures();assert len(out['rows'])==20
    events=[dict(kind='started')];r['rows']=[]
    for case,row in zip(cases,out['rows'][:12],strict=True):
        cap();r['rows'].append(domain(case,row));save(r)
        events += [dict(kind='started_case',name=case['name']),dict(kind='completed_case',name=case['name'])]
    for (name,args),row in zip(guards,out['rows'][12:],strict=True):
        cap();r['rows'].append(guard(name,args,row));save(r)
        events += [dict(kind='started_case',name=name),dict(kind='completed_case',name=name)]
    by={v['name']:v for v in r['rows']};assert by['mixed_four']['range']==by['permuted_four']['range']
    events.append(dict(kind='completed'))
    assert [json.loads(s) for s in (a/'output/progress.jsonl').read_text().splitlines()]==events and len(events)==out['events']==42
    cap();r.update(complete=True,pass_all=True,domains=12,guards=8,events=42,independent_systems=sum(v['systems'] for v in r['rows'][:12]),author_cpu=out['cpu_seconds'],input_pins=completed['pins'],author_calls=0,model_cdf_rng_calls=0,limitation='Exact finite synthetic affine domains/adapters only. No trained improvement, probability mass, full cost or full-estimator accuracy claim; degenerate closed domains need not have positive probability.')
if __name__=='__main__':
    assert not(P/'results.json').exists() and not(P/'results.tmp').exists()
    result=dict(complete=False,pass_all=False);save(result)
    try:run(result)
    except BaseException as e:result.update(complete=False,pass_all=False,error=repr(e),traceback=traceback.format_exc());raise
    finally:save(result)
