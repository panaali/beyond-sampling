"""Saved-only synthetic helper audit. Never imports candidate/reference modules."""
from pathlib import Path
from fractions import Fraction as F
import ast,hashlib,json,math,os,resource,struct,time,traceback
P=Path(__file__).resolve().parent; START=time.process_time()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(pins):
    assert isinstance(pins,dict) and pins
    for p,h in pins.items():assert type(h)is str and len(h)==64 and sha(p)==h,p

def save(r):
    r['cpu_seconds']=time.process_time()-START;r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t=P/'results.tmp';t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');os.replace(t,P/'results.json')
def cap():assert time.process_time()-START<19

def shape(v):
    if not isinstance(v,list):return []
    if not v:return [0]
    s=shape(v[0]);assert all(shape(x)==s for x in v);return [len(v)]+s

def flat(v):
    if isinstance(v,list):
        for x in v:yield from flat(x)
    else:yield v

def array(a,dtype,expected=None,allow_nonfinite=False):
    assert set(a)=={'dtype','shape','values'} and a['dtype']==dtype
    assert a['shape']==shape(a['values'])
    for v in flat(a['values']):
        if isinstance(v,dict):assert allow_nonfinite and set(v)=={'nonfinite'} and v['nonfinite'] in ('inf','nan')
        else:
            assert type(v) in (int,float) and math.isfinite(v)
            if dtype=='int64':assert type(v)is int
    if expected is not None:assert a['values']==expected
    return a['values']

def arr(v,dtype):return dict(dtype=dtype,shape=shape(v),values=v)
def zeros(dims):return [zeros(dims[1:]) for _ in range(dims[0])] if dims else 0.

def inputs():
    x=[list(map(float,range(-5,5))),list(map(float,range(4,-6,-1))),[0.]*10,[1.]*10,[-1.]*10,[1.,-1.]*5,[2.,2.,1.,1.,0.,0.,-1.,-1.,-2.,-2.],[j/8-.5 for j in range(10)],[1.]+[0.]*9,[0.]*9+[1.],[1048576.,-1048576.,3.,-3.,2.,-2.,1.,-1.,.5,-.5],[-0.,0.]*5]
    h=[[[float(F((i+1)*(s+1)*(j+1),16)) for j in range(4)] for s in range(11)] for i in range(12)]
    return x,h

def design(row,hrow,order,t,f):
    # Independent scalar polynomial cell ID and column construction. Sorting
    # itself shares the explicitly pinned NumPy quicksort primitive.
    ranked=[row[k] for k in order]+[0.]*(10-t)
    ages=[t-1-k for k in order]+[t]*(10-t)
    nr=[0,0,1,2,3,3,4,4,3,3,3,2][f]
    code=sum(ages[k]*(t+1)**(nr-k-1) for k in range(nr))
    if f>=5:code=4*code+min(sum(v>0 for v in row[:t]),3)
    take=3 if f<4 else 10 if f==7 else 5
    v=[row[t-1]]+ranked[:take]
    if f==1:
        theta=2*math.pi/8.42
        for k in range(3):v.extend([ranked[k]*math.cos(theta*ages[k]),ranked[k]*math.sin(theta*ages[k])])
    if f in (5,6,8,9,10,11):
        tail=[F.from_float(z) for z in ranked[5:t]]
        v.extend([float(sum((z for z in tail if z>0),F(0))),float(sum((z for z in tail if z<0),F(0)))])
    if f>=8:
        lag={8:1,9:3,10:9,11:9}[f]
        for s in range(t-1,max(0,t-lag-1),-1):v.extend(hrow[s])
    return code,v

def selections():
    a=zeros([12,16]);b=[[float(f)+j/32 for j in range(16)] for f in range(12)];c=[[-2.]*16 for _ in range(12)]
    for j in range(16):c[j%12][j]=1.;c[(j+5)%12][j]=1.
    return [('all_tie',a),('increasing',b),('column_ties',c),('negative',[[-v for v in row] for row in b])]

def endpoints(mode):
    a=zeros([12,3,16]);w=zeros([10,16]);c=[j%12 for j in range(16)]
    if mode in ('signed','readout_tie'):
        a=[[[float(F((f+1)*(r+1)*(j-7),16)) for j in range(16)] for r in range(3)] for f in range(12)]
        w[0][:8]=[1.]*8;w[1][8:]=[1.]*8;w[2][8:]=[1. if mode=='readout_tie' else -1.]*8
    if mode=='float32_boundary':
        for f in range(12):
            for r in range(3):a[f][r][0]=16777217.;a[f][r][1]=16777219.
            a[f][1][0]=2.**-150;a[f][1][1]=2.**-149;a[f][2]=[-0.]*16
        w[0][0]=1.;w[0][1]=-1.;w[1][0]=-1.;w[1][1]=1.
    return a,c,w

def nearest_even(q):
    n,r=divmod(q.numerator,q.denominator)
    return n+int(2*r>q.denominator or (2*r==q.denominator and n%2))

def binary32(v):
    # Integer round-to-nearest-even, including subnormals. No NumPy/struct cast
    # from the input float. struct only decodes the independently assembled bits.
    sign=struct.unpack('>Q',struct.pack('>d',v))[0]>>63;q=abs(F.from_float(v))
    if not q:bits=sign<<31
    else:
        e=q.numerator.bit_length()-q.denominator.bit_length()
        two=lambda n:F(2**n) if n>=0 else F(1,2**(-n))
        if q<two(e):e-=1
        if e < -126:
            m=nearest_even(q*2**149);assert m<=2**23;bits=(sign<<31)+m
        else:
            m=nearest_even(q/two(e-23))
            if m==2**24:m//=2;e+=1
            assert -126<=e<=127 and 2**23<=m<2**24
            bits=(sign<<31)+((e+127)<<23)+(m-2**23)
    return float(struct.unpack('>f',struct.pack('>I',bits))[0])

def guard_expected(label,x,h):
    import copy
    xx=copy.deepcopy(x);hh=copy.deepcopy(h);t=10;f=0;dx='float64';dh='float32'
    if label=='x_dtype':dx='float32'
    elif label=='x_shape':xx=[r[:9] for r in xx]
    elif label=='x_infinite':xx[0][0]={'nonfinite':'inf'}
    elif label=='t_zero':t=0
    elif label=='t_boolean':t=True
    elif label=='family_range':f=12
    elif label=='hinge_shape':hh=[r[:10] for r in hh]
    elif label=='hinge_dtype':dh='float64'
    if label=='score_nonfinite':
        s=zeros([12,16]);s[0][0]={'nonfinite':'nan'};return [arr(s,'float64')],'finite twelve-family validation scores'
    if label=='choice_range':return [arr(zeros([12,1,16]),'float64'),arr([12]+[0]*15,'int64'),arr(zeros([10,16]),'float64')],'choice range'
    reason='finite float64 input' if label in ('x_dtype','x_shape','x_infinite') else 'prefix/family' if label in ('t_zero','t_boolean','family_range') else 'stored float32 hinges'
    return [arr(xx,dx),t,f,arr(hh,dh)],reason

def main(r):
    scope=read(P/'scope.json');g=read(P/'gate.json');completed=read(P/'completed_inputs.json');review=read(g['source_review'])
    own={str(P/n) for n in ('check.py','scope.json','README.md')}
    assert g['ready'] is True and g['source_review_publication'] and g['execution_publication']
    assert review['complete'] is True and review['pass_all'] is True and review['execution_authorized'] is True
    assert own<=set(review['pins']) and own|{str(P/'completed_inputs.json'),g['source_review']}<=set(g['pins'])
    for d in (scope['pins'],g['pins'],completed['pins'],review['pins']):verify(d)
    a=Path(scope['author']);p=read(a/'protocol.json');ag=read(a/'gate.json');sr=read(p['source_review']);wr=read(p['written_review'])
    required={str(a/n) for n in ('core.py','fixtures.py','reference.py','run.py','launch.sh','PROTOCOL.md','protocol.json','manifest.json','gate.json','run1.log','output/results.json','output/progress.jsonl')}|{p['source_review'],p['written_review']}
    assert required<=set(completed['pins']) and completed['raw_commit'] and completed['publication']
    assert ag['ready'] is True and sr['complete'] is True and sr['pass_all'] is True and sr['execution_authorized'] is True
    assert wr['complete'] is True and wr['pass_all'] is True and wr['execution_allowed'] is False
    for d in (p['pins'],ag['pins'],sr['pins'],wr['pins'],read(a/'manifest.json')['pins']):verify(d)
    assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,)*2
    import numpy as np
    assert np.__version__==p['runtime']['numpy_version']=='2.5.4' and str(Path(np.__file__).resolve())==p['runtime']['numpy_init']
    con=a/'continuation';cp=read(con/'protocol.json');cg=read(con/'gate.json');cr=read(cp['review'])
    crequired={str(con/n) for n in ('run.py','launch.sh','PROTOCOL.md','protocol.json','gate.json','output/results.json','output/progress.jsonl','run1.log')}|{cp['review'],str(a/'ATTEMPT1.md')}
    assert crequired<=set(completed['pins']) and cg['ready'] is True
    assert cr['complete'] is True and cr['pass_all'] is True and cr['execution_authorized'] is True
    for v in (cp['pins'],cg['pins'],cr['pins']):verify(v)
    d=read(a/'output/results.json');nd=read(con/'output/results.json')
    assert d['complete'] is False and d['pass_all'] is False and 'internal CPU cap' in d['failure']
    assert nd['complete'] is True and nd['pass_all'] is True and nd['pins']==cg['pins']
    assert nd['limits']=={'cpu':[16,20],'AS':[2**31]*2}
    assert math.isfinite(nd['cpu_seconds']) and 0<=nd['cpu_seconds']<15 and type(nd['maxrss_kib'])is int and nd['maxrss_kib']>0
    assert [nd[k] for k in ('guard_attempts','events','candidate_pairs_repeated','reference_calls','fit_calls','model_calls','rng_calls')]==[6,14,0,0,0,0,0]
    assert nd['prior_cpu_seconds']==d['cpu_seconds'] and nd['prior_rows']==132 and nd['prior_events']==264
    assert nd['prior_missing_events']==['completed_row index131','original completed']
    assert nd['reuse']=={'candidate_pairs':128,'guards':4,'source':str(a/'output/results.json')}
    assert nd['combined_coverage']=={'candidate_attempts':128,'candidate_completed':128,'reference_attempts':128,'reference_completed':128,'guard_attempts':10,'rows':138,'old_events':264,'new_events':14,'old_complete':False,'qualification':'Two distinct progress streams; historical missing events are not reconstructed.'}
    assert len(d['rows'])==132 and len(nd['rows'])==6
    all_rows=d['rows']+nd['rows'];assert len(all_rows)==138
    assert d['pins']==ag['pins'] and d['protocol_pins']==p['pins'] and d['limits']=={'cpu':[16,20],'AS':[2**31]*2}
    assert math.isfinite(d['cpu_seconds']) and 15<=d['cpu_seconds']<20 and type(d['maxrss_kib'])is int and d['maxrss_kib']>0
    for k in ('candidate_attempts','candidate_completed','reference_attempts','reference_completed'):assert d[k]==128
    assert [d[k] for k in ('guard_attempts','events','network_calls','fit_calls','rng_calls')]==[4,264,0,0,0]
    tree=ast.parse(Path(p['owner_encode']).read_text());nodes=[]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in {'prefix_stats','_get','_age','base_vector','cell_index','design','design_h'}:nodes.append(n)
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'THETA','FAMILIES','HFAMILIES'}:nodes.append(n)
    assert ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)==d['reference_extraction_ast']
    x,h=inputs();array(d['fixed_inputs']['x'],'float64',x);array(d['fixed_inputs']['hinges'],'float32',h)
    r.update(rows=[],feature_values=0,shared_quicksort_calls=0,endpoint_scalar_casts=0,clock_max_relative=0.)
    for t in range(1,11):
        order=np.argsort(-np.array(x,dtype=np.float64)[:,:t],axis=1,kind='quicksort').tolist();r['shared_quicksort_calls']+=1
        for f in range(12):
            cap();ix=(t-1)*12+f;row=all_rows[ix]
            assert row['kind']=='features' and row['t']==t and row['family']==f and row['status']=='passed'
            expected=[design(xx,hh,ii,t,f) for xx,hh,ii in zip(x,h,order,strict=True)]
            ec=[a for a,b in expected];ev=[b for a,b in expected]
            matrices=[]
            for side in ('candidate','reference'):
                assert len(row[side])==2;array(row[side][0],'int64',ec);v=array(row[side][1],'float64');assert shape(v)==shape(ev)
                for got,want in zip(flat(v),flat(ev),strict=True):
                    err=abs(got-want)/max(1.,abs(want))
                    if f==1:assert err<=4e-15;r['clock_max_relative']=max(r['clock_max_relative'],err)
                    else:assert got==want
                matrices.append(v)
            a1,b1=matrices;exact=a1==b1;delta=max(abs(u-v)/max(1.,abs(v)) for u,v in zip(flat(a1),flat(b1),strict=True))
            assert row['cells_equal'] is True and row['base_numeric_equal']==exact and row['normalized_max_abs']==delta
            assert delta<=4e-15 if f==1 else exact
            r['feature_values']+=sum(1 for _ in flat(ev));r['rows'].append({'kind':'features','t':t,'family':f,'columns':len(ev[0]),'passed':True});save(r)
    for off,(label,scores) in enumerate(selections(),120):
        cap();row=all_rows[off];assert row['kind']=='selection' and row['label']==label and row['status']=='passed';array(row['scores'],'float64',scores)
        expected=[[max(range(stop),key=lambda k:scores[k][j]) for j in range(16)] for stop in (12,8)]
        for side in ('candidate','reference'):
            assert len(row[side])==2
            for got,want in zip(row[side],expected,strict=True):array(got,'int64',want)
        r['rows'].append({'kind':'selection','label':label,'passed':True});save(r)
    for off,label in enumerate(('zero','signed','readout_tie','float32_boundary'),124):
        cap();row=all_rows[off];assert row['kind']=='endpoint' and row['label']==label and row['status']=='passed'
        a0,c,w=endpoints(label);array(row['inputs']['predictions'],'float64',a0);array(row['inputs']['choices'],'int64',c);array(row['inputs']['readout'],'float64',w)
        chosen=[[binary32(a0[c[j]][i][j]) for j in range(16)] for i in range(3)];r['endpoint_scalar_casts']+=48
        hidden=[[max(0.,z) for z in row] for row in chosen]
        logits=[[float(sum((F.from_float(z)*F.from_float(weight) for z,weight in zip(hh,ww,strict=True)),F(0))) for ww in w] for hh in hidden]
        predictions=[max(range(10),key=v.__getitem__) for v in logits];expected={'chosen':('float32',chosen),'hidden':('float64',hidden),'logits':('float64',logits),'prediction':('int64',predictions)}
        for side in ('candidate','reference'):
            assert set(row[side])==set(expected)
            for k,(dtype,value) in expected.items():array(row[side][k],dtype,value)
        r['rows'].append({'kind':'endpoint','label':label,'predictions':predictions,'passed':True});save(r)
    labels=['x_dtype','x_shape','x_infinite','t_zero','t_boolean','family_range','hinge_shape','hinge_dtype','score_nonfinite','choice_range']
    for off,label in enumerate(labels,128):
        cap();row=all_rows[off];args,reason=guard_expected(label,x,h)
        assert row['kind']=='guard' and row['label']==label and row['status']=='passed' and row['rejection']==reason
        # JSON comparison retains True versus 1 in malformed domain records.
        assert json.dumps(row['arguments'],sort_keys=True)==json.dumps(args,sort_keys=True)
        r['rows'].append({'kind':'guard','label':label,'passed':True});save(r)
    events=[{'event':'started'}]
    for i in range(132):
        events.append({'event':'started_row','index':i})
        if i<131:events.append({'event':'completed_row','index':i})
    assert len(events)==264 and [json.loads(s) for s in (a/'output/progress.jsonl').read_text().splitlines()]==events
    new_events=[{'event':'started'}]
    for i in range(6):new_events.extend([{'event':'started_row','index':i},{'event':'completed_row','index':i}])
    new_events.append({'event':'completed'})
    assert len(new_events)==14 and [json.loads(s) for s in (con/'output/progress.jsonl').read_text().splitlines()]==new_events
    cap();r.update(complete=True,pass_all=True,total_rows=138,events=278,progress_streams={'original':264,'continuation':14,'original_complete':False,'original_missing_events':['completed_row index131','original completed']},author_cpu_attempt1=d['cpu_seconds'],author_cpu_continuation=nd['cpu_seconds'],author_cpu_total=d['cpu_seconds']+nd['cpu_seconds'],author_calls=0,model_fit_rng_calls=0,scope_sha256=sha(P/'scope.json'),input_pins=completed['pins'])

if __name__=='__main__':
    assert not (P/'results.json').exists() and not (P/'results.tmp').exists()
    r={'complete':False,'pass_all':False,'rows':[]};save(r)
    try:main(r)
    except BaseException:r.update(complete=False,pass_all=False,failure=traceback.format_exc());raise
    finally:save(r)
