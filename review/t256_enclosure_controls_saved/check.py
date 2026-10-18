"""Independent saved physical-Fraction reconstruction. Never imports author modules."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from collections import Counter
from math import isfinite
import copy,hashlib,json,resource,time,traceback
from vertex import independent_vertices
P=Path(__file__).resolve().parent;START=time.process_time()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pins(x):
    assert x
    for p,h in x.items():assert sha(p)==h,p
def q(x):
    if isinstance(x,F):return x  # reviewer-generated physical coordinates
    assert type(x) in (str,int),type(x)
    return F(x)
def rv(xs):return list(map(q,xs))
def rt(x):
    if isinstance(x,dict):return {k:rt(v) for k,v in x.items()}
    if isinstance(x,list):return list(map(rt,x))
    return q(x)
def eq(saved,expected):assert rt(saved)==expected,(saved,expected)
def cap():assert time.process_time()-START<19,'19 CPU review cap'
def save(r):
    r['review_cpu']=time.process_time()-START;r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t=P/'results.tmp';t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(P/'results.json')
def dot(a,b):return sum((x*y for x,y in zip(a,b,strict=True)),F(0))
def interval(a,ivs):
    terms=[sorted((w*l,w*u)) for w,(l,u) in zip(a,ivs,strict=True)]
    return [sum((t[k] for t in terms),F(0)) for k in (0,1)]
def classify(margins,j,T):
    allwin=True;loses=False
    for k in range(T):
        if k==j:continue
        lo,hi=margins[f'{max(j,k)},{min(j,k)}']
        if j<k:lo,hi=-hi,-lo
        allwin &= lo>0 if k<j else lo>=0
        loses |= hi<=0 if k<j else hi<0
    assert not(allwin and loses)
    return 'correct' if allwin else 'wrong' if loses else 'unknown'
SIGNED=dict(wi=[1,-1],wh=[[0,0],[0,0]],wo=[[1,0],[0,1],[0,0],[-1,-1]],D=1,H=2,T=4)
PARENT=[[1,2],[-1,0],[-1,1],[-1,1]];ORIGIN=[[0,0]]*4

def inherited():
    def make(name,wi,wh,wo,b,expected=None,D=1,stable=False):return dict(name=name,model=dict(wi=wi,wh=wh,wo=wo,D=D,H=len(wi),T=4),box=b,expected=expected,stable=stable)
    z=[[0,0],[0,0]];w=SIGNED['wo'];mixed=[[-2,2]]*4
    return [make('zero_readout',[1],[[0]],[[0]]*4,mixed,['correct','wrong','wrong','wrong']),make('identical_active',[1],[[0]],[[1]]*4,[[2,4]]*4,['correct','wrong','wrong','wrong'],stable=True),make('negative_final',[1,-1],z,w,[[-2,2]]*3+[[-4,-2]],['wrong','correct','wrong','wrong']),make('positive_final',[1,-1],z,w,[[-2,2]]*3+[[2,4]],['correct','wrong','wrong','wrong']),make('crossing_final',[1,-1],z,w,mixed,['unknown','unknown','wrong','wrong']),make('signed_recurrence',[1,-1],[[1,-1],[-1,1]],w,mixed),make('dense_denominator',[1,-2,3],[[1,-1,2],[-2,1,1],[3,0,-1]],[[1,0,-1],[0,1,1],[-1,2,0],[2,-1,1]],[[-4,-2],[0,2],[-2,2],[2,4]],D=2),make('all_stable',[1,2],[[1,1],[0,1]],[[1,0],[0,1],[1,1],[-1,-1]],[[2,4]]*4,['wrong','wrong','correct','wrong'],stable=True),make('zero_input',[0],[[1]],[[0],[1],[-1],[0]],mixed,['correct','wrong','wrong','wrong'],stable=True)]

def reconstruct(model,box,saved,domain=None,old=None):
    """Use independently enumerated vertices, not the author's breakpoint optimizer."""
    T,H,D=(model[k] for k in ('T','H','D'));box=[rv(v) for v in box];M=T*(T-1)//2;G=T*H
    K=None;systems=0
    if domain is None:vertices=list(product(*box));assert saved['feasibility'] is None
    else:
        i,j,g=domain['leader'],domain['second'],q(domain['gap'])
        planes,vertices,systems,nonsingular=independent_vertices(box,i,j,g)
        f=saved['feasibility'];assert f['status']==('feasible' if vertices else 'empty')
        lo=max([box[j][0]]+[box[k][0]+g for k in range(T) if k not in (i,j)]);hi=min(box[j][1],box[i][1]-g)
        eq(f['second_interval'],[lo,hi]);K=sorted({lo,hi}|{v for v in [box[i][0]-g]+[box[k][1]+g for k in range(T) if k not in(i,j)] if lo<=v<=hi}) if vertices else []
        assert len(f['candidates'])==len(K)
        # The feasibility call has zero objective, so verify its completions directly.
        for t,r in zip(K,f['candidates'],strict=True):
            eq(r['t'],t);eq(r['values'],[F(0),F(0)])
            x=[t if k==j else max(l,t+g) if k==i else l for k,(l,u) in enumerate(box)]
            eq(r['low'],x);eq(r['high'],x)
            assert all(dot(normal,x)>=rhs for normal,rhs in planes)
        work=dict(input_rational_values=3*T+2,ordered_pairs=1,candidate_values=len(K),endpoint_completions=2*len(K),objective_products=2*T*len(K),objective_additions=2*(T+1)*len(K))
        assert f['work']==work and 'not fully priced' in f['cost_scope']
        if vertices:
            eq(f['range'],[F(0),F(0)]);eq(f['witnesses'],[rv(f['candidates'][0]['low'])]*2)
        else:assert f['range'] is None and f['witnesses'] is None
    count=Counter(box_rational_values=2*T)
    if domain is not None:count['feasibility_range_calls']=1
    if not vertices:
        assert saved['status']=='empty' and saved['trace']==[]
        for k in ('raw_margins','final_margins','old_margins','raw_statuses','statuses'):assert saved[k] is None
        assert saved['accounting']['selected_counts']==dict(count)
        return dict(empty=True,systems=systems,trace_rows=0),None
    assert saved['status']=='feasible' and len(saved['trace'])==T
    wi=[F(v,D) for v in model['wi']];wh=[[F(v,D) for v in row] for row in model['wh']];wo=[[F(v,D) for v in row] for row in model['wo']]
    count['weight_rational_values']=H+H*H+T*H
    A=[[F(0)]*T for _ in range(H)];c=[F(0)]*H;E=[[F(0),F(0)] for _ in range(H)];B=copy.deepcopy(E)
    def arange(cc,aa):
        values=[cc+dot(aa,x) for x in vertices];return [min(values),max(values)]
    for t,step in enumerate(saved['trace']):
        assert step['step']==t and len(step['rows'])==H
        new=[]
        for k,row in enumerate(step['rows']):
            w=wh[k];cc=dot(w,c);aa=[dot(w,[A[h][j] for h in range(H)]) for j in range(T)];aa[t]+=wi[k]
            ee=interval(w,E);av=arange(cc,aa);aff=[av[0]+ee[0],av[1]+ee[1]]
            direct=interval(w,B);inp=interval([wi[k]],[box[t]]);direct=[direct[z]+inp[z] for z in (0,1)]
            lo,hi=max(aff[0],direct[0]),min(aff[1],direct[1]);assert lo<=hi
            if lo>=0:lam,res,kind=F(1),F(0),'on'
            elif hi<=0:lam,res,kind=F(0),F(0),'off'
            else:
                options=[(s*(ee[1]-ee[0])+max(-s*lo,(1-s)*hi),s,max(-s*lo,(1-s)*hi)) for s in (F(0),F(1,2),F(1))]
                _,lam,res=min(options);kind='crossing';count['crossing_slope_candidates']+=3
            nc=lam*cc;na=[lam*v for v in aa];ne=[lam*ee[0],lam*ee[1]+res];nb=[max(F(0),lo),max(F(0),hi)]
            expected=dict(pre_constant=cc,pre_A=aa,pre_error=ee,affine_range=aff,direct_range=direct,pre_range=[lo,hi],post_constant=nc,post_A=na,post_error=ne,post_range=nb,slope=lam,remainder_upper=res)
            assert set(row)==set(expected)|{'kind'} and row['kind']==kind
            for name,value in expected.items():eq(row[name],value)
            count['gate_'+kind]+=1;count['slope_'+str(lam)]+=1;new.append((nc,na,ne,nb))
        c,A,E,B=map(list,zip(*new));cap()
    raw={};assert len(saved['margin_trace'])==M
    for i in range(T):
        for j in range(i):
            w=[wo[i][h]-wo[j][h] for h in range(H)];cc=dot(w,c);aa=[dot(w,[A[h][k] for h in range(H)]) for k in range(T)];ee=interval(w,E)
            av=arange(cc,aa);aff=[av[z]+ee[z] for z in (0,1)];direct=interval(w,B);lo,hi=max(aff[0],direct[0]),min(aff[1],direct[1]);assert lo<=hi
            key=f'{i},{j}';raw[key]=[lo,hi];eq(saved['margin_trace'][key],dict(constant=cc,coeff=aa,error=ee,affine_range=aff,direct_range=direct))
    eq(saved['raw_margins'],raw);final=copy.deepcopy(raw)
    if old is None:assert saved['old_margins'] is None
    else:
        eq(saved['old_margins'],old)
        for key in raw:final[key]=[max(raw[key][0],old[key][0]),min(raw[key][1],old[key][1])];assert final[key][0]<=final[key][1]
        count['fallback_intersections']=M
    eq(saved['final_margins'],final)
    assert saved['raw_statuses']==[classify(raw,j,T) for j in range(T)] and saved['statuses']==[classify(final,j,T) for j in range(T)]
    whole=domain is None
    count.update(dot_calls=G*(T+7+2*whole)+M*(T+5+2*whole),dot_terms=G*(H*(T+5)+2+2*T*whole)+M*(H*(T+5)+2*T*whole),interval_sign_choices=G*(2*H+1+T*whole)+M*(2*H+T*whole),range_intersections=G+M,gate_decisions=G,readout_contrasts=M)
    count['whole_affine_range_calls' if whole else 'restricted_affine_range_calls']=G+M
    if not whole:
        for k,n in work.items():count['range_'+k]=(G+M)*n
    assert saved['accounting']['selected_counts']==dict(count),(saved['accounting']['selected_counts'],dict(count))
    acc=saved['accounting'];assert type(acc['observed_numerator_bits']) is int and acc['observed_numerator_bits']>=0 and type(acc['observed_denominator_bits']) is int and acc['observed_denominator_bits']>=0 and 'incomplete' in acc['scope']
    return dict(empty=False,systems=systems,vertices=len(vertices),trace_rows=G,contrasts=M,selected_counts=dict(count)),raw

def run(r):
    scope=read(P/'scope.json');gate=read(P/'gate.json');done=read(P/'completed_inputs.json')
    assert gate['ready'] is True and gate['review_publication'] and gate['execution_publication'] and done['raw_commit'] and done['publication']
    review=read(gate['source_review']);assert review['complete'] is True and review['pass_all'] is True and review['execution_authorized'] is True
    required={str(P/n) for n in ('check.py','vertex.py','scope.json','README.md')}
    assert required<=set(review['pins']) and required|{gate['source_review'],str(P/'completed_inputs.json')}<=set(gate['pins'])
    for x in (scope['pins'],gate['pins'],review['pins'],done['pins']):pins(x)
    a=Path(scope['author']);ap=read(a/'protocol.json');ag=read(a/'gate.json')
    need={str(a/n) for n in ('range_primitive.py','enclosure.py','fixtures.py','run.py','launch.sh','PROTOCOL.md','protocol.json','gate.json','output/results.json','output/progress.jsonl','run1.log')}|{ap[k] for k in ('source_review','theorem_review','prior_review','prior_scope','prior_results','primitive_review')}
    assert need<=set(done['pins']);assert ag['ready'] is True
    ar=read(ap['source_review']);assert ar['complete'] is True and ar['pass_all'] is True and ar['execution_authorized'] is True
    for x in (ap['pins'],ag['pins'],ar['pins']):pins(x)
    for k in ('theorem_review','primitive_review'):
        z=read(ap[k]);assert z['complete'] is True and z['pass_all'] is True
    ps=read(ap['prior_scope']);pr=read(ap['prior_review']);pins(ps['pins']);assert ap['prior_results'] in ps['pins']
    assert pr['complete'] is True and pr['rows']==44 and pr['synthetic_point_reconstructions']==153 and pr['scope_sha256']==sha(ap['prior_scope'])
    out=read(a/'output/results.json');old=read(ap['prior_results']);assert old['complete'] is True
    assert out['complete'] is True and out['pass_all'] is True and out['pins']==ag['pins'] and out['protocol_pins']==ap['pins'] and out['prior_scope_sha256']==sha(ap['prior_scope'])
    assert out['limits']==dict(cpu=[16,20],address_space=[2*1024**3]*2)
    assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2*1024**3,)*2
    assert isfinite(out['cpu_seconds']) and out['cpu_seconds']>=0 and out['maxrss_kib']>0
    assert tuple(out[k] for k in ('certificate_attempts','certificate_returns','synthetic_propagations_completed','normalization_attempts','normalization_returns','guard_rejections'))==(20,15,14,4,3,6)
    assert all(out[k]==0 for k in ('old_candidate_calls','direct_point_forwards','trained_model_calls','cdf_calls','rng_calls'))
    rows=out['rows'];assert len(rows)==24 and all(z['status']=='passed' for z in rows)
    oldrows=[z for z in old['rows'] if z['kind']=='network'];fs=inherited();assert len(oldrows)==9
    r['rows']=[];events=[dict(kind='started')]
    def record(row,diag):
        r['rows'].append(dict(name=row['name'],**diag));save(r)
        events.extend([dict(kind='started_case',name=row['name']),dict(kind='completed_case',name=row['name'])]);cap()
    for f,row,prev in zip(fs,rows[:9],oldrows,strict=True):
        assert row['kind']=='inherited' and row['name']==f['name'] and row['input']['fixture']==prev['fixture']==f and prev['passed'] is True and row['attempted'] is True
        box=[[F(v,2) for v in iv] for iv in f['box']];eq(row['input']['physical_box'],box)
        diag,raw=reconstruct(f['model'],box,row['certificate']);new=row['certificate'];saved=prev['certificate'];assert len(saved['trace'])==4
        n=0
        for ns,os in zip(new['trace'],saved['trace'],strict=True):
            assert ns['step']==os['step'] and len(ns['rows'])==len(os['rows'])==f['model']['H']
            for nr,orr in zip(ns['rows'],os['rows'],strict=True):
                for phase in ('pre','post'):
                    den=os[phase+'_denominator'];assert type(den) is int and den>0
                    eq(nr[phase+'_constant'],F(0));eq(nr[phase+'_A'],[F(2*v,den) for v in orr[phase+'_A']])
                    for k in ('error','range'):eq(nr[phase+'_'+k],[F(v,den) for v in orr[phase+'_'+k]])
                assert nr['kind']==orr['kind'];eq(nr['slope'],F(orr['slope_multiplier'],2));eq(nr['remainder_upper'],F(orr['remainder_upper'],os['post_denominator']));n+=1
        den=saved['margin_denominator'];assert type(den) is int and den>0
        assert raw=={k:[F(v,den) for v in iv] for k,iv in saved['margins'].items()}
        assert new['statuses']==saved['statuses'] and row['compared_trace_rows']==n and row['saved_margin_denominator']==den
        record(row,diag)
    baselines={}
    for name,box,row in zip(('partial_parent','origin_parent'),(PARENT,ORIGIN),rows[9:11],strict=True):
        assert row['kind']=='baseline' and row['name']==name and row['input']==dict(model=SIGNED,box=box) and row['attempted'] is True
        diag,raw=reconstruct(SIGNED,box,row['certificate']);baselines[name]=raw;record(row,diag)
    domains=[dict(name='partial_negative_final',baseline='partial_parent',domain=dict(leader=0,second=1,gap='1/2'),last_range=[-1,'-1/2'],target=1,expected='correct',positive_volume=True),dict(name='empty',baseline='partial_parent',domain=dict(leader=0,second=1,gap=2),last_range=None,target=1,expected='empty',positive_volume=False),dict(name='touching_negative_final',baseline='partial_parent',domain=dict(leader=0,second=1,gap=1),last_range=[-1,-1],target=1,expected='correct',positive_volume=False),dict(name='origin_tie',baseline='origin_parent',domain=dict(leader=0,second=1,gap=0),last_range=[0,0],target=1,expected='wrong',positive_volume=False)]
    for f,row in zip(domains,rows[11:15],strict=True):
        box=PARENT if f['baseline']=='partial_parent' else ORIGIN
        assert row['kind']=='domain' and row['name']==f['name'] and row['input']['fixture']==f and row['input']['model']==SIGNED and row['input']['box']==box and row['attempted'] is True
        oldmargin=baselines[f['baseline']];eq(row['input']['old_margins'],oldmargin)
        diag,raw=reconstruct(SIGNED,box,row['certificate'],f['domain'],oldmargin)
        if raw is not None:
            last=rv(f['last_range']);coeff=[0,-1,0,1];expected={f'{i},{j}':interval([F(coeff[i]-coeff[j])],[last]) for i in range(4) for j in range(i)}
            assert raw==expected;eq(row['analytic_margins'],expected);eq(row['certificate']['final_margins'],expected)
            assert row['certificate']['statuses'][1]==f['expected']
            assert row['rank_label_note']==('g0 supplied target only, task-rank tie ownership not inferred' if f['domain']['gap']==0 else 'strict positive gaps identify the supplied second label')
        else:assert f['expected']=='empty'
        record(row,diag)
    normal=[('nonzero_offset_error',dict(constant=3,coeff=[2,-1,0],error=[-2,1],bounds=[-1,8],denominator=4)),('thirds',dict(constant=-2,coeff=[-3,4,1],error=[0,0],bounds=[-6,9],denominator=3)),('eighths',dict(constant=0,coeff=[1,-2,4,0],error=[-3,5],bounds=[-10,20],denominator=8))]
    for (name,args),row in zip(normal,rows[15:18],strict=True):
        assert row['kind']=='normalization' and row['name']==name and row['input']['name']==name and row['input']['args']==args
        d=args['denominator'];expected=dict(constant=F(args['constant'],d),coeff=[F(2*v,d) for v in args['coeff']],error=[F(v,d) for v in args['error']],bounds=[F(v,d) for v in args['bounds']])
        eq(row['normalized'],expected);eq(row['input']['expected'],expected);record(row,dict(normalization_verified=True))
    guardnames=['zero_model_denominator','boolean_dimension','reversed_box','float_weight','same_domain_labels','zero_normalization_denominator']
    for name,row in zip(guardnames,rows[18:],strict=True):
        args=dict(model=copy.deepcopy(SIGNED),box=copy.deepcopy(PARENT))
        if name=='zero_model_denominator':args['model']['D']=0
        elif name=='boolean_dimension':args['model']['H']=True
        elif name=='reversed_box':args['box'][0]=[2,1]
        elif name=='float_weight':args['model']['wi'][0]=1.0
        elif name=='same_domain_labels':args['domain']=dict(leader=0,second=0,gap=0)
        else:args=dict(normal[0][1],denominator=0)
        assert row['kind']=='guard' and row['name']==name and json.dumps(row['input'],sort_keys=True)==json.dumps(args,sort_keys=True) and row['attempted'] is True
        assert isinstance(row['error'],str) and row['error'] and 'unexpected_result' not in row
        record(row,dict(invalid_input_confirmed=True))
    events.append(dict(kind='completed'));assert [json.loads(s) for s in (a/'output/progress.jsonl').read_text().splitlines()]==events and out['events']==len(events)==50
    cap();r.update(complete=True,pass_all=True,total_rows=24,events=50,exact_physical_reconstructions=14,empty_domains=1,normalizations=3,guards=6,author_cpu=out['cpu_seconds'],input_pins=done['pins'],author_calls=0,model_cdf_rng_calls=0,scope_sha256=sha(P/'scope.json'),limitations='Finite synthetic reconstruction only. Exact active-constraint vertex ranges independent of author breakpoint range. Named counts checked; peak-bit internal observation ledger not replayed. No trained accuracy, probability mass, full-cost or original point-forward validation.')
if __name__=='__main__':
    assert not(P/'results.json').exists() and not(P/'results.tmp').exists()
    r=dict(complete=False,pass_all=False,rows=[]);save(r)
    try:run(r)
    except BaseException as e:r.update(complete=False,pass_all=False,error=repr(e),traceback=traceback.format_exc());raise
    finally:save(r)
