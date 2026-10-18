"""Reviewer reconstruction reused from independently passed a3479b55; no author imports."""
from fractions import Fraction as F
from collections import Counter
from itertools import product
import copy,time
from vertex import independent_vertices
START=time.process_time()
def cap():assert time.process_time()-START<19

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
