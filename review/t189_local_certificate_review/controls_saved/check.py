"""Independent saved-only exact audit; never imports author numerical modules."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product, permutations
from math import factorial,isqrt
import hashlib,json,time,resource,traceback
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def eq(a,b):
    assert a==b,(a,b)

def dot(a,b):return sum((x*y for x,y in zip(a,b,strict=True)),F(0))
def rng(a,boxes):
    return (sum((min(w*l,w*u) for w,(l,u) in zip(a,boxes,strict=True)),F(0)),
            sum((max(w*l,w*u) for w,(l,u) in zip(a,boxes,strict=True)),F(0)))
def intersect(a,b):
    l,u=max(a[0],b[0]),min(a[1],b[1]);assert l<=u;return l,u

def decoded(a):return [F(q) for q in a]
def compare_scaled(stored,physical,den):eq([F(q,den) for q in stored],list(physical))
def independent_certificate(f,cert):
    m=f['model'];H,T,D=m['H'],m['T'],m['D'];box=[tuple(map(F,q)) for q in f['box']]
    wi=[F(w,D) for w in m['wi']];wh=[[F(w,D) for w in r] for r in m['wh']]
    wo=[[F(w,D) for w in r] for r in m['wo']]
    coeff=[[F(0)]*T for _ in range(H)];err=[(F(0),F(0))]*H;bounds=err.copy()
    eq(len(cert['trace']),T)
    for t,saved in enumerate(cert['trace']):
        eq(saved['step'],t);eq(saved['pre_denominator'],2*(2*D)**t*D);eq(saved['post_denominator'],4*(2*D)**t*D)
        nextc=[];nexte=[];nextb=[];eq(len(saved['rows']),H)
        for i,r in enumerate(saved['rows']):
            a=[dot(wh[i],[coeff[j][k] for j in range(H)]) for k in range(T)];a[t]+=wi[i]/2
            e=rng(wh[i],err);ab=rng(a,box);aff=(ab[0]+e[0],ab[1]+e[1]);ob=rng(wh[i],bounds)
            fresh=rng([wi[i]/2],[box[t]]);ordinary=(ob[0]+fresh[0],ob[1]+fresh[1]);L,U=intersect(aff,ordinary)
            if L>=0:k,R,kind=2,F(0),'on'
            elif U<=0:k,R,kind=0,F(0),'off'
            else:
                remainders=[2*U,max(-L,U),-2*L]
                k=min(range(3),key=lambda j:(j*(e[1]-e[0])+remainders[j],j));R=remainders[k];kind='crossing'
            c=[F(k,2)*v for v in a];ee=(k*e[0]/2,(k*e[1]+R)/2);bb=(max(F(0),L),max(F(0),U))
            eq(r['kind'],kind);eq(r['slope_multiplier'],k);eq(F(r['remainder_upper'],saved['pre_denominator']),R)
            for key,val in [('pre_A',a),('pre_error',e),('pre_range',(L,U))]:compare_scaled(r[key],val,saved['pre_denominator'])
            for key,val in [('post_A',c),('post_error',ee),('post_range',bb)]:compare_scaled(r[key],val,saved['post_denominator'])
            nextc.append(c);nexte.append(ee);nextb.append(bb)
        coeff,err,bounds=nextc,nexte,nextb
    eq(cert['margin_denominator'],2*(2*D)**T*D)
    margins={}
    for a in range(T):
        for b in range(a):
            w=[x-y for x,y in zip(wo[a],wo[b],strict=True)];c=[dot(w,[coeff[j][k] for j in range(H)]) for k in range(T)]
            e=rng(w,err);ab=rng(c,box);v=intersect((ab[0]+e[0],ab[1]+e[1]),rng(w,bounds));key=f'{a},{b}';margins[key]=v
            compare_scaled(cert['margins'][key],v,cert['margin_denominator'])
    eq(set(cert['margins']),set(margins));statuses=[]
    for target in range(T):
        good=True;bad=False
        for other in range(T):
            if other==target:continue
            l,u=margins[f'{max(target,other)},{min(target,other)}']
            if target<other:l,u=-u,-l
            good &= l>0 if other<target else l>=0
            bad |= u<=0 if other<target else u<0
        assert not(good and bad);statuses.append('correct' if good else 'wrong' if bad else 'unknown')
    eq(cert['statuses'],statuses)
    if f['expected'] is not None:eq(statuses,f['expected'])
    return wi,wh,wo

def direct_points(row,wi,wh,wo):
    f=row['fixture'];m=f['model'];H,T=m['H'],m['T']
    points=[list(map(F,q)) for q in product(*f['box'])]+[[F(a+b,2) for a,b in f['box']]]
    eq(len(row['exact_points']),17)
    for point,stored in zip(points,row['exact_points'],strict=True):
        eq(decoded(stored['v']),point);assert stored['complete'];h=[F(0)]*H;eq(len(stored['steps']),T)
        for t,step in enumerate(stored['steps']):
            z=[wi[i]*point[t]/2+dot(wh[i],h) for i in range(H)];h=[max(F(0),v) for v in z]
            eq(decoded(step['pre']),z);eq(decoded(step['hidden']),h)
        logits=[dot(w,h) for w in wo];winner=min(range(T),key=lambda j:(-logits[j],j))
        eq(decoded(stored['logits']),logits);eq(stored['winner'],winner)
        for j,status in enumerate(row['certificate']['statuses']):
            if status=='correct':eq(winner,j)
            if status=='wrong':assert winner!=j

def imul(a,b):
    q=[x*y for x in a for y in b];return min(q),max(q)
def atan64(q):
    lo=sum(((-1)**j*q**(2*j+1)/F(2*j+1) for j in range(64)),F(0));return lo,lo+q**129/F(129)
def sqrt160(x):
    S=1<<160;k=isqrt(x.numerator*S*S//x.denominator);l=F(k,S);u=l if l*l==x else F(k+1,S);assert l*l<=x<=u*u;return l,u
def cdf_interval(x):
    if x<0:
        l,u=cdf_interval(-x);return 1-u,1-l
    if not x:return F(1,2),F(1,2)
    p,q=atan64(F(1,5)),atan64(F(1,239));pi=(16*p[0]-4*q[1],16*p[1]-4*q[0])
    root=(sqrt160(2*pi[0])[0],sqrt160(2*pi[1])[1]);norm=(1/root[1],1/root[0])
    # Direct factorial sum, independent of the author's recursive integrated term.
    terms=[(-1)**n*x**(2*n+1)/F(2**n*factorial(n)*(2*n+1)) for n in range(97)]
    integral=(sum(terms[:96],F(0)),sum(terms,F(0)));l,u=imul(norm,integral)
    return F(1,2)+l,F(1,2)+u

def probability_check(setup,rows):
    Q=1<<80;boundaries=[-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8];dy=[]
    eq(len(setup['cdf_rows']),13);eq(len(rows),13)
    for v,s,c in zip(boundaries,setup['cdf_rows'],rows,strict=True):
        eq(s['v'],v);eq(c['v'],v);l,u=cdf_interval(F(v,2));eq((F(s['exact']['lower']),F(s['exact']['upper'])),(l,u))
        a=l.numerator*Q//l.denominator;b=-((-u.numerator*Q)//u.denominator);eq(s['dyadic'],[a,b]);dy.append((a,b))
        assert l<=F(c['reference120'])<=u
    bins=[(b[0]-a[1],b[1]-a[0]) for a,b in zip(dy,dy[1:])]
    eq(setup['bin_numerators'],[list(q) for q in bins]);eq(setup['bin_denominator'],Q);eq(setup['box_denominator'],Q**4)
    central=(dy[-1][0]-dy[0][1],dy[-1][1]-dy[0][0]);eq(setup['tail_numerators'],[Q**4-central[1]**4,Q**4-central[0]**4])
    assert all(0<a<=b for a,b in bins)

def main():
    out=HERE/'results.json';assert not out.exists() and not (HERE/'results.tmp').exists()
    resource.setrlimit(resource.RLIMIT_CPU,(19,20));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));start=time.process_time()
    result={'complete':False,'author_imports':0,'trained_model_calls':0,'random_draws':0,'synthetic_point_reconstructions':0}
    def save():
        result['cpu_seconds']=time.process_time()-start;result['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t=HERE/'results.tmp';t.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');t.replace(out)
    try:
        p=read(HERE/'run_scope.json');assert p['source_gate_commit']!='pending'
        for name,d in p['pins'].items():eq(sha(name),d)
        result['scope_sha256']=sha(HERE/'run_scope.json');save()
        source=Path(p['author_directory']);r=read(source/'controls_results.json');assert r['complete']
        protocol=read(source/'controls_protocol.json');eq(r['protocol_sha256'],sha(source/'controls_protocol.json'))
        eq(r['gate_sha256'],sha(source/'controls_gate.json'))
        for name,d in protocol['pins'].items():eq(sha(name),d)
        rows=r['rows'];eq(len(rows),44);assert all(q['passed'] for q in rows)
        by={k:[q for q in rows if q['kind']==k] for k in ['network','remainder','rank','cdf','crossover','guard']}
        eq([len(by[k]) for k in by],[9,9,6,13,4,3]);eq([q['fixture']['name'] for q in by['network']],protocol['network_names'])
        for row in by['network']:
            weights=independent_certificate(row['fixture'],row['certificate']);direct_points(row,*weights);result['synthetic_point_reconstructions']+=17;save()
        for q,(L,U,width) in zip(by['remainder'],[(L,U,w) for L,U in [(-3,1),(-1,3),(-1,1)] for w in [0,2,17]],strict=True):
            eq((q['L'],q['U'],q['error_width']),(L,U,width));rs=[2*U,max(-L,U),-2*L];eq(q['remainders'],rs);eq(q['chosen'],min(range(3),key=lambda j:(j*width+rs[j],j)))
        for q in by['rank']:
            bins=q['bin_ids'];second=sorted(bins,reverse=True)[1];group=[i for i,b in enumerate(bins) if b==second];eq(q['group'],group)
            counts=[0]*4;total=0
            for perm in permutations(range(4)):
                if list(bins[i] for i in perm)!=sorted(bins):continue
                total+=1;counts[perm[-2]]+=1
            eq(q['counts'],counts);eq(q['total'],total)
            for i in range(4):eq(F(counts[i],total),F(1,len(group)) if i in group else F(0))
        probability_check(r['probability_setup'],by['cdf'])
        for q,w in zip(by['crossover'],['0','1/2','1/4','1'],strict=True):
            eq(q['width'],w);w=F(w);z=q['result']
            if not w:assert z['crossover'] is None and not z['sufficient_tenfold'];continue
            threshold=25/w**2;n=threshold.numerator//threshold.denominator+1
            eq(z['minimum_samples'],n);eq(z['minimum_common_budget'],n);eq(z['frozen_samples'],10**7);eq(z['sufficient_tenfold'],10**7*w*w>25)
        eq([q['name'] for q in by['guard']],['zero_denominator','bad_box','boolean_dimension'])
        eq(r['synthetic_exact_forwards'],153);eq(r['candidate_certification_attempts'],12)
        for key in ['trained_forwards','checkpoint_loads','random_draws']:eq(r[key],0)
        events=[json.loads(x) for x in (source/'controls_progress.jsonl').read_text().splitlines()]
        expected=[('started',None)]
        for name in protocol['network_names']:expected.extend([('network_attempt',name),('network_complete',name)])
        expected += [('remainder_complete',None)]*9+[('rank_complete',None)]*6+[('cdf_complete',None)]*13+[('crossover_complete',None)]*4
        for name in ['zero_denominator','bad_box','boolean_dimension']:expected.extend([('guard_attempt',name),('guard_complete',name)])
        expected.append(('completed',None));eq([(e['stage'],e.get('name')) for e in events],expected)
        eq(events[-1]['results_sha256'],sha(source/'controls_results.json'))
        result.update(complete=True,rows=44,progress_records=len(events),author_cpu_seconds=r['cpu_seconds'],scope='All nine physical-unit affine certificates and153 direct saved synthetic points reconstructed; exact probability/ownership/crossover data checked. Named author operation/peak-bit ledger retained but not independently replayed or converted to FLOPs.')
        save()
    except BaseException:
        result['failure']=traceback.format_exc();save();raise
if __name__=='__main__':main()
