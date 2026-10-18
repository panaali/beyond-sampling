"""Independent saved-record audit; no author imports or geometry evaluation."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,time,resource,traceback
HERE=Path(__file__).resolve().parent
A=Path('/workspace/work/recover-small_h2_56/m16/approaches/future_equivalence/domain_sufficient')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rat(p):return F(int(p[0]),int(p[1]))
def mat(p):return tuple(tuple(rat(x) for x in r) for r in p)
def pack(x):
    if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [pack(v) for v in x]
    return x
def same(actual,expected):assert actual==pack(expected),(str(actual)[:160],str(pack(expected))[:160])
def compact(x):return {k:v for k,v in x.items() if v}
def normal(B,V,L):
    n=len(L[0]);K=len(L);linear=[list(r) for r in L];terms={};nonzero=negative=0
    for i,b in enumerate(B):
        nonzero_entries=[x for x in b if x]
        if not nonzero_entries:continue
        a=nonzero_entries[0];nonzero+=1;negative+=a<0
        u=tuple(x/a for x in b)
        contribution=tuple(abs(a)*r[i] for r in V)
        terms[u]=tuple(x+y for x,y in zip(terms.get(u,(F(0),)*K),contribution))
        if a<0:linear=[[x+r[i]*y for x,y in zip(l,b)] for l,r in zip(linear,V)]
    hinges=sorted((u,c) for u,c in terms.items() if any(c))
    key=(linear,hinges)
    nonzero_values=[abs(x) for r in linear for x in r if x]+[abs(x) for _,c in hinges for x in c if x]
    projective=key
    if nonzero_values:
        scale=nonzero_values[0];projective=([[x/scale for x in r] for r in linear],[(u,[x/scale for x in c]) for u,c in hinges])
    counts=compact(dict(div=nonzero*n+(K*n+K*len(hinges) if nonzero_values else 0),mul=nonzero*K+negative*K*n,add=nonzero*K+negative*K*n))
    return key,projective,counts

def domain(saved,ba,bb,va,vb,R):
    n=4;K=3;B=ba+bb;V=[list(x)+[-y for y in z] for x,z in zip(va,vb)];L=[[F(0)]*n for _ in range(K)]
    expected=[];kept=[];columns=[];nlinear=0
    for i,b in enumerate(B):
        dots=[sum((x*y for x,y in zip(b,r)),F(0)) for r in R]
        low=min(dots);high=max(dots)
        branch='drop' if high<=0 else ('linear' if low>=0 else 'keep')
        col=[row[i] for row in V]
        if branch=='linear':
            nlinear+=1;L=[[x+c*y for x,y in zip(row,b)] for row,c in zip(L,col)]
        if branch=='keep':kept.append(b);columns.append(col)
        expected.append(dict(form=b,coefficients=col,dots=dots,branch=branch))
    coefficients=[[c[k] for c in columns] for k in range(K)]
    same(saved['rays'],R);same(saved['hinges'],expected);same(saved['simplified'],dict(B=kept,V=coefficients,L=L))
    key,projective,cost=normal(kept,coefficients,L)
    same(saved['canonical']['key'],key);same(saved['canonical']['projective'],projective);same(saved['canonical']['counts'],cost)
    yes=not key[1] and not any(x for row in key[0] for x in row)
    assert saved['certified']==yes and saved['status']==('certified_equal' if yes else 'unknown')
    rc=dict(mul=K*len(bb)+len(B)*len(R)*n+nlinear*K*n,sub=K*n,add=len(B)*len(R)*n+nlinear*K*n)
    same(saved['restriction_counts'],rc);assert saved['sign_comparison_allowance']==2*len(B)*len(R)
    return rc,cost,saved['sign_comparison_allowance']

def pair_at(n,index):
    # Independent triangular-block inversion instead of constructing combinations.
    for i in range(n-1):
        width=n-i-1
        if index<width:return i,i+1+index
        index-=width
    raise AssertionError('pair index')

def check():
    pins=read(HERE/'pins.json')
    for path,digest in pins['pins'].items():assert sha(path)==digest,path
    p=read(A/'trained_protocol.json');d=read(A/'trained.json');full=read(p['full'])
    assert d['complete'] and full['complete'] and d['protocol_sha256']==sha(A/'trained_protocol.json')
    for path,digest in p['pins'].items():assert sha(path)==digest,path
    assert d['scope']==p['scope'] and d['new_geometry_network_forward_random_reference_mass_calls']==0
    events=[json.loads(x) for x in (A/'trained_progress.jsonl').read_text().splitlines()]
    assert len(events)==134 and events[0]==dict(stage='phase_attempt',protocol_sha256=d['protocol_sha256']) and events[-1]==dict(stage='complete')
    ev=1;hashes={};reports=[]
    assert [m['H'] for m in d['models']]==[4,8]
    for model,saved in zip(full['models'],d['models'],strict=True):
        H=model['H'];assert saved['H']==H and saved['checkpoint_sha256']==model['checkpoint_sha256']
        sources={}
        for item in model['inputs']:
            path=item['path'];assert sha(path)==item['sha256'];hashes[path]=item['sha256'];s=read(path)
            assert s['complete'] and s['checkpoint_sha256']==model['checkpoint_sha256'];sources[path]=s
        encoded=next(iter(sources.values()))['model'];assert saved['model']==encoded and all(s['model']==encoded for s in sources.values())
        wi=tuple(map(rat,encoded[0]));wh=mat(encoded[1]);wo=mat(encoded[2]);V=tuple(tuple(x-y for x,y in zip(row,wo[0])) for row in wo[1:])
        choices=[];pops=[]
        for truth in range(4):
            groups={}
            for r in model['rows']:
                if r['truth']!=truth:continue
                key=mat(r['exact_state']);identity=(r['parent'],r['source_index'])
                if key not in groups or identity<(groups[key]['parent'],groups[key]['source_index']):groups[key]=r
            reps=sorted(groups.values(),key=lambda r:(r['parent'],r['source_index']));n=len(reps);npairs=n*(n-1)//2;assert npairs>=8
            indices=[k*(npairs-1)//7 for k in range(8)];assert len(set(indices))==8
            pops.append(dict(truth=truth,representatives=[[r['parent'],r['source_index']] for r in reps],pair_count=npairs,selected_pair_indices=indices))
            for rank,index in enumerate(indices):
                i,j=pair_at(n,index);choices.append((truth,rank,index,reps[i],reps[j]))
        same(saved['populations'],pops);assert events[ev]==dict(stage='selection',H=H,populations=pops);ev+=1
        seen=set();ray_decodes=0;restriction=Counter();canonical=Counter();signs=0
        assert len(saved['rows'])==32
        for row,(truth,rank,index,a,b) in zip(saved['rows'],choices,strict=True):
            assert {k:row[k] for k in ('H','truth','rank','pair_index')}==dict(H=H,truth=truth,rank=rank,pair_index=index)
            assert events[ev]==dict(stage='pair_attempt',H=H,truth=truth,rank=rank,pair_index=index);ev+=1
            fs=[];rs=[]
            for tag,selected in [('a',a),('b',b)]:
                binding={k:selected[k] for k in ('occurrence_id','source','source_index','parent','truth','history','exact_state')};same(row[tag],binding)
                src=sources[selected['source']];raw=src['states'][selected['source_index']];analysis=src['analysis']['rows'][selected['source_index']]
                assert raw['depth']==3 and all(raw[k]==selected[k] for k in ('parent','truth','history'))
                assert raw['history']==analysis['history'] and selected['exact_state']==analysis['exact_state']
                M=mat(selected['exact_state']);assert len(M)==H and all(len(r)==3 for r in M);seen.add(M)
                B=tuple(tuple(sum((x*y for x,y in zip(w,col)),F(0)) for col in zip(*M))+(wi[i],) for i,w in enumerate(wh))
                same(row['function_'+tag],dict(B=B,V=V));fs.append(B)
                rays=tuple(tuple(F(float.fromhex(x)) for x in r) for r in raw['cone']['R']);assert rays and all(len(r)==4 for r in rays);rs.append(rays);ray_decodes+=4*len(rays)
            assert a['exact_state']!=b['exact_state'];answer=row['result'];assert len(answer['domains'])==2
            for dom,rays in zip(answer['domains'],rs,strict=True):
                rc,cc,sc=domain(dom,fs[0],fs[1],V,V,rays);restriction.update(rc);canonical.update(cc);signs+=sc
            yes=all(x['certified'] for x in answer['domains']);assert answer['certified']==yes and answer['status']==('certified_equal' if yes else 'unknown')
            assert events[ev]==dict(stage='pair_result',row=row);ev+=1
        summary=dict(pairs=32,both_hulls_certified=sum(r['result']['certified'] for r in saved['rows']),left_hull_certified=sum(r['result']['domains'][0]['certified'] for r in saved['rows']),right_hull_certified=sum(r['result']['domains'][1]['certified'] for r in saved['rows']),unknown_both=sum(not r['result']['certified'] for r in saved['rows']))
        same(saved['summary'],summary);assert events[ev]==dict(stage='model_result',H=H,summary=summary);ev+=1
        same(saved['formation_counts'],dict(sub=3*H,mul=len(seen)*3*H*H,add=len(seen)*3*H*H))
        assert saved['unique_selected_state_forms']==len(seen) and saved['state_decodes']==len(seen)*3*H and saved['weight_decodes']==H*H+5*H and saved['ray_decodes']==ray_decodes
        same(saved['restriction_arithmetic'],dict(restriction));same(saved['canonical_arithmetic_including_unused_projective'],dict(canonical));assert saved['sign_comparison_allowance']==signs
        reports.append(dict(H=H,summary=summary,unique_states=len(seen),ray_decodes=ray_decodes,restriction=dict(restriction),canonical=dict(canonical),formation=saved['formation_counts']))
    assert ev==133 and hashes==d['source_hashes']
    return dict(pass_all=True,models=reports,progress_records=len(events),author_cpu_seconds=d['cpu_seconds'],author_calls=0,geometry_forward_mass_random_reference_calls=0,exclusions=['Peak-bit ledger internals not replayed','No rank or completeness inference','No actual neural-domain or label/probability/cost-saving inference'])

def main():
    out=HERE/'results.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(19,20));start=time.process_time()
    result=dict(complete=False,checker_sha256=sha(__file__),pins_sha256=sha(HERE/'pins.json'))
    out.write_text(json.dumps(result,indent=2)+'\n')
    try:result.update(check(),complete=True)
    except BaseException as e:result.update(failure=repr(e),traceback=traceback.format_exc());raise
    finally:
        result.update(review_cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
