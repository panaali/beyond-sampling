"""Diagnostic saved box audit with explicit physical coordinates; no author imports."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import gzip, hashlib, json, math, resource, time, traceback
HERE=Path(__file__).resolve().parent
START=time.process_time()
B=(-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8)
G=(0,1,2,4)
KEYS=('correct','wrong','network_unknown','inner','outer','boundary_box')
HASHES={}

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    p=str(p)
    if p not in HASHES: HASHES[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return HASHES[p]
def pins(v):
    assert v
    for p,h in v.items(): assert sha(p)==h,p

def status(m,j):
    # Reconstruct intervals of rival logits with the candidate logit fixed at 0.
    lo=[0]*4; hi=[0]*4
    for k in range(4):
        if k==j: continue
        a,b=m[f'{max(j,k)},{min(j,k)}']
        lo[k],hi[k]=(-b,-a) if j>k else (a,b)
    best=max(range(4),key=lambda k:(lo[k],-k))
    worst=max(range(4),key=lambda k:(hi[k],-k))
    if worst==j: return 'correct'
    if best!=j: return 'wrong'
    return 'unknown'

def ownership(box,g):
    inner=[]; outer=[]
    for i in range(4):
        for j in range(4):
            if i==j: continue
            others=[k for k in range(4) if k not in (i,j)]
            # Each gap is affine: its minimum occurs at these two endpoints.
            minima=[box[i][0]-box[j][1]]+[box[j][0]-box[k][1] for k in others]
            if all(x>=g for x in minima): inner.append([i,j])
            # Independent feasibility test at every critical second-coordinate
            # endpoint. Complete the largest coordinate at its upper endpoint
            # and both remaining coordinates at their lower endpoints.
            candidates={box[j][0],box[j][1],box[i][1]-g}
            candidates.update(box[k][0]+g for k in others)
            for q in candidates:
                if not box[j][0]<=q<=box[j][1]: continue
                if box[i][1]-q<g: continue
                if any(q-box[k][0]<g for k in others): continue
                outer.append([i,j]); break
    return inner,outer

def sum_into(dest,mm,n):
    dest[0]+=n*mm[0];dest[1]+=n*mm[1]
def ratio(v,den): return [str(F(x,den)) for x in v]
def cap(): assert time.process_time()-START<19,'19 CPU review cap'
def save(out):
    out['review_cpu']=time.process_time()-START
    out['review_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    temporary=HERE/'results.tmp'
    temporary.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    temporary.replace(HERE/'results.json')

def run(out):
    scope=read(HERE/'scope.json');gate=read(HERE/'gate.json')
    assert scope['diagnostic_only'] is True and scope['physical_gap_denominator']==8
    assert scope['original_key_denominator']==4 and scope['physical_gap_one_tested'] is False
    correction=read(scope['independent_unit_correction'])
    assert correction['complete'] is True and correction['intended_physical_gap_interpretation_valid'] is False
    assert correction['implemented_physical_gaps']==['0','1/8','1/4','1/2']
    pins(correction['pins'])
    assert gate['ready'] is True and gate['review_publication'] and gate['execution_publication']
    review=read(gate['source_review'])
    assert review['complete'] is True and review['pass_all'] is True and review['execution_authorized'] is True
    need={str(HERE/'check.py'),str(HERE/'scope.json')}
    assert need<=set(review['pins'])
    assert need|{gate['source_review'],str(HERE/'completed_inputs.json')}<=set(gate['pins'])
    for v in (scope['pins'],gate['pins'],review['pins']): pins(v)
    completed=read(HERE/'completed_inputs.json');pins(completed['pins'])
    assert completed['raw_commit'] and completed['publication']
    author=Path(scope['author']);p=read(author/'protocol.json');ag=read(author/'gate.json')
    required={str(author/n) for n in ['run.py','launch.sh','PROTOCOL.md','protocol.json','gate.json','output/results.json','output/progress.jsonl','run1.log']}
    required|={str(author/'output'/f'chunk_{k:03}.json.gz') for k in range(81)}
    assert required<=set(completed['pins'])
    assert ag['complete'] is True and ag['execution_authorized'] is True
    assert ag['protocol_sha256']==sha(author/'protocol.json')
    ar=read(ag['source_review'])
    assert ar['complete'] is True and ar['pass_all'] is True and ar['execution_authorized'] is True
    assert ag['source_review'] in completed['pins'] and ag['source_review'] in ag['pins']
    for v in (p['pins'],ag['pins'],ar['pins']):pins(v)
    r=read(author/'output/results.json')
    assert r['complete'] is True and r['pass_all'] is True and r['rows']==20736 and len(r['chunks'])==81
    assert set(r['gap_summaries'])=={'0','1/4','1/2','1'}
    assert r['protocol_sha256']==sha(author/'protocol.json') and r['gate_sha256']==sha(author/'gate.json')
    assert r['limits']==dict(cpu=[19,20],**{'as':[2*1024**3]*2})
    assert math.isfinite(r['cpu_seconds']) and r['cpu_seconds']>=0 and r['maxrss_kib']>0
    assert r['zero_calls']==dict(recurrence=0,checkpoint=0,cdf=0,model=0,rng=0,integration=0)
    assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20)
    assert resource.getrlimit(resource.RLIMIT_AS)==(2*1024**3,)*2
    src=Path(p['source']);old=read(src/'application/results.json');app=read(src/'application_protocol.json')
    assert old['complete'] is True and old['completed_boxes']==41472 and app['boundaries_v']==list(B) and app['boxes_per_model']==20736
    assert old['protocol_sha256']==sha(src/'application_protocol.json') and old['gate_sha256']==sha(src/'application_gate.json')
    prior=read(p['prior_review']);prior_scope=read(p['prior_review_scope'])
    assert prior['complete'] is True and prior['rows_verified']==41472 and prior['independent_affine_recurrences']==128
    assert prior['run_scope_sha256']==sha(p['prior_review_scope'])
    inherited={str(src/'application'/f'H8T4seed0_{k:03}.json.gz') for k in range(81)}
    inherited|={str(src/n) for n in ['application/results.json','application_protocol.json','application_gate.json','controls_results.json']}
    assert inherited<=set(prior_scope['pins']) and inherited<=set(p['pins'])
    for path in inherited:assert prior_scope['pins'][path]==p['pins'][path]
    toy=read(p['synthetic_review']);assert toy['complete'] is True and toy['pass_all'] is True and len(toy['rows'])==31
    mass_source=read(src/'controls_results.json');assert mass_source['complete'] is True
    prob=mass_source['probability_setup'];Q=1<<80;den=12*Q**4
    assert prob['bin_denominator']==Q and prob['box_denominator']==Q**4
    bins=prob['bin_numerators'];tail=prob['tail_numerators']
    assert len(bins)==12 and all(len(a)==2 and all(type(v)is int for v in a) and 0<a[0]<=a[1]<=Q for a in bins)
    assert len(tail)==2 and all(type(v)is int for v in tail) and 0<=tail[0]<=tail[1]<=Q**4
    model=next(v for v in old['models'] if v['name']=='H8T4seed0')
    origin=next(v for v in read(app['model_manifest'])['models'] if v['name']=='H8T4seed0')
    assert model['complete'] is True and model['rows']==20736 and len(model['chunks'])==81 and model['mass_denominator']==den
    assert model['model']==origin['model'] and model['checkpoint_sha256']==origin['checkpoint_sha256']
    assert (model['model']['H'],model['model']['T'],model['model']['D'])==(8,4,2147483648)
    assert r['mass_denominator']==den and r['model_identity']==dict(name=model['name'],checkpoint_sha256=model['checkpoint_sha256'],exact_model_sha256=hashlib.sha256(json.dumps(model['model'],sort_keys=True).encode()).hexdigest())
    work=r['inherited_work'];assert work['h8_accounting']==model['accounting']
    for key,value in [('two_model_application_cpu',old['cpu_seconds']),('two_model_certification_attempts',old['certification_attempts']),('primitive_total_cpu',mass_source['cpu_seconds']),('primitive_probability_accounting',mass_source['probability_accounting'])]:assert work[key]==value
    sums={str(g):{key:[0,0] for key in KEYS} for g in G};counts={str(g):Counter() for g in G}
    compressed=0;out['checked_chunks']=[];events=[dict(stage='started')]
    for k in range(81):
        cap();meta=model['chunks'][k];ometa=r['chunks'][k]
        path=src/'application'/f'H8T4seed0_{k:03}.json.gz';dest=author/'output'/f'chunk_{k:03}.json.gz'
        assert meta==dict(file=path.name,first=256*k,last=256*k+255,rows=256,sha256=sha(path))
        assert ometa==dict(file=dest.name,sha256=sha(dest),rows=256)
        data=path.read_bytes();compressed+=len(data)
        original=json.loads(gzip.decompress(data));observed=json.loads(gzip.decompress(dest.read_bytes()))
        assert original['name']=='H8T4seed0' and original['first']==observed['first']==256*k and original['last']==observed['last']==256*k+255
        assert len(original['rows'])==len(observed['rows'])==256
        for n,(raw,row) in enumerate(zip(original['rows'],observed['rows'],strict=True)):
            index=256*k+n;ids=[index//12**power%12 for power in (3,2,1,0)]
            assert row['index']==raw['index']==index and raw['bin_ids']==ids and row['source']==raw
            cert=raw['certificate'];m=cert['margins']
            assert set(m)=={f'{i},{j}' for i in range(4) for j in range(i)}
            assert all(len(v)==2 and all(type(x)is int for x in v) and v[0]<=v[1] for v in m.values())
            assert cert['margin_denominator']==2*(2*2147483648)**4*2147483648
            ss=[status(m,j) for j in range(4)];assert ss==cert['statuses']==row['rechecked_statuses']
            group=[j for j in range(4) if ids[j]==sorted(ids)[-2]];assert group==raw['target_group']
            mm=[1,1]
            for b in ids:mm[0]*=bins[b][0];mm[1]*=bins[b][1]
            assert mm==raw['box_mass_numerators'] and set(row['gaps'])=={'0','1','2','4'}
            z=sums['0'];sum_into(z['inner'],mm,12);sum_into(z['outer'],mm,12)
            for j in group:sum_into(z['network_unknown' if ss[j]=='unknown' else ss[j]],mm,12//len(group))
            assert row['gaps']['0']==dict(kind='whole_event_rank_group',group=group);counts['0']['inner']+=1
            # Inherited v=B equals 2*x. Express PHYSICAL x endpoints in
            # eighths: 8*x=4*B. The old integer threshold g therefore means
            # physical g/8, retaining the failed original g/4 labels separately.
            box=[(4*B[b],4*B[b+1]) for b in ids]
            for g in G[1:]:
                inner,outer=ownership(box,g);assert len(inner)<=1 and all(pair in outer for pair in inner)
                target=inner[0][1] if inner else None
                label=('network_unknown' if ss[target]=='unknown' else ss[target]) if inner else ('boundary_box' if outer else 'outside')
                assert row['gaps'][str(g)]==dict(inner_pairs=inner,outer_pairs=outer,selected=label,target=target)
                z=sums[str(g)]
                if outer:sum_into(z['outer'],mm,12)
                if inner:
                    assert group==[target];sum_into(z['inner'],mm,12);sum_into(z[label],mm,12)
                elif outer:sum_into(z['boundary_box'],mm,12)
                counts[str(g)][label]+=1
        out['checked_chunks'].append(dict(chunk=k,rows=256,original_sha256=sha(path),output_sha256=sha(dest)))
        save(out)
        events.append(dict(stage='chunk_completed',chunk=k,rows=256*(k+1)))
    events.append(dict(stage='completed',rows=20736))
    assert [json.loads(s) for s in (author/'output/progress.jsonl').read_text().splitlines()]==events and len(events)==83
    assert r['raw_sums']==sums and r['row_counts']=={g:dict(v) for g,v in counts.items()}
    assert r['work_counts']==dict(margin_pair_sign_checks=20736*24,gap_ordered_pair_tests=20736*36,saved_probability_product_factors=20736*8,gap_cases_processed=20736*4,compressed_input_bytes=compressed)
    for key,oldkey in [('correct','correct'),('wrong','wrong'),('network_unknown','unknown')]:assert sums['0'][key]==model['raw_mass_sums'][oldkey]
    assert sums['0']['inner']==[12*x for x in model['central_box_sums']]
    summaries={}
    for g in G:
        s=sums[str(g)]
        assert all(sum(s[key][j] for key in ('correct','wrong','network_unknown'))==s['inner'][j] and s['inner'][j]+s['boundary_box'][j]==s['outer'][j] for j in (0,1))
        c=[F(x,den) for x in s['correct']];w=[F(x,den) for x in s['wrong']]
        pbox=[F(1),F(1)] if g==0 else [F(s['inner'][0],den),min(F(1),F(s['outer'][1],den)+F(tail[1],Q**4))]
        u=[max(F(0),pbox[0]-c[1]-w[1]),min(F(1),pbox[1]-c[0]-w[0])]
        assert 0<=u[0]<=u[1]<=1 and 0<=pbox[0]<=pbox[1]<=1
        cond=None if pbox[0]==0 else [str(c[0]/pbox[1]),str((pbox[1]-w[0])/pbox[1])]
        expected=dict(event=list(map(str,pbox)),selected_correct=list(map(str,c)),selected_wrong=list(map(str,w)),inner_network_unknown=ratio(s['network_unknown'],den),boundary_event=['0',str(F(s['boundary_box'][1],den))],outside_cube_event=['0',str(F(tail[1],Q**4))],unresolved_event=list(map(str,u)),conditional_accuracy=cond)
        observed=r['gap_summaries'][str(F(g,4))]
        assert {key:observed[key] for key in expected}==expected
        summaries[str(F(g,8))]=dict(original_report_key=str(F(g,4)),physical_gap=str(F(g,8)),checked=expected)
    for small,big in zip(G,G[1:]):
        for key in ('correct','wrong','inner','outer'):assert all(sums[str(big)][key][j]<=sums[str(small)][key][j] for j in (0,1))
    cap();out.update(complete=True,pass_all=True,diagnostic_only=True,intended_campaign_passed=False,physical_gap_one_tested=False,rows=20736,gap_cases=82944,events=83,gap_summaries=summaries,raw_sums=sums,author_cpu=r['cpu_seconds'],author_maxrss_kib=r['maxrss_kib'],author_work_counts=r['work_counts'],input_pins=completed['pins'],zero_author_calls=True,limitation='Diagnostic contraction at physical gaps 0,1/8,1/4,1/2 only; original intended-gap protocol remains failed. Old probability intervals and affine margin enclosures remain premises; no recurrence, CDF, model, integration, RNG or author calls. Partial gap intersections remain unresolved. No full native cost or performance conclusion.')

if __name__=='__main__':
    result_path=HERE/'results.json';assert not result_path.exists() and not (HERE/'results.tmp').exists(),'Preserve prior attempt'
    result=dict(complete=False,pass_all=False)
    save(result)
    try:run(result)
    except BaseException as e:result.update(complete=False,pass_all=False,error=repr(e),traceback=traceback.format_exc());raise
    finally:
        save(result)
