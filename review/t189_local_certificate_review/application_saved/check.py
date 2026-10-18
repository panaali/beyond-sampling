"""Saved box audit: exhaustive contraction, fixed128 independent affine recurrences."""
from pathlib import Path
from fractions import Fraction as F
from math import isfinite,prod
from collections import Counter
import gzip,hashlib,json,time,resource,traceback
HERE=Path(__file__).resolve().parent
N=12**4
IDS=[j*(N-1)//63 for j in range(64)]
BOUNDARIES=[-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def eq(a,b):assert a==b,(a,b)
def dot(a,b):return sum((u*v for u,v in zip(a,b,strict=True)),F(0))
def interval(w,bb):
    return (sum((min(v*l,v*u) for v,(l,u) in zip(w,bb,strict=True)),F(0)),sum((max(v*l,v*u) for v,(l,u) in zip(w,bb,strict=True)),F(0)))
def intersect(a,b):
    l,u=max(a[0],b[0]),min(a[1],b[1]);assert l<=u;return l,u

def physical_margins(m,box):
    H,T,D=m['H'],m['T'],m['D'];wi=[F(v,D) for v in m['wi']];wh=[[F(v,D) for v in r] for r in m['wh']];wo=[[F(v,D) for v in r] for r in m['wo']]
    a=[[F(0)]*T for _ in range(H)];e=[(F(0),F(0))]*H;b=e.copy()
    for t in range(T):
        nexta=[];nexte=[];nextb=[]
        for i in range(H):
            coeff=[dot(wh[i],[a[j][q] for j in range(H)]) for q in range(T)];coeff[t]+=wi[i]/2
            rem=interval(wh[i],e);ar=interval(coeff,box);pr=interval(wh[i],b);fresh=interval([wi[i]/2],[box[t]])
            l,u=intersect((ar[0]+rem[0],ar[1]+rem[1]),(pr[0]+fresh[0],pr[1]+fresh[1]))
            if l>=0:k,R=2,F(0)
            elif u<=0:k,R=0,F(0)
            else:
                rs=[2*u,max(-l,u),-2*l];k=min(range(3),key=lambda j:(j*(rem[1]-rem[0])+rs[j],j));R=rs[k]
            nexta.append([F(k,2)*v for v in coeff]);nexte.append((k*rem[0]/2,(k*rem[1]+R)/2));nextb.append((max(F(0),l),max(F(0),u)))
        a,e,b=nexta,nexte,nextb
    margins={}
    for i in range(T):
        for j in range(i):
            w=[x-y for x,y in zip(wo[i],wo[j],strict=True)];coeff=[dot(w,[a[k][q] for k in range(H)]) for q in range(T)]
            er=interval(w,e);ar=interval(coeff,box);margins[f'{i},{j}']=intersect((ar[0]+er[0],ar[1]+er[1]),interval(w,b))
    return margins

def statuses(margins):
    answer=[]
    for j in range(4):
        good=True;bad=False
        for o in range(4):
            if j==o:continue
            l,u=margins[f'{max(j,o)},{min(j,o)}'];assert l<=u
            if j<o:l,u=-u,-l
            if o<j:good &= l>0;bad |= u<=0
            else:good &= l>=0;bad |= u<0
        assert not(good and bad);answer.append('correct' if good else 'wrong' if bad else 'unknown')
    return answer

def crossover(w,cs):
    if not w:return dict(crossover=None,frozen_budget=10**7,sufficient_tenfold=False)
    q=25/w**2;n=q.numerator//q.denominator+1
    return dict(minimum_samples=n,minimum_common_budget=n*cs,frozen_budget=10**7,frozen_samples=10**7//cs,sufficient_tenfold=(10**7//cs)*w*w>25)

def main():
    out=HERE/'results.json';assert not out.exists() and not (HERE/'results.tmp').exists()
    resource.setrlimit(resource.RLIMIT_CPU,(19,20));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));start=time.process_time()
    result=dict(complete=False,author_imports=0,trained_forwards=0,random_draws=0,independent_affine_recurrences=0,rows_verified=0,models=[])
    def save():
        result.update(cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp=HERE/'results.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    try:
        scope=read(HERE/'run_scope.json');assert scope['source_gate_commit']!='pending'
        for name,d in scope['pins'].items():eq(sha(name),d)
        result['run_scope_sha256']=sha(HERE/'run_scope.json');save()
        source=Path(scope['author_directory']);p=read(source/'application_protocol.json');g=read(source/'application_gate.json');r=read(source/'application/results.json')
        for name,d in {**p['pins'],**g['pins']}.items():eq(sha(name),d)
        assert g['complete'] and r['complete'];eq(r['protocol_sha256'],sha(source/'application_protocol.json'));eq(g['protocol_sha256'],r['protocol_sha256']);eq(r['gate_sha256'],sha(source/'application_gate.json'))
        eq(p['review_box_ids'],IDS);eq(r['review_box_ids'],IDS);eq(p['boundaries_v'],BOUNDARIES);eq(p['boxes_per_model'],N);eq(p['chunk_size'],256)
        controls=read(source/'controls_results.json');assert controls['complete'];eq(r['mass_dependency_sha256'],sha(source/'controls_results.json'))
        mass=controls['probability_setup'];bins=mass['bin_numerators'];tail=mass['tail_numerators'];Q=1<<80;den=12*Q**4
        eq(mass['bin_denominator'],Q);eq(mass['box_denominator'],Q**4);eq(len(bins),12)
        models=read(p['model_manifest']);assert models['complete'];eq(len(models['models']),2);eq(len(r['models']),2)
        expected_events=[('started',None,None)]
        for saved,prior,H,cs in zip(r['models'],models['models'],[4,8],[218,682],strict=True):
            name=f'H{H}T4seed0';eq(saved['name'],name);eq(prior['name'],name);eq(saved['model'],prior['model']);eq(saved['checkpoint_sha256'],prior['checkpoint_sha256']);assert saved['complete']
            model=saved['model'];eq((model['H'],model['T']),(H,4));eq(model['D'],{4:134217728,8:2147483648}[H]);eq(saved['mass_denominator'],den)
            assert len(model['wi'])==H and len(model['wh'])==H and all(len(q)==H for q in model['wh']) and len(model['wo'])==4 and all(len(q)==H for q in model['wo'])
            assert all(type(x)is int for x in model['wi']) and all(type(x)is int for key in ['wh','wo'] for row in model[key] for x in row)
            sums={k:[0,0] for k in ['correct','wrong','unknown']};central=[0,0];counts=Counter();done=0;checked=[];eq(len(saved['chunks']),81)
            expected_events.append(('model_started',name,None))
            for chunk_idx,meta in enumerate(saved['chunks']):
                eq(meta['first'],256*chunk_idx);eq(meta['last'],256*chunk_idx+255);eq(meta['rows'],256);eq(meta['file'],name+'_'+str(chunk_idx).zfill(3)+'.json.gz')
                file=source/'application'/meta['file'];eq(sha(file),meta['sha256']);chunk=json.loads(gzip.decompress(file.read_bytes()))
                eq((chunk['name'],chunk['first'],chunk['last']),(name,meta['first'],meta['last']));eq(len(chunk['rows']),256)
                for row in chunk['rows']:
                    eq(row['index'],done);ids=[(done//12**power)%12 for power in [3,2,1,0]];eq(row['bin_ids'],ids)
                    cert=row['certificate'];margins=cert['margins'];eq(set(margins),{f'{a},{b}' for a in range(4) for b in range(a)})
                    assert all(len(v)==2 and all(type(x)is int for x in v) for v in margins.values())
                    eq(cert['margin_denominator'],2*(2*model['D'])**4*model['D']);ss=statuses(margins);eq(cert['statuses'],ss)
                    second=sorted(ids,reverse=True)[1];group=[j for j,b in enumerate(ids) if b==second];eq(row['target_group'],group)
                    lo=prod(bins[j][0] for j in ids);hi=prod(bins[j][1] for j in ids);eq(row['box_mass_numerators'],[lo,hi]);assert 0<lo<=hi
                    for j in group:
                        sums[ss[j]][0]+=(12//len(group))*lo;sums[ss[j]][1]+=(12//len(group))*hi;counts[ss[j]]+=1
                    central[0]+=lo;central[1]+=hi
                    if done in IDS:
                        independent=physical_margins(model,[(F(BOUNDARIES[j]),F(BOUNDARIES[j+1])) for j in ids]);Dmargin=cert['margin_denominator']
                        eq({k:tuple(F(x,Dmargin) for x in v) for k,v in margins.items()},independent);checked.append(done);result['independent_affine_recurrences']+=1
                    done+=1;result['rows_verified']+=1
                expected_events.append(('chunk_completed',name,meta['last']));save()
            eq(done,N);eq(saved['rows'],N);eq(checked,IDS);eq(saved['raw_mass_sums'],sums);eq(saved['central_box_sums'],central);eq(saved['status_rank_counts'],dict(counts))
            eq(central,[sum(a for a,b in bins)**4,sum(b for a,b in bins)**4]);eq(saved['tail_numerators'],tail);eq(saved['tail_denominator'],Q**4)
            for side in [0,1]:eq(sum(v[side] for v in sums.values()),12*central[side])
            masses={k:[F(v,den) for v in values] for k,values in sums.items()};masses['unknown']=[v+F(t,Q**4) for v,t in zip(masses['unknown'],tail,strict=True)]
            eq(saved['mass_intervals_unclipped'],{k:[str(v) for v in values] for k,values in masses.items()});eq(saved['probability_intervals'],{k:[str(v[0]),str(min(F(1),v[1]))] for k,v in masses.items()})
            assert sum(v[0] for v in masses.values())<=1<=sum(v[1] for v in masses.values())
            c,w,u=(masses[k][0] for k in ['correct','wrong','unknown']);eq(saved['accuracy_interval'],[str(c),str(1-w)]);eq(saved['accuracy_width'],str(1-c-w));eq(saved['abstention_lower'],str(u));assert 0<=u<=1-c-w<=1
            compare=crossover(u,cs)
            for k,v in compare.items():eq(saved['sampling_comparison'][k],v)
            result['models'].append(dict(name=name,boxes=done,independent_recurrence_ids=checked,accuracy_interval=saved['accuracy_interval'],accuracy_width=str(1-c-w),abstention_lower=str(u),sampling_comparison=compare));save()
            expected_events.append(('model_completed',name,None))
        expected_events.append(('completed',None,None));events=[json.loads(s) for s in (source/'application/progress.jsonl').read_text().splitlines()]
        eq([(q['stage'],q.get('name'),q.get('last')) for q in events],expected_events);eq(events[-1]['results_sha256'],sha(source/'application/results.json'))
        for k in ['completed_boxes','certification_attempts']:eq(r[k],2*N)
        for k in ['checkpoint_loads','random_draws','sampled_forwards','new_cdf_calls']:eq(r[k],0)
        eq(result['independent_affine_recurrences'],128);eq(result['rows_verified'],2*N)
        assert type(r['cpu_seconds']) in (int,float) and isfinite(r['cpu_seconds']) and 0<=r['cpu_seconds']<=90;assert type(r['maxrss_kib'])is int and r['maxrss_kib']>0
        result.update(complete=True,progress_records=len(events),author_cpu_seconds=r['cpu_seconds'],scope='Exhaustive41472 saved margin/status/rank/mass/coverage contraction plus128 frozen physical-unit affine recurrence checks. Not exhaustive recurrence recomputation; named operation/peak-bit internals retained, not replayed or priced.')
        save()
    except BaseException:
        result['failure']=traceback.format_exc();save();raise
if __name__=='__main__':main()
