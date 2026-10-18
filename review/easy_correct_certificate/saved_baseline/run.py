"""Fixed saved-certificate filtering only; no recurrence or probability evaluation."""
from pathlib import Path
from fractions import Fraction as F
from math import prod
from collections import Counter
import gzip, hashlib, json, resource, time, traceback, sys
HERE=Path(__file__).resolve().parent
B=[-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8]
G=[0,1,2,4]  # quarters
N=12**4

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def dump(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def archive(p,v):
    assert not p.exists()
    t=p.with_suffix(p.suffix+'.tmp');t.write_bytes(gzip.compress(json.dumps(v,separators=(',',':'),allow_nan=False).encode(),compresslevel=1,mtime=0));t.replace(p)
def add(a,b,factor=1):
    for k in (0,1): a[k]+=factor*b[k]
def fp(v,den): return [str(F(x,den)) for x in v]
def classify(m,j,c):
    good,bad=True,False
    for o in range(4):
        if o==j:continue
        l,u=m[f'{max(j,o)},{min(j,o)}']
        if j<o:l,u=-u,-l
        c['margin_pair_sign_checks']+=2
        good &= l>0 if o<j else l>=0
        bad |= u<=0 if o<j else u<0
    assert not(good and bad)
    return 'correct' if good else 'wrong' if bad else 'unknown'
def pairs(box,g,c):
    inner,outer=[],[]
    for i in range(4):
        for j in range(4):
            if i==j:continue
            c['gap_ordered_pair_tests']+=1
            rest=[q for q in range(4) if q!=i and q!=j]
            if 4*(box[i][0]-box[j][1])>=g and 4*(box[j][0]-max(box[q][1] for q in rest))>=g:inner.append([i,j])
            lo=max(4*box[j][0],max(4*box[q][0] for q in rest)+g)
            hi=min(4*box[j][1],4*box[i][1]-g)
            if lo<=hi:outer.append([i,j])
    assert len(inner)<=1 and all(q in outer for q in inner)
    return inner,outer

def main():
    started=time.process_time();out=HERE/'output';assert not out.exists(),'no overwrite or resume'
    out.mkdir();sys.set_int_max_str_digits(0)
    result=dict(complete=False,pass_all=False,rows=0,chunks=[],zero_calls=dict(recurrence=0,checkpoint=0,cdf=0,model=0,rng=0,integration=0))
    pending=[];counts=Counter();sums={str(g):{k:[0,0] for k in ('correct','wrong','network_unknown','inner','outer','boundary_box')} for g in G}
    rowcounts={str(g):Counter() for g in G}
    def save():
        result.update(cpu_seconds=time.process_time()-started,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,work_counts=dict(counts),raw_sums=sums,row_counts={g:dict(v) for g,v in rowcounts.items()})
        dump(out/'results.json',result)
    def emit(stage,**kw):
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,**kw),allow_nan=False)+'\n')
    def cap():assert time.process_time()-started<17,'17CPU internal cap'
    try:
        protocol=read(HERE/'protocol.json');gate=read(HERE/'gate.json')
        assert gate['complete'] and gate['execution_authorized'] and gate['protocol_sha256']==sha(HERE/'protocol.json')
        required={str(HERE/'run.py'),str(HERE/'launch.sh'),str(HERE/'PROTOCOL.md'),str(HERE/'protocol.json')}
        assert required<=set(gate['pins'])
        for path,h in {**protocol['pins'],**gate['pins']}.items():assert sha(path)==h,path
        reviewer=read(gate['source_review']);assert reviewer['complete'] and reviewer['pass_all'] and reviewer['execution_authorized']
        assert gate['source_review'] in gate['pins']
        for path,h in reviewer['pins'].items():assert sha(path)==h,path
        assert required<=set(reviewer['pins'])
        prior_review=read(protocol['prior_review']);assert prior_review['complete'] and prior_review['rows_verified']==41472 and prior_review['independent_affine_recurrences']==128
        toy_review=read(protocol['synthetic_review']);assert toy_review['complete'] and toy_review['pass_all'] and len(toy_review['rows'])==31
        limits={k:list(resource.getrlimit(v)) for k,v in [('cpu',resource.RLIMIT_CPU),('as',resource.RLIMIT_AS)]}
        assert limits['cpu']==[19,20] and limits['as']==[2*1024**3]*2
        source=Path(protocol['source']);old=read(source/'application/results.json');assert old['complete'] and old['completed_boxes']==41472
        app=read(source/'application_protocol.json');assert app['boundaries_v']==B and app['boxes_per_model']==N
        oldgate=read(source/'application_gate.json');assert oldgate['complete'] and oldgate['protocol_sha256']==sha(source/'application_protocol.json')
        assert old['protocol_sha256']==sha(source/'application_protocol.json') and old['gate_sha256']==sha(source/'application_gate.json')
        controls=read(source/'controls_results.json');assert controls['complete'];mass=controls['probability_setup'];Q=mass['bin_denominator'];den=12*Q**4
        assert Q==1<<80 and mass['box_denominator']==Q**4
        bins=mass['bin_numerators'];tail=mass['tail_numerators'];assert len(bins)==12 and all(0<=l<=u<=Q for l,u in bins)
        assert len(tail)==2 and 0<=tail[0]<=tail[1]<=Q**4
        model=next(m for m in old['models'] if m['name']=='H8T4seed0');assert model['complete'] and model['rows']==N and len(model['chunks'])==81 and model['mass_denominator']==den
        models=read(app['model_manifest']);pinned_model=next(m for m in models['models'] if m['name']=='H8T4seed0')
        assert model['model']==pinned_model['model'] and model['checkpoint_sha256']==pinned_model['checkpoint_sha256']
        assert model['model']['H']==8 and model['model']['T']==4 and model['model']['D']==2147483648
        result.update(limits=limits,protocol_sha256=sha(HERE/'protocol.json'),gate_sha256=sha(HERE/'gate.json'),mass_denominator=den,model_identity=dict(name=model['name'],checkpoint_sha256=model['checkpoint_sha256'],exact_model_sha256=hashlib.sha256(json.dumps(model['model'],sort_keys=True).encode()).hexdigest()),inherited_work=dict(h8_accounting=model['accounting'],two_model_application_cpu=old['cpu_seconds'],two_model_certification_attempts=old['certification_attempts'],primitive_total_cpu=controls['cpu_seconds'],primitive_probability_accounting=controls['probability_accounting'],qualification='Original whole-process totals are not measured H8-only CPU or complete FLOPs.'))
        emit('started');save()
        for k,meta in enumerate(model['chunks']):
            cap();path=source/'application'/meta['file'];assert meta['file']==f'H8T4seed0_{k:03}.json.gz' and meta['first']==256*k and meta['last']==256*k+255 and meta['rows']==256
            assert str(path) in protocol['pins'] and sha(path)==meta['sha256']==protocol['pins'][str(path)]
            raw=path.read_bytes();counts['compressed_input_bytes']+=len(raw);chunk=json.loads(gzip.decompress(raw))
            assert chunk['name']=='H8T4seed0' and chunk['first']==meta['first'] and chunk['last']==meta['last'] and len(chunk['rows'])==256
            for source_row in chunk['rows']:
                cap();index=result['rows'];ids=[index//12**power%12 for power in (3,2,1,0)];box=[[B[b],B[b+1]] for b in ids]
                record=dict(index=index,source=source_row,gaps={});pending.append(record)
                assert source_row['index']==index and source_row['bin_ids']==ids
                cert=source_row['certificate'];m=cert['margins'];assert set(m)=={f'{i},{j}' for i in range(4) for j in range(i)}
                assert all(len(v)==2 and all(type(x)is int for x in v) and v[0]<=v[1] for v in m.values())
                assert cert['margin_denominator']==2*(2*2147483648)**4*2147483648
                statuses=[classify(m,j,counts) for j in range(4)];assert statuses==cert['statuses'];record['rechecked_statuses']=statuses
                second=sorted(ids,reverse=True)[1];group=[j for j in range(4) if ids[j]==second];assert source_row['target_group']==group
                mm=[prod(bins[j][e] for j in ids) for e in (0,1)];assert source_row['box_mass_numerators']==mm
                counts['saved_probability_product_factors']+=8
                assert 0<mm[0]<=mm[1]
                z=sums['0'];add(z['inner'],mm,12);add(z['outer'],mm,12)
                for j in group:
                    key=statuses[j] if statuses[j]!='unknown' else 'network_unknown';add(z[key],mm,12//len(group))
                record['gaps']['0']=dict(kind='whole_event_rank_group',group=group);rowcounts['0']['inner']+=1
                previous_inner=previous_outer=None
                for g in G[1:]:
                    inner,outer=pairs(box,g,counts);z=sums[str(g)];label='outside';target=None
                    if outer:add(z['outer'],mm,12)
                    if inner:
                        target=inner[0][1];assert group==[target];label=statuses[target] if statuses[target]!='unknown' else 'network_unknown'
                        add(z['inner'],mm,12);add(z[label],mm,12)
                    elif outer:label='boundary_box';add(z[label],mm,12)
                    record['gaps'][str(g)]=dict(inner_pairs=inner,outer_pairs=outer,selected=label,target=target)
                    rowcounts[str(g)][label]+=1
                    if previous_inner is not None:
                        assert all(q in previous_inner for q in inner) and all(q in previous_outer for q in outer)
                    previous_inner,previous_outer=inner,outer
                counts['gap_cases_processed']+=4;result['rows']+=1
            outpath=out/f'chunk_{k:03}.json.gz';archive(outpath,dict(first=256*k,last=256*k+255,rows=pending));pending=[]
            result['chunks'].append(dict(file=outpath.name,sha256=sha(outpath),rows=256));save();emit('chunk_completed',chunk=k,rows=result['rows'])
        assert result['rows']==N and len(result['chunks'])==81 and not pending
        for key,oldkey in [('correct','correct'),('wrong','wrong'),('network_unknown','unknown')]:assert sums['0'][key]==model['raw_mass_sums'][oldkey]
        assert sums['0']['inner']==[12*x for x in model['central_box_sums']]
        result['gap_summaries']={}
        for g in G:
            s=sums[str(g)];assert all(sum(s[key][e] for key in ('correct','wrong','network_unknown'))==s['inner'][e] for e in (0,1))
            assert all(s['inner'][e]+s['boundary_box'][e]==s['outer'][e] for e in (0,1))
            c=[F(x,den) for x in s['correct']];w=[F(x,den) for x in s['wrong']]
            p=[F(1),F(1)] if g==0 else [F(s['inner'][0],den),min(F(1),F(s['outer'][1]+12*tail[1],den))]
            unresolved=[max(F(0),p[0]-c[1]-w[1]),min(F(1),p[1]-c[0]-w[0])]
            assert 0<=p[0]<=p[1]<=1 and 0<=unresolved[0]<=unresolved[1]<=1
            conditional=None if p[0]==0 else [str(c[0]/p[1]),str(1-w[0]/p[1])]
            result['gap_summaries'][str(F(g,4))]=dict(event=list(map(str,p)),selected_correct=list(map(str,c)),selected_wrong=list(map(str,w)),inner_network_unknown=fp(s['network_unknown'],den),boundary_event=['0',str(F(s['boundary_box'][1],den))],outside_cube_event=['0',str(F(tail[1],Q**4))],unresolved_event=list(map(str,unresolved)),conditional_accuracy=conditional,qualification='Whole-box selected method only; partial intersection masses not evaluated.')
        for a,b in zip(G,G[1:]):
            for key in ('correct','wrong','inner','outer'):
                assert all(sums[str(b)][key][e]<=sums[str(a)][key][e] for e in (0,1))
        cap();result.update(complete=True,pass_all=True);save();emit('completed',rows=N);print(json.dumps(dict(complete=True,rows=N,cpu_seconds=result['cpu_seconds'])))
    except BaseException as e:
        if pending:archive(out/'failed_partial.json.gz',dict(rows=pending))
        result.update(complete=False,pass_all=False,failure=repr(e),traceback=traceback.format_exc());save();emit('failed',failure=repr(e));raise
if __name__=='__main__':main()
