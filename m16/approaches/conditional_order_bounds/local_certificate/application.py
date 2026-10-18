"""Frozen all-box application; blocked until separately published dependency gates."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from collections import Counter
import gzip,hashlib,json,time,resource,traceback,sys
HERE=Path(__file__).resolve().parent
N=20736;CHUNK=256

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def archive(p,d):
    assert not p.exists();tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_bytes(gzip.compress(json.dumps(d,separators=(',',':'),allow_nan=False).encode(),compresslevel=1,mtime=0));tmp.replace(p)
def fraction(x):return str(F(x))
def main():
    outdir=HERE/'application';assert not outdir.exists(),'no resume or overwrite'
    outdir.mkdir();sys.set_int_max_str_digits(0);start=time.process_time()
    progress=outdir/'progress.jsonl';result=outdir/'results.json'
    r=dict(complete=False,models=[],completed_boxes=0,certification_attempts=0,checkpoint_loads=0,random_draws=0,sampled_forwards=0,new_cdf_calls=0)
    pending=[];ledger=None;current=None
    def emit(stage,**kw):
        with progress.open('a') as f:f.write(json.dumps(dict(stage=stage,**kw),allow_nan=False)+'\n')
    def save():
        r['cpu_seconds']=time.process_time()-start;r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if current is not None and ledger is not None:current['accounting']=ledger.report()
        dump(result,r)
    def cap():assert time.process_time()-start<75,'75CPU total application soft cap'
    try:
        p=read(HERE/'application_protocol.json');g=read(HERE/'application_gate.json')
        assert g['complete'] and g['protocol_sha256']==sha(HERE/'application_protocol.json')
        for path,h in {**p['pins'],**g['pins']}.items():assert sha(path)==h,path
        review=read(p['primitive_saved_review']);review_scope=read(p['primitive_saved_scope'])
        assert review['complete'] and review['rows']==44 and review['synthetic_point_reconstructions']==153
        assert review['scope_sha256']==sha(p['primitive_saved_scope'])
        for path,h in review_scope['pins'].items():assert sha(path)==h,path
        controls=read(HERE/'controls_results.json');assert controls['complete'] and len(controls['rows'])==44
        assert all(row['passed'] for row in controls['rows'])
        assert controls['synthetic_exact_forwards']==153 and controls['candidate_certification_attempts']==12
        mass=controls['probability_setup'];Q=mass['bin_denominator'];assert Q==1<<80
        bins=mass['bin_numerators'];tail=mass['tail_numerators'];assert len(bins)==12 and mass['box_denominator']==Q**4
        boundaries=p['boundaries_v'];assert boundaries==[-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8]
        assert p['boxes_per_model']==N and p['chunk_size']==CHUNK and N%CHUNK==0
        saved=read(p['model_manifest']);assert saved['complete']
        assert [m['name'] for m in saved['models']]==['H4T4seed0','H8T4seed0']
        assert p['review_box_ids']==[j*(N-1)//63 for j in range(64)]
        from enclosure import Ledger,certify,target_group,validate
        from probability import box_mass,crossover
        r.update(protocol_sha256=sha(HERE/'application_protocol.json'),gate_sha256=sha(HERE/'application_gate.json'),mass_dependency_sha256=sha(HERE/'controls_results.json'),review_box_ids=p['review_box_ids'])
        emit('started');save()
        den=12*Q**4
        for model_record,H,cs in zip(saved['models'],[4,8],[218,682],strict=True):
            cap();model=model_record['model'];assert model['H']==H and model['T']==4
            assert model['D']=={4:134217728,8:2147483648}[H]
            validate(model,[[-8,-6]]*4)
            ledger=Ledger();pending=[];sums={k:[0,0] for k in ['correct','wrong','unknown']};central=[0,0];counts=Counter()
            current=dict(name=model_record['name'],model=model,checkpoint_sha256=model_record['checkpoint_sha256'],complete=False,rows=0,chunks=[],status_rank_counts={},raw_mass_sums=sums,mass_denominator=den,central_box_sums=central)
            r['models'].append(current);emit('model_started',name=current['name']);save()
            for index,ids_tuple in enumerate(product(range(12),repeat=4)):
                cap();ids=list(ids_tuple);box=[[boundaries[j],boundaries[j+1]] for j in ids]
                row=dict(index=index,bin_ids=ids,certificate=None);pending.append(row)
                r['certification_attempts']+=1
                certificate=certify(model,box,ledger,retain_trace=False);row['certificate']=certificate
                group=target_group(ids);lo,hi=box_mass(ids,bins);factor=12//len(group);assert factor*len(group)==12
                row.update(target_group=group,box_mass_numerators=[lo,hi])
                for target in group:
                    status=certificate['statuses'][target];assert status in sums
                    sums[status][0]+=factor*lo;sums[status][1]+=factor*hi;counts[status]+=1
                central[0]+=lo;central[1]+=hi;current['rows']+=1;r['completed_boxes']+=1
                if len(pending)==CHUNK:
                    path=outdir/(current['name']+'_'+str(index//CHUNK).zfill(3)+'.json.gz')
                    archive(path,dict(name=current['name'],first=index+1-CHUNK,last=index,rows=pending))
                    current['chunks'].append(dict(file=path.name,sha256=sha(path),first=index+1-CHUNK,last=index,rows=CHUNK))
                    pending=[];current['status_rank_counts']=dict(counts);save();emit('chunk_completed',name=current['name'],last=index,sha256=sha(path))
            assert not pending and current['rows']==N and len(current['chunks'])==81
            assert all(sum(sums[k][i] for k in sums)==12*central[i] for i in [0,1])
            assert central[0]==sum(a for a,b in bins)**4 and central[1]==sum(b for a,b in bins)**4
            assert central[0]+tail[0]<=Q**4<=central[1]+tail[1]
            masses={k:[F(v[0],den),F(v[1],den)] for k,v in sums.items()}
            masses['unknown'][0]+=F(tail[0],Q**4);masses['unknown'][1]+=F(tail[1],Q**4)
            assert all(0<=v[0]<=v[1] for v in masses.values())
            assert sum(v[0] for v in masses.values())<=1<=sum(v[1] for v in masses.values())
            correct=masses['correct'][0];wrong=masses['wrong'][0];unknown=masses['unknown'][0]
            assert 0<=correct<=1-wrong<=1 and 0<=unknown<=1
            current.update(complete=True,tail_numerators=tail,tail_denominator=Q**4,mass_intervals_unclipped={k:list(map(fraction,v)) for k,v in masses.items()},probability_intervals={k:[fraction(v[0]),fraction(min(F(1),v[1]))] for k,v in masses.items()},accuracy_interval=[fraction(correct),fraction(1-wrong)],accuracy_width=fraction(1-wrong-correct),abstention_lower=fraction(unknown),sampling_comparison=crossover(unknown,cs),status_rank_counts=dict(counts))
            save();emit('model_completed',name=current['name'],rows=N)
        assert r['completed_boxes']==2*N and r['certification_attempts']==2*N
        assert all(m['complete'] for m in r['models']);cap();r['complete']=True;save();emit('completed',results_sha256=sha(result))
        print(json.dumps(dict(complete=True,boxes=r['completed_boxes'],cpu_seconds=r['cpu_seconds'],models=[dict(name=m['name'],abstention_lower=m['abstention_lower'],sampling_comparison=m['sampling_comparison']) for m in r['models']])))
    except BaseException as e:
        if pending:archive(outdir/'failed_partial.json.gz',dict(name=None if current is None else current['name'],rows=pending))
        r.update(failure=repr(e),traceback=traceback.format_exc());save();emit('failed',error=repr(e));raise
if __name__=='__main__':main()
