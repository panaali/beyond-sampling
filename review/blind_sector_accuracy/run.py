"""One bounded synthetic-control plus saved reference contraction; no model calls."""
from pathlib import Path
from fractions import Fraction as F
import gzip,hashlib,json,math,resource,sys,time,traceback
P=Path(__file__).resolve().parent;OUT=P/'output'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def enc(x):
    if isinstance(x,F):return {'q':[str(x.numerator),str(x.denominator)]}
    if isinstance(x,dict):return {str(k):enc(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [enc(v) for v in x]
    return x

def main():
    assert not OUT.exists();OUT.mkdir();start=time.process_time()
    r={'complete':False,'pass_all':False,'controls':[],'events':0,'rows':0,'chunks':[],'orthant_attempts':0,'law_attempts':0,'model_calls':0,'geometry_generated':0,'mass_calls':0,'rng_calls':0}
    def cap():assert time.process_time()-start<17,'internal CPU cap'
    def atomic(name,obj):
        temp=OUT/(name+'.tmp');temp.write_text(json.dumps(enc(obj),separators=(',',':'),allow_nan=False)+'\n');temp.replace(OUT/(name+'.json'))
    def save():
        r.update(cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic('results',r)
    def event(name,**kw):
        cap()
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'event':name,**kw})+'\n')
        r['events']+=1;save()
    def begin(kind,name,**kw):
        c={'kind':kind,'name':name,'passed':False,**kw};r['controls'].append(c);event('control_started',index=len(r['controls'])-1);return c
    def end(c):c['passed']=True;event('control_completed',index=len(r['controls'])-1)
    try:
        save();protocol=read(P/'protocol.json');gate=read(P/'gate.json');review=read(protocol['source_review'])
        assert gate['ready'] is True and all(review[k] is True for k in ['complete','pass_all','execution_authorized'])
        for k in ['source_publication','review_publication','execution_publication']:assert type(gate[k])is str and len(gate[k])>=8
        own={str(P/n) for n in ['sector.py','fixtures.py','run.py','launch.sh','PROTOCOL.md','protocol.json']}
        assert own<=set(review['pins']) and own|set(protocol['pins'])|{protocol['source_review']}<=set(gate['pins'])
        for pins in [protocol['pins'],gate['pins'],review['pins']]:
            for f,h in pins.items():assert sha(f)==h,f
        assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
        assert str(Path(sys.executable).resolve())==protocol['python_resolved']
        r['pins']=gate['pins'];r['runtime']={'invoked':sys.executable,'resolved':str(Path(sys.executable).resolve()),'python':sys.version}
        from sector import orthant,binary_law
        from fixtures import cases
        event('started');valid,guards,laws,invalid=cases();p=F(1,16)
        for name,rays,key,det in valid:
            c=begin('orthant',name,rays=rays,expected=[key,det]);r['orthant_attempts']+=1;save();value=orthant(rays);c['value']=value;save();assert value==(key,det);end(c)
        for name,rays,msg in guards:
            c=begin('orthant_guard',name,rays=rays);r['orthant_attempts']+=1;save()
            try:value=orthant(rays)
            except ValueError as exc:c['rejection']=str(exc);save();assert str(exc)==msg
            else:c['unexpected_return']=value;save();raise AssertionError('guard accepted')
            end(c)
        for name,lo,hi,losslo,losshi in laws:
            c=begin('law',name,interval=[lo,hi],expected_loss=[losslo,losshi]);r['law_attempts']+=1;save();value=binary_law(lo,hi);c['value']=value;save();assert value['best_binary_loss']==[losslo,losshi] and value['error']==[p-hi,p-lo] and value['conditional_correct']==[lo/p,hi/p];end(c)
        for name,lo,hi in invalid:
            c=begin('law_guard',name,interval=[lo,hi]);r['law_attempts']+=1;save()
            try:value=binary_law(lo,hi)
            except ValueError as exc:c['rejection']=str(exc);save();assert str(exc)=='physical probability interval'
            else:c['unexpected_return']=value;save();raise AssertionError('guard accepted')
            end(c)
        assert len(r['controls'])==17 and all(c['passed'] for c in r['controls']);cap()
        data=json.loads(gzip.decompress(Path(protocol['geometry']).read_bytes()));ref=read(protocol['reference'])
        assert ref['complete'] is True and ref['H']==ref['T']==4 and ref['exact_artifact_sha256']==protocol['pins'][protocol['geometry']]
        simplices=data['correct_simplices'];bounds=ref['per_simplex_intervals'];assert len(simplices)==len(bounds)==ref['simplices']==5058
        r['checkpoint_sha256']=data['checkpoint_sha256'];r['inherited_reference_cost']=ref['cold_cost'];r['inherited_reference_interval']=ref['accuracy']['probability_interval'];r['reference_classification']=ref['classification'];save()
        totals={k:{'count':0,'lower':F(0),'upper':F(0)} for k in range(16)};rawlo=rawhi=F(0);contraction_start=time.process_time()
        for chunk,first in enumerate(range(0,5058,256)):
            event('chunk_started',index=chunk,first=first);file=OUT/f'chunk_{chunk:02d}.jsonl';assert not file.exists()
            with file.open('x') as stream:
                for i in range(first,min(first+256,5058)):
                    cap();item=simplices[i];bound=bounds[i];active={'index':i,'source_simplex':item,'source_bound':bound,'phase':'before_orthant'};atomic('active',active)
                    rays=[[int(x) for x in row] for row in item['integer_rays']];assert all(str(v)==x for row,src in zip(rays,item['integer_rays'],strict=True) for v,x in zip(row,src,strict=True))
                    r['orthant_attempts']+=1;key,det=orthant(rays);active.update(sector=key,determinant=det);atomic('active',active)
                    assert det==F(*map(int,item['normalized_absolute_determinant']))
                    qlo,qhi=bound['quadrature'];err=bound['truncation_upper'];assert all(type(v)is float and math.isfinite(v) for v in [qlo,qhi,err]) and 0<=qlo<=qhi and err>=0
                    low=F.from_float(qlo)-F.from_float(err);high=F.from_float(qhi)+F.from_float(err);rawlo+=low;rawhi+=high
                    cl,cu=max(F(0),low),min(p,high);assert cl<=cu
                    row={'index':i,'sector':key,'normalized_determinant':det,'raw_interval':[low,high],'physical_interval':[cl,cu]};active.update(phase='completed',value=row);atomic('active',active)
                    stream.write(json.dumps(enc(row),separators=(',',':'))+'\n');stream.flush();totals[key]['count']+=1;totals[key]['lower']+=cl;totals[key]['upper']+=cu;r['rows']+=1
            r['chunks'].append({'index':chunk,'first':first,'rows':min(256,5058-first),'path':file.name,'sha256':sha(file)});event('chunk_completed',index=chunk)
        r['contraction_cpu']=time.process_time()-contraction_start
        # Exact summation of stored outward component endpoints fits inside the broader historical floating sum.
        reflo,refhi=map(F.from_float,ref['accuracy']['probability_interval']);assert reflo<=rawlo<=rawhi<=refhi
        r['raw_global_interval']=[rawlo,rawhi];sectors=[]
        for key in range(16):
            t=totals[key];lo=max(F(0),t['lower']);hi=min(p,t['upper']);r['law_attempts']+=1;law=binary_law(lo,hi)
            sectors.append({'sector':key,'positive_positions':[j for j in range(4) if key&(1<<j)],'simplex_count':t['count'],'unclipped_sum':[t['lower'],t['upper']],'sector_mass':p,**law})
        blind=[x for x in sectors if x['sector'] in [0,1,2,4,8]];mass=5*p;correct=[sum(x['correct'][i] for x in blind) for i in range(2)];loss=[sum(x['best_binary_loss'][i] for x in blind) for i in range(2)]
        r.update(sectors=sectors,blind={'sector_ids':[0,1,2,4,8],'mass':mass,'correct':correct,'error':[mass-correct[1],mass-correct[0]],'conditional_accuracy':[x/mass for x in correct],'best_binary_loss':loss,'nonconstant_sector_ids':[x['sector'] for x in blind if x['strictly_nonconstant']]});save()
        assert r['rows']==5058 and len(r['chunks'])==20 and r['orthant_attempts']==5066 and r['law_attempts']==25
        event('completed');assert r['events']==76;r['complete']=r['pass_all']=True;save()
    except BaseException:r['failure']=traceback.format_exc();save();raise

if __name__=='__main__':main()
