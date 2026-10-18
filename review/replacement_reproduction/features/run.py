"""Fixed synthetic helper controls; no fitting/network/data generation."""
from pathlib import Path
import ast,hashlib,json,math,os,resource,sys,time,traceback
P=Path(__file__).resolve().parent;OUT=P/'output'
FILES=['core.py','fixtures.py','reference.py','run.py','launch.sh','PROTOCOL.md','protocol.json','manifest.json']

def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def verify(pins):
    assert isinstance(pins,dict) and pins
    for f,h in pins.items():assert len(h)==64 and digest(f)==h,f

def main():
    assert not OUT.exists() or not any(OUT.iterdir()),'refuse existing output'
    OUT.mkdir(exist_ok=True);start=time.process_time()
    result={'complete':False,'pass_all':False,'rows':[],'candidate_attempts':0,'candidate_completed':0,'reference_attempts':0,'reference_completed':0,'guard_attempts':0,'events':0,'network_calls':0,'fit_calls':0,'rng_calls':0}
    def raw(x):
        if hasattr(x,'dtype') and hasattr(x,'shape'):
            return {'dtype':str(x.dtype),'shape':list(x.shape),'values':raw(x.tolist())}
        if isinstance(x,dict):return {k:raw(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [raw(v) for v in x]
        if isinstance(x,float) and not math.isfinite(x):return {'nonfinite':str(x)}
        return x
    def save():
        result['cpu_seconds']=time.process_time()-start;result['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tmp=OUT/'results.tmp';tmp.write_text(json.dumps(raw(result),indent=2,allow_nan=False)+'\n');tmp.replace(OUT/'results.json')
    def cap():assert time.process_time()-start<15,'internal CPU cap'
    def event(kind,**kw):
        cap()
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'event':kind,**kw})+'\n')
        result['events']+=1;save()
    def call(row,which,fun,args):
        cap();result[which+'_attempts']+=1;save();value=fun(*args);result[which+'_completed']+=1;row[which]=value;save();return value
    try:
        save();p=read(P/'protocol.json');g=read(P/'gate.json')
        assert g['ready'] is True
        required=[str(P/f) for f in FILES]+[p['written_review'],p['source_review']]
        assert all(f in g['pins'] for f in required)
        for key in ['source_publication','source_review_publication','execution_scope_publication']:
            v=g[key];assert isinstance(v,str) and 8<=len(v)<=40 and all(c in '0123456789abcdef' for c in v)
        verify(p['pins']);verify(g['pins']);verify(read(P/'manifest.json')['pins'])
        wr=read(p['written_review']);assert wr['complete'] and wr['pass_all'] and wr['execution_allowed'] is False;verify(wr['pins'])
        sr=read(p['source_review']);assert sr['complete'] and sr['pass_all'] and sr['execution_authorized'];assert all(f in sr['pins'] for f in required[:-1]);verify(sr['pins'])
        assert resource.getrlimit(resource.RLIMIT_CPU)==(16,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
        assert str(Path(sys.executable).resolve())==p['runtime']['python_resolved']
        import numpy as np
        assert np.__version__==p['runtime']['numpy_version'] and str(Path(np.__file__).resolve())==p['runtime']['numpy_init']
        import core,fixtures,reference
        result.update(pins=g['pins'],protocol_pins=p['pins'],limits={'cpu':[16,20],'AS':[2**31,2**31]},runtime={'python':sys.version,'numpy':np.__version__})
        ns,tree=reference.extract(p['owner_encode']);result['reference_extraction_ast']=tree
        x,h=fixtures.inputs();assert x.shape==(12,10) and h.shape==(12,11,4);result['fixed_inputs']={'x':x,'hinges':h};event('started')
        for t in range(1,11):
            for f in range(12):
                row={'kind':'features','t':t,'family':f,'status':'started'};result['rows'].append(row);event('started_row',index=len(result['rows'])-1)
                a=call(row,'candidate',core.features,(x,t,f,h));b=call(row,'reference',reference.design,(ns,x,t,f,h))
                same_cells=np.array_equal(a[0],b[0]);delta=float(np.max(np.abs(a[1]-b[1])/np.maximum(1.,np.abs(b[1]))));exact=np.array_equal(a[1],b[1])
                row.update(cells_equal=same_cells,base_numeric_equal=exact,normalized_max_abs=delta);save()
                assert a[0].dtype==b[0].dtype==np.dtype('int64') and a[1].dtype==b[1].dtype==np.dtype('float64')
                assert same_cells and (delta<=4e-15 if f==1 else exact)
                row['status']='passed';event('completed_row',index=len(result['rows'])-1)
        for label,scores in fixtures.selections():
            row={'kind':'selection','label':label,'scores':scores,'status':'started'};result['rows'].append(row);event('started_row',index=len(result['rows'])-1)
            a=call(row,'candidate',core.select,(scores,));b=call(row,'reference',reference.choices,(scores,));assert all(np.array_equal(u,v) for u,v in zip(a,b))
            row['status']='passed';event('completed_row',index=len(result['rows'])-1)
        for label,a,c,w in fixtures.endpoints():
            row={'kind':'endpoint','label':label,'inputs':{'predictions':a,'choices':c,'readout':w},'status':'started'};result['rows'].append(row);event('started_row',index=len(result['rows'])-1)
            got=call(row,'candidate',core.assemble,(a,c,w));ref=call(row,'reference',reference.endpoint,(a,c,w));assert set(got)==set(ref) and all(np.array_equal(got[k],ref[k]) and got[k].dtype==ref[k].dtype for k in got)
            row['status']='passed';event('completed_row',index=len(result['rows'])-1)
        for label in fixtures.guards():
            xx=x.copy();hh=h.copy();tt=10;ff=0;fun=core.features
            if label=='x_dtype':xx=xx.astype(np.float32)
            elif label=='x_shape':xx=xx[:,:9]
            elif label=='x_infinite':xx[0,0]=float('inf')
            elif label=='t_zero':tt=0
            elif label=='t_boolean':tt=True
            elif label=='family_range':ff=12
            elif label=='hinge_shape':hh=hh[:,:10]
            elif label=='hinge_dtype':hh=hh.astype(np.float64)
            args=(xx,tt,ff,hh)
            if label=='score_nonfinite':fun=core.select;scores=np.zeros((12,16));scores[0,0]=float('nan');args=(scores,)
            if label=='choice_range':fun=core.assemble;c=np.zeros(16,dtype=np.int64);c[0]=12;args=(np.zeros((12,1,16)),c,np.zeros((10,16)))
            row={'kind':'guard','label':label,'arguments':args,'status':'started'};result['rows'].append(row);event('started_row',index=len(result['rows'])-1);result['guard_attempts']+=1;save()
            try:fun(*args)
            except ValueError as exc:row['rejection']=str(exc);save()
            else:raise AssertionError('accepted invalid input '+label)
            row['status']='passed';event('completed_row',index=len(result['rows'])-1)
        assert result['candidate_attempts']==result['candidate_completed']==result['reference_attempts']==result['reference_completed']==128
        assert result['guard_attempts']==10 and len(result['rows'])==138 and all(r['status']=='passed' for r in result['rows'])
        event('completed');assert result['events']==278
        result['complete']=result['pass_all']=True;save()
    except BaseException:
        result.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
