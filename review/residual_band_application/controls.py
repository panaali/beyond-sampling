"""Fixed rational interval and Gaussian-displacement controls; no target/model archive."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,resource,sys,time,traceback
P=Path(__file__).resolve().parent

def main():
    out=P/'controls_output';assert not out.exists();out.mkdir();start=time.process_time()
    result={'complete':False,'pass_all':False,'rows':[],'events':0,'partition_calls':0,'direct_synthetic_calls':0,'masked_synthetic_calls':0,'mask_setup_calls':0,'guard_attempts':0,'target_calls':0,'rng_calls':0}
    def encode(x):
        if isinstance(x,F):return {'q':[str(x.numerator),str(x.denominator)]}
        if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [encode(v) for v in x]
        return x
    def save():
        result['cpu_seconds']=time.process_time()-start;result['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tmp=out/'results.tmp';tmp.write_text(json.dumps(encode(result),indent=2,allow_nan=False)+'\n');tmp.replace(out/'results.json')
    def event(kind,**kw):
        assert time.process_time()-start<17,'internal CPU cap'
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps({'event':kind,**kw})+'\n')
        result['events']+=1;save()
    def begin(kind,name,**kw):
        row={'kind':kind,'name':name,'passed':False,**kw};result['rows'].append(row);event('started_row',index=len(result['rows'])-1);return row
    def finish(row):row['passed']=True;event('completed_row',index=len(result['rows'])-1)
    try:
        save();protocol=json.loads((P/'controls_protocol.json').read_text());gate=json.loads((P/'controls_gate.json').read_text())
        assert gate['ready'] is True
        for k in ['source_publication','review_publication','execution_publication']:
            v=gate[k];assert isinstance(v,str) and 8<=len(v)<=40 and all(c in '0123456789abcdef' for c in v)
        own=[str(P/n) for n in ['intervals.py','gaussian.py','controls.py','controls_launch.sh','controls_protocol.json','PLAN.md','CONTROLS.md']]
        required=own+[protocol['review_path']]+list(protocol['pins'])
        for f,h in protocol['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
        assert all(k in gate['pins'] for k in required)
        for f,h in gate['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
        review=json.loads(Path(protocol['review_path']).read_text());assert review['complete'] and review['pass_all'] and review['execution_authorized']
        assert all(k in review['pins'] for k in own)
        for f,h in review['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
        assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
        assert str(Path(sys.executable).resolve())==protocol['python_resolved']
        result['runtime']={'python':sys.version,'invoked':sys.executable,'resolved':str(Path(sys.executable).resolve())}
        from intervals import Ledger,partition,errors,signed_difference
        from gaussian import correction
        result['pins']=gate['pins'];event('started')
        zero=((F(0),F(0)),((F(0),F(0)),(F(0),F(0))),((F(1),F(0)),(F(0),F(1))))
        sign=((F(1),F(-1)),zero[1],zero[2]);identity=((F(1),F(0)),(F(0),F(1)))
        recurrent=(sign[0],identity,sign[2]);lo,hi=F(-2),F(2);direction=(F(1),F(-1))
        fixtures=[('zero',zero,(F(0),F(0)),[[],[(lo,hi)]]),('sign',sign,(F(0),F(0)),[[(F(0),hi)],[(lo,F(0))]]),('shifted',sign,(F(0),F(1,2)),[[(F(1,2),hi)],[(lo,F(1,2))]]),('recurrent',recurrent,(F(1,2),F(-1,4)),[[(F(0),hi)],[(lo,F(0))]])]
        probes=[F(-3,2),F(-3,4),F(-1,4),F(0),F(1,4),F(3,4),F(3,2)]
        for name,model,y,expected in fixtures:
            row=begin('network',name,model=model,y=y,direction=direction,window=[lo,hi]);ledger=Ledger()
            result['partition_calls']+=1;save();cells,layers=partition(model,y,direction,lo,hi,ledger);row.update(cells=cells,layers=layers);save()
            result['partition_calls']+=1;save();frozen,flayers=partition(model,y,direction,lo,hi,ledger,frozen=True);row.update(frozen=frozen,frozen_layers=flayers);save()
            error=[errors(cells,k,lo,hi,ledger) for k in range(2)];row['errors']=error;save();assert error==expected
            checks=[];row['probes']=checks
            result['mask_setup_calls']+=1;save();hh=[F(0),F(0)];words=[]
            for t in range(2):
                zz=[model[0][i]*y[t]+sum(model[1][i][j]*hh[j] for j in range(2)) for i in range(2)];word=[z>0 for z in zz];words.append(word);hh=[z if on else F(0) for z,on in zip(zz,word)]
            row['frozen_words']=words;save()
            for x in probes:
                result['direct_synthetic_calls']+=1;save();h=[F(0),F(0)]
                for t in range(2):
                    inp=y[t]+direction[t]*x;h=[max(F(0),model[0][i]*inp+sum(model[1][i][j]*h[j] for j in range(2))) for i in range(2)]
                logits=[sum(w*z for w,z in zip(r,h)) for r in model[2]];winner=max(range(2),key=lambda j:logits[j]);piece=next(z for z in cells if z[0]<=x<=z[1]);affine=[c+s*x for c,s in piece[2]]
                result['masked_synthetic_calls']+=1;save();mh=[F(0),F(0)]
                for t in range(2):
                    inp=y[t]+direction[t]*x;zz=[model[0][i]*inp+sum(model[1][i][j]*mh[j] for j in range(2)) for i in range(2)];mh=[z if on else F(0) for z,on in zip(zz,words[t])]
                masked=[sum(w*z for w,z in zip(r,mh)) for r in model[2]];faffine=[c+s*x for c,s in frozen[0][2]]
                checks.append({'x':x,'logits':logits,'affine':affine,'winner':winner,'masked':masked,'frozen_affine':faffine});save();assert affine==logits and masked==faffine
                for k in range(2):
                    if x not in {v for pair in error[k] for v in pair}:assert any(a<x<b for a,b in error[k])==(winner!=k)
            row.update(ledger=ledger.counts,max_returned_bits=[ledger.max_returned_num_bits,ledger.max_returned_den_bits]);finish(row)
        diffcases=[('overlap',[(1,2),(3,4)],[(1,5)],[(2,3,-1),(4,5,-1)]),('actual_negative',[(-2,0)],[],[(-2,0,1)]),('frozen_negative',[],[(-2,0)],[(-2,0,-1)]),('alternating',[(0,1),(2,3)],[(1,2)],[(0,1,1),(1,2,-1),(2,3,1)])]
        for name,a,b,expected in diffcases:
            a=[tuple(map(F,x)) for x in a];b=[tuple(map(F,x)) for x in b];row=begin('difference',name,actual=a,frozen=b);ledger=Ledger();value=signed_difference(a,b,ledger);row.update(value=value,ledger=ledger.counts);save();assert value==[(F(l),F(r),s) for l,r,s in expected];finish(row)
        for name,a,b in [('near_positive',F(0),F(1,4)),('near_negative',F(-1,4),F(0)),('wide_positive',F(1),F(3)),('wide_negative',F(-3),F(-1))]:
            row=begin('gaussian',name,band=[a,b,1]);value=correction([(a,b,1)]);row['value']=value;save()
            assert abs(value['direct']-value['first'])<=value['remainder1']+3e-15
            assert abs(value['direct']-value['second'])<=value['remainder2']+3e-15
            if name.startswith('wide'):assert value['second']<0
            finish(row)
        guards=['shape','nonfraction','window','truth_boolean','cross_zero','bad_sign']
        for name in guards:
            row=begin('guard',name);ledger=Ledger();result['guard_attempts']+=1;save()
            try:
                if name=='shape':partition((sign[0],((F(0),),),sign[2]),(F(0),F(0)),direction,lo,hi,ledger)
                elif name=='nonfraction':partition(sign,(0,F(0)),direction,lo,hi,ledger)
                elif name=='window':partition(sign,(F(0),F(0)),direction,hi,lo,ledger)
                elif name=='truth_boolean':errors([(lo,hi,[(F(0),F(0)),(F(0),F(0))])],True,lo,hi,ledger)
                elif name=='cross_zero':correction([(F(-1),F(1),1)])
                else:correction([(F(0),F(1),0)])
            except ValueError as exc:row['rejection']=str(exc);save()
            else:raise AssertionError('guard accepted '+name)
            finish(row)
        assert len(result['rows'])==18 and result['partition_calls']==8 and result['direct_synthetic_calls']==28 and result['masked_synthetic_calls']==28 and result['mask_setup_calls']==4 and result['guard_attempts']==6
        event('completed');assert result['events']==38;result['complete']=result['pass_all']=True;save()
    except BaseException:result['failure']=traceback.format_exc();save();raise

if __name__=='__main__':main()
