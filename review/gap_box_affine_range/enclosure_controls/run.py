"""One gated fixed synthetic integration attempt; no old candidate/forward calls."""
from pathlib import Path
from fractions import Fraction as F
import copy,hashlib,json,os,resource,time,traceback
HERE=Path(__file__).resolve().parent
OUT=HERE/'output'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def serial(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return list(map(serial,x))
    return x
def rational_tree(x):
    if isinstance(x,dict):return {k:rational_tree(v) for k,v in x.items()}
    if isinstance(x,list):return list(map(rational_tree,x))
    return F(x)
def verify(pins):
    assert isinstance(pins,dict) and pins
    for p,h in pins.items():
        assert isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h)
        assert sha(p)==h,p

def main():
    assert not OUT.exists() or not any(OUT.iterdir()),'refuse all existing output'
    OUT.mkdir(exist_ok=True);start=time.process_time()
    result=dict(complete=False,pass_all=False,rows=[],events=0,certificate_attempts=0,certificate_returns=0,synthetic_propagations_completed=0,normalization_attempts=0,normalization_returns=0,guard_rejections=0,old_candidate_calls=0,direct_point_forwards=0,trained_model_calls=0,cdf_calls=0,rng_calls=0)
    def save():
        result['cpu_seconds']=time.process_time()-start;result['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tmp=OUT/'results.tmp';tmp.write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n');os.replace(tmp,OUT/'results.json')
    def cap():assert time.process_time()-start<15,'15 CPU internal cap'
    def event(kind,**kw):
        cap()
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,**kw))+'\n')
        result['events']+=1;save()
    def begin(kind,name,inputs):
        row=dict(kind=kind,name=name,input=inputs,status='started');result['rows'].append(row);save();event('started_case',name=name);return row
    def finish(row):row['status']='passed';save();event('completed_case',name=row['name'])
    def certificate(row,fn,args):
        cap();result['certificate_attempts']+=1;row['attempted']=True;save()
        value=fn(**args);row['certificate']=value;result['certificate_returns']+=1
        if value['status']=='feasible':result['synthetic_propagations_completed']+=1
        save();return value
    try:
        save();p=read(HERE/'protocol.json');g=read(HERE/'gate.json')
        direct=[HERE/n for n in ('range_primitive.py','enclosure.py','fixtures.py','run.py','launch.sh','PROTOCOL.md','protocol.json')]
        mandatory=direct+[Path(p[k]) for k in ('source_review','theorem_review','prior_review','prior_scope','prior_results','primitive_review')]
        assert g['ready'] is True and all(str(x) in g['pins'] for x in mandatory)
        for key in ('source_review_commit','source_review_publication','execution_scope_publication'):
            s=g[key];assert isinstance(s,str) and len(s)>=8 and all(c in '0123456789abcdef' for c in s)
        verify(p['pins']);verify(g['pins'])
        review=read(p['source_review']);assert review['complete'] is True and review['pass_all'] is True and review['execution_authorized'] is True
        assert all(str(x) in review['pins'] for x in direct);verify(review['pins'])
        theorem=read(p['theorem_review']);assert theorem['complete'] is True and theorem['pass_all'] is True
        primitive_review=read(p['primitive_review']);assert primitive_review['complete'] is True and primitive_review['pass_all'] is True
        prior_scope=read(p['prior_scope']);verify(prior_scope['pins'])
        assert p['prior_results'] in prior_scope['pins']
        prior_review=read(p['prior_review']);assert prior_review['complete'] is True and prior_review['rows']==44 and prior_review['synthetic_point_reconstructions']==153
        assert prior_review['scope_sha256']==sha(p['prior_scope'])
        cpu=resource.getrlimit(resource.RLIMIT_CPU);mem=resource.getrlimit(resource.RLIMIT_AS)
        assert cpu==(16,20) and mem==(2*1024**3,)*2
        result.update(limits=dict(cpu=cpu,address_space=mem),pins=g['pins'],protocol_pins=p['pins'],prior_scope_sha256=sha(p['prior_scope']))
        event('started')
        from enclosure import certify,normalize_saved
        from fixtures import inherited_fixtures,INHERITED_NAMES,SIGNED_MODEL,BASELINES,DOMAINS,NORMALIZATIONS,GUARDS
        inherited=read(p['prior_results']);assert inherited['complete'] is True
        oldrows=[r for r in inherited['rows'] if r['kind']=='network'];fs=inherited_fixtures()
        assert [r['fixture']['name'] for r in oldrows]==[f['name'] for f in fs]==INHERITED_NAMES
        assert len(fs)==9
        for f,old in zip(fs,oldrows,strict=True):
            cap();assert old['fixture']==f and old['passed'] is True
            physicalbox=[[F(v,2) for v in iv] for iv in f['box']]
            row=begin('inherited',f['name'],dict(fixture=f,physical_box=physicalbox))
            new=certificate(row,certify,dict(model=f['model'],box=physicalbox));saved=old['certificate']
            assert new['status']=='feasible' and len(new['trace'])==len(saved['trace'])==4
            compared=0
            for ns,os_ in zip(new['trace'],saved['trace'],strict=True):
                assert ns['step']==os_['step'] and len(ns['rows'])==len(os_['rows'])==f['model']['H']
                for nr,orr in zip(ns['rows'],os_['rows'],strict=True):
                    for phase in ('pre','post'):
                        den=os_[phase+'_denominator'];assert type(den)is int and den>0
                        assert nr[phase+'_constant']==0
                        assert nr[phase+'_A']==[F(2*a,den) for a in orr[phase+'_A']]
                        assert nr[phase+'_error']==[F(a,den) for a in orr[phase+'_error']]
                        assert nr[phase+'_range']==[F(a,den) for a in orr[phase+'_range']]
                    assert nr['kind']==orr['kind'] and nr['slope']==F(orr['slope_multiplier'],2)
                    assert nr['remainder_upper']==F(orr['remainder_upper'],os_['post_denominator']);compared+=1
            den=saved['margin_denominator'];assert type(den)is int and den>0
            expected={k:[F(a,den) for a in v] for k,v in saved['margins'].items()}
            assert new['raw_margins']==new['final_margins']==expected
            assert new['statuses']==new['raw_statuses']==saved['statuses']
            row['compared_trace_rows']=compared;row['saved_margin_denominator']=den;save();finish(row)
        baseline={}
        for f in BASELINES:
            row=begin('baseline',f['name'],dict(model=SIGNED_MODEL,box=f['box']))
            baseline[f['name']]=certificate(row,certify,dict(model=SIGNED_MODEL,box=f['box']))
            if f['name']=='partial_parent':assert row['certificate']['statuses']==['unknown','unknown','wrong','wrong']
            else:assert row['certificate']['statuses']==['correct','wrong','wrong','wrong']
            finish(row)
        boxes={f['name']:f['box'] for f in BASELINES}
        for f in DOMAINS:
            old=baseline[f['baseline']];row=begin('domain',f['name'],dict(fixture=f,model=SIGNED_MODEL,box=boxes[f['baseline']],old_margins=old['raw_margins']))
            new=certificate(row,certify,dict(model=SIGNED_MODEL,box=boxes[f['baseline']],domain=f['domain'],old_margins=old['raw_margins']))
            if f['expected']=='empty':assert new['status']=='empty' and new['trace']==[] and new['statuses'] is None
            else:
                assert new['status']=='feasible' and new['statuses'][f['target']]==f['expected']
                # Exact analytic logits for x3<=0 are (0,-x3,0,x3).
                l,u=map(F,f['last_range']);coeff=[0,-1,0,1];expected={}
                for i in range(4):
                    for j in range(i):
                        a=coeff[i]-coeff[j];expected[f'{i},{j}']=[a*l,a*u] if a>=0 else [a*u,a*l]
                row['analytic_margins']=expected;save()
                assert new['raw_margins']==new['final_margins']==expected
                assert new['old_margins']==old['raw_margins']
                for a,b in zip(old['statuses'],new['statuses'],strict=True):assert a=='unknown' or a==b
                if f['name']=='partial_negative_final':assert old['statuses'][1]=='unknown' and new['raw_statuses'][1]=='correct'
                row['rank_label_note']='g0 supplied target only, task-rank tie ownership not inferred' if F(f['domain']['gap'])==0 else 'strict positive gaps identify the supplied second label'
                save()
            finish(row)
        for f in NORMALIZATIONS:
            row=begin('normalization',f['name'],f);result['normalization_attempts']+=1;save();cap()
            value=normalize_saved(**f['args']);row['normalized']=value;result['normalization_returns']+=1;save()
            assert value==rational_tree(f['expected']);finish(row)
        for name in GUARDS:
            model=copy.deepcopy(SIGNED_MODEL);box=copy.deepcopy(BASELINES[0]['box']);args=dict(model=model,box=box);fn=certify
            if name=='zero_model_denominator':model['D']=0
            elif name=='boolean_dimension':model['H']=True
            elif name=='reversed_box':box[0]=[2,1]
            elif name=='float_weight':model['wi'][0]=1.0
            elif name=='same_domain_labels':args['domain']=dict(leader=0,second=0,gap=0)
            elif name=='zero_normalization_denominator':fn=normalize_saved;args=dict(NORMALIZATIONS[0]['args'],denominator=0)
            else:raise AssertionError(name)
            row=begin('guard',name,args);row['attempted']=True
            counter='normalization_attempts' if fn is normalize_saved else 'certificate_attempts';result[counter]+=1;save();cap()
            try:value=fn(**args)
            except ValueError as e:row['error']=str(e);result['guard_rejections']+=1;save()
            else:row['unexpected_result']=value;save();raise AssertionError('guard accepted '+name)
            finish(row)
        assert len(result['rows'])==24 and all(r['status']=='passed' for r in result['rows'])
        assert (result['certificate_attempts'],result['certificate_returns'],result['synthetic_propagations_completed'],result['normalization_attempts'],result['normalization_returns'],result['guard_rejections'])==(20,15,14,4,3,6)
        event('completed');assert result['events']==50;result['complete']=result['pass_all']=True;save()
    except BaseException:
        result['complete']=result['pass_all']=False;result['failure']=traceback.format_exc();save();raise
if __name__=='__main__':main()
