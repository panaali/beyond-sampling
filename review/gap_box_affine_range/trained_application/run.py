"""One separately gated eight-certificate trained-domain discriminator."""
from pathlib import Path
from fractions import Fraction as F
import gzip, hashlib, json, os, resource, time, traceback
HERE=Path(__file__).resolve().parent
OUT=HERE/'output'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if isinstance(x,F): return str(x)
    if isinstance(x,dict): return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return list(map(serial,x))
    return x
def verify(pins):
    assert isinstance(pins,dict) and pins
    for p,h in pins.items():
        assert isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h)
        assert sha(p)==h,p
def hexid(s): return isinstance(s,str) and len(s)>=8 and all(c in '0123456789abcdef' for c in s)
def main():
    assert not OUT.exists() or not any(OUT.iterdir()),'refuse any existing output'
    OUT.mkdir(exist_ok=True);start=time.process_time()
    r=dict(complete=False,pass_all=False,rows=[],events=0,certificate_attempts=0,certificate_returns=0,whole_box_calls=0,restricted_calls=0,model_manifest_decodes=0,selected_archive_decodes=0,old_candidate_calls=0,checkpoint_loads=0,direct_point_forwards=0,cdf_calls=0,rng_calls=0,mass_calls=0)
    def save():
        r['cpu_seconds']=time.process_time()-start;r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t=OUT/'results.tmp';t.write_text(json.dumps(serial(r),indent=2,allow_nan=False)+'\n');os.replace(t,OUT/'results.json')
    def cap(): assert time.process_time()-start<15,'15CPU internal cap'
    def event(kind,**kw):
        cap()
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,**kw),allow_nan=False)+'\n')
        r['events']+=1;save()
    def call(kind,case,inputs,certify):
        row=dict(kind=kind,name=case['name'],input=inputs,attempted=False,status='started');r['rows'].append(row);save();event('started_case',kind_of_call=kind,name=case['name'])
        r['certificate_attempts']+=1;r[kind+'_calls']+=1;row['attempted']=True;save();cap();t=time.process_time()
        answer=certify(**inputs);row['certificate']=answer;row['call_cpu_seconds']=time.process_time()-t;r['certificate_returns']+=1;save()
        return row,answer
    def finish(row):row['status']='passed';save();event('completed_case',kind_of_call=row['kind'],name=row['name'])
    try:
        save();p=read(HERE/'protocol.json');gate=read(HERE/'gate.json')
        direct=[HERE/n for n in ('run.py','enclosure.py','range_primitive.py','launch.sh','PROTOCOL.md','README.md','protocol.json')]
        req=direct+[Path(p[k]) for k in ('selection','provenance','written_review','synthetic_review','synthetic_scope','prior_review','prior_scope','source_review')]
        assert gate['ready'] is True and all(str(x) in gate['pins'] for x in req)
        assert all(hexid(gate[k]) for k in ('source_review_commit','source_review_publication','execution_scope_publication'))
        verify(p['pins']);verify(gate['pins'])
        sr=read(p['source_review']);assert sr['complete'] is True and sr['pass_all'] is True and sr['execution_authorized'] is True
        assert all(str(x) in sr['pins'] for x in direct);verify(sr['pins'])
        wr=read(p['written_review']);assert wr['complete'] is True and wr['pass_all'] is True;verify(wr['pins'])
        syn=read(p['synthetic_review']);assert syn['complete'] is True and syn['pass_all'] is True and syn['total_rows']==24 and syn['exact_physical_reconstructions']==14
        assert syn['scope_sha256']==sha(p['synthetic_scope'])
        ps=read(p['prior_scope']);pr=read(p['prior_review']);prov=read(p['provenance'])
        assert pr['complete'] is True and pr['run_scope_sha256']==sha(p['prior_scope'])
        assert prov['prior_scope_sha256']==sha(p['prior_scope']) and prov['prior_review_sha256']==sha(p['prior_review'])
        assert len(prov['selected_payloads'])==6
        for d in prov['selected_payloads']:
            assert ps['pins'][d['path']]==d['sha256']==p['pins'][d['path']]
            assert Path(d['path']).stat().st_size==d['bytes'] and sha(d['path'])==d['sha256']
        cpu=resource.getrlimit(resource.RLIMIT_CPU);mem=resource.getrlimit(resource.RLIMIT_AS)
        assert cpu==(16,20) and mem==(2*1024**3,)*2
        r.update(limits=dict(cpu=cpu,address_space=mem),pins=gate['pins'],protocol_pins=p['pins'],prior_scope_sha256=sha(p['prior_scope']))
        selection=read(p['selection']);assert selection['model']=='H8T4seed0' and selection['physical_coordinate']=='x=v/2'
        assert selection['boundaries_v']==[-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8]
        assert len(selection['rows'])==4 and selection['proposed_new_calls']==dict(whole_box=4,restricted=4)
        for shift,c in enumerate(selection['rows']):
            ids=[0]*4
            for j,v in enumerate([8,7,6,5]):ids[(j+shift)%4]=v
            index=sum(v*12**(3-j) for j,v in enumerate(ids))
            assert c==dict(name='rotation_'+str(shift),bin_ids=ids,index=index,chunk=index//256,offset=index%256,leader=shift,second=(shift+1)%4,physical_gap='1/4',box_v=[[selection['boundaries_v'][j],selection['boundaries_v'][j+1]] for j in ids])
        event('started')
        r['model_manifest_decodes']+=1;save();manifest=read(selection['model_manifest']);assert manifest['complete'] is True
        matches=[m for m in manifest['models'] if m['name']==selection['model']];assert len(matches)==1
        model=matches[0]['model'];r['model_record']=matches[0];save()
        assert model['H']==8 and model['T']==4 and model['D']==2147483648
        from enclosure import certify
        r['selected_inputs']=[];r['setup_counts']=dict(archive_rows_decoded=0,selected_rows=0,physical_endpoint_conversions=0,old_margin_rational_values=0);save()
        keys={f'{i},{j}' for i in range(4) for j in range(i)}
        for case in selection['rows']:
            cap();path=Path(p['application_dir'])/('H8T4seed0_'+str(case['chunk']).zfill(3)+'.json.gz')
            assert str(path) in p['pins'] and ps['pins'][str(path)]==p['pins'][str(path)]
            r['selected_archive_decodes']+=1;save()
            chunk=json.loads(gzip.decompress(path.read_bytes()));r['setup_counts']['archive_rows_decoded']+=len(chunk['rows']);save();assert chunk['name']==selection['model'] and chunk['first']==256*case['chunk'] and chunk['last']==256*case['chunk']+255 and len(chunk['rows'])==256
            oldrow=chunk['rows'][case['offset']];r['setup_counts']['selected_rows']+=1;r['selected_inputs'].append(dict(selection=case,path=str(path),source_row=oldrow));save()
            assert oldrow['index']==case['index'] and oldrow['bin_ids']==case['bin_ids'] and oldrow['target_group']==[case['second']]
            old=oldrow['certificate'];den=old['margin_denominator'];assert type(den)is int and den>0 and set(old['margins'])==keys and len(old['statuses'])==4
            for pair in old['margins'].values():assert len(pair)==2 and all(type(v)is int for v in pair)
            expected={k:[F(v,den) for v in iv] for k,iv in old['margins'].items()};box=[[F(v,2) for v in iv] for iv in case['box_v']]
            r['setup_counts']['physical_endpoint_conversions']+=8;r['setup_counts']['old_margin_rational_values']+=12;save()
            inputs=dict(model=model,box=box);row,base=call('whole_box',case,inputs,certify)
            row['expected_old_margins']=expected;row['expected_old_statuses']=old['statuses'];save()
            assert base['status']=='feasible' and base['raw_margins']==base['final_margins']==expected
            assert base['raw_statuses']==base['statuses']==old['statuses'];finish(row)
            inputs=dict(model=model,box=box,domain=dict(leader=case['leader'],second=case['second'],gap=F(case['physical_gap'])),old_margins=expected)
            row,new=call('restricted',case,inputs,certify)
            assert new['status']=='feasible' and new['feasibility']['second_interval']==[F(1,2),F(1)]
            assert F(3,4) in [c['t'] for c in new['feasibility']['candidates']]
            assert new['old_margins']==expected
            for prior,status in zip(old['statuses'],new['statuses'],strict=True):assert prior=='unknown' or prior==status
            comparison={}
            for k in sorted(keys):
                before=expected[k];raw=new['raw_margins'][k];final=new['final_margins'][k]
                assert final==[max(before[0],raw[0]),min(before[1],raw[1])] and final[0]<=final[1]
                ow=before[1]-before[0];nw=raw[1]-raw[0];fw=final[1]-final[0]
                comparison[k]=dict(old=before,raw=raw,final=final,old_width=ow,raw_width=nw,final_width=fw,raw_width_change='narrower' if nw<ow else 'equal' if nw==ow else 'wider',raw_contained_in_old=before[0]<=raw[0]<=raw[1]<=before[1],old_contained_in_raw=raw[0]<=before[0]<=before[1]<=raw[1])
            target=case['second'];row['comparison']=comparison;row['target_statuses']=dict(old=old['statuses'][target],raw=new['raw_statuses'][target],fallback=new['statuses'][target]);save();finish(row)
        assert len(r['rows'])==8 and all(v['status']=='passed' for v in r['rows'])
        assert (r['certificate_attempts'],r['certificate_returns'],r['whole_box_calls'],r['restricted_calls'],r['model_manifest_decodes'],r['selected_archive_decodes'])==(8,8,4,4,1,4)
        assert r['setup_counts']==dict(archive_rows_decoded=1024,selected_rows=4,physical_endpoint_conversions=32,old_margin_rational_values=48)
        r['summary']=[dict(name=z['name'],target_statuses=z['target_statuses'],raw_width_changes={k:z['comparison'][k]['raw_width_change'] for k in sorted(keys)},accounting=z['certificate']['accounting']) for z in r['rows'] if z['kind']=='restricted'];save()
        event('completed');assert r['events']==18;r['complete']=r['pass_all']=True;save()
    except BaseException:
        r.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
