"""Saved-only fixed eight-certificate audit; no author, model or CDF module imports."""
from pathlib import Path
from fractions import Fraction as F
import gzip,hashlib,json,resource,time,traceback
from math import isfinite
from reconstruction import reconstruct,rv,rt,eq,classify
P=Path(__file__).resolve().parent;START=time.process_time()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pins(x):
    assert isinstance(x,dict) and x
    for p,h in x.items():assert sha(p)==h,p
def cap():assert time.process_time()-START<19

def save(r):
    r['review_cpu']=time.process_time()-START;r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t=P/'results.tmp';t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(P/'results.json')

def selections():
    boundaries=[-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8];base=[8,7,6,5];out=[]
    for shift in range(4):
        ids=base[-shift:]+base[:-shift] if shift else base[:]
        index=0
        for x in ids:index=12*index+x
        out.append(dict(name='rotation_'+str(shift),bin_ids=ids,index=index,chunk=index//256,offset=index%256,leader=shift,second=(shift+1)%4,physical_gap='1/4',box_v=[[boundaries[i],boundaries[i+1]] for i in ids]))
    return out

def run(r):
    s=read(P/'scope.json');g=read(P/'gate.json');d=read(P/'completed_inputs.json');review=read(g['source_review'])
    assert g['ready'] is True and g['review_publication'] and g['execution_publication'] and d['raw_commit'] and d['publication']
    assert review['complete'] is True and review['pass_all'] is True and review['execution_authorized'] is True
    required={str(P/n) for n in ('check.py','reconstruction.py','vertex.py','scope.json','README.md')}
    assert required<=set(review['pins']) and required|{g['source_review'],str(P/'completed_inputs.json')}<=set(g['pins'])
    for z in (s['pins'],g['pins'],d['pins'],review['pins']):pins(z)
    a=Path(s['author']);ap=read(a/'protocol.json');ag=read(a/'gate.json')
    need={str(a/n) for n in ('run.py','enclosure.py','range_primitive.py','launch.sh','PROTOCOL.md','README.md','protocol.json','gate.json','output/results.json','output/progress.jsonl','run1.log')}|{ap[k] for k in ('selection','provenance','written_review','synthetic_review','synthetic_scope','prior_review','prior_scope','source_review')}
    assert need<=set(d['pins']) and ag['ready'] is True
    ar=read(ap['source_review']);wr=read(ap['written_review'])
    assert ar['complete'] is True and ar['pass_all'] is True and ar['execution_authorized'] is True and wr['complete'] is True and wr['pass_all'] is True
    for z in (ap['pins'],ag['pins'],ar['pins'],wr['pins']):pins(z)
    syn=read(ap['synthetic_review']);assert syn['complete'] is True and syn['pass_all'] is True and syn['total_rows']==24 and syn['exact_physical_reconstructions']==14 and syn['scope_sha256']==sha(ap['synthetic_scope'])
    ps=read(ap['prior_scope']);pr=read(ap['prior_review']);prov=read(ap['provenance'])
    assert pr['complete'] is True and pr['run_scope_sha256']==sha(ap['prior_scope'])
    assert prov['prior_scope_sha256']==sha(ap['prior_scope']) and prov['prior_review_sha256']==sha(ap['prior_review']) and len(prov['selected_payloads'])==6
    for z in prov['selected_payloads']:
        assert ps['pins'][z['path']]==ap['pins'][z['path']]==z['sha256']==sha(z['path']) and Path(z['path']).stat().st_size==z['bytes']
    assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,)*2
    selection=read(ap['selection']);cases=selections();assert selection['rows']==cases and selection['model']=='H8T4seed0' and selection['physical_coordinate']=='x=v/2'
    assert selection['boundaries_v']==[-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8] and selection['proposed_new_calls']==dict(whole_box=4,restricted=4)
    assert selection['model_manifest'] in {z['path'] for z in prov['selected_payloads']}
    # Scientific payload decode begins only after all gates, source and selected byte bindings pass.
    manifest=read(selection['model_manifest']);assert manifest['complete'] is True
    models=[m for m in manifest['models'] if m['name']=='H8T4seed0'];assert len(models)==1
    model=models[0]['model'];assert (model['H'],model['T'],model['D'])==(8,4,2147483648)
    assert type(model['H']) is int and type(model['T']) is int and type(model['D']) is int
    assert len(model['wi'])==8 and len(model['wh'])==8 and all(len(x)==8 for x in model['wh']) and len(model['wo'])==4 and all(len(x)==8 for x in model['wo'])
    assert all(type(v) is int for v in model['wi']) and all(type(v) is int for name in ('wh','wo') for row in model[name] for v in row)
    out=read(a/'output/results.json');assert out['complete'] is True and out['pass_all'] is True and out['model_record']==models[0]
    assert out['pins']==ag['pins'] and out['protocol_pins']==ap['pins'] and out['prior_scope_sha256']==sha(ap['prior_scope'])
    assert out['limits']==dict(cpu=[16,20],address_space=[2**31]*2)
    assert isfinite(out['cpu_seconds']) and out['cpu_seconds']>=0 and out['maxrss_kib']>0
    assert tuple(out[k] for k in ('certificate_attempts','certificate_returns','whole_box_calls','restricted_calls','model_manifest_decodes','selected_archive_decodes'))==(8,8,4,4,1,4)
    assert all(out[k]==0 for k in ('old_candidate_calls','checkpoint_loads','direct_point_forwards','cdf_calls','rng_calls','mass_calls'))
    assert out['setup_counts']==dict(archive_rows_decoded=1024,selected_rows=4,physical_endpoint_conversions=32,old_margin_rational_values=48)
    assert len(out['rows'])==8 and len(out['selected_inputs'])==4 and len(out['summary'])==4
    events=[dict(kind='started')];r['rows']=[];r['review_model_manifest_decodes']=1;r['review_archive_decodes']=0
    keys={f'{i},{j}' for i in range(4) for j in range(i)}
    for k,case in enumerate(cases):
        cap();path=Path(ap['application_dir'])/('H8T4seed0_'+str(case['chunk']).zfill(3)+'.json.gz');assert str(path) in ap['pins']
        chunk=json.loads(gzip.decompress(path.read_bytes()));r['review_archive_decodes']+=1
        assert chunk['name']=='H8T4seed0' and chunk['first']==256*case['chunk'] and chunk['last']==256*case['chunk']+255 and len(chunk['rows'])==256
        original=chunk['rows'][case['offset']];assert original['index']==case['index'] and original['bin_ids']==case['bin_ids'] and original['target_group']==[case['second']]
        assert out['selected_inputs'][k]==dict(selection=case,path=str(path),source_row=original)
        cert=original['certificate'];den=cert['margin_denominator'];assert type(den) is int and den>0 and set(cert['margins'])==keys and len(cert['statuses'])==4
        assert all(len(iv)==2 and all(type(x) is int for x in iv) for iv in cert['margins'].values())
        prior={key:[F(x,den) for x in iv] for key,iv in cert['margins'].items()};box=[[F(x,2) for x in iv] for iv in case['box_v']]
        statuses=[classify(prior,j,4) for j in range(4)];assert statuses==cert['statuses']
        whole,part=out['rows'][2*k:2*k+2]
        for kind,row in (('whole_box',whole),('restricted',part)):
            assert row['kind']==kind and row['name']==case['name'] and row['status']=='passed' and row['attempted'] is True
            assert row['input']['model']==model and isfinite(row['call_cpu_seconds']) and row['call_cpu_seconds']>=0;eq(row['input']['box'],box)
            events += [dict(kind='started_case',kind_of_call=kind,name=case['name']),dict(kind='completed_case',kind_of_call=kind,name=case['name'])]
        assert set(whole['input'])=={'model','box'}
        dg,raw=reconstruct(model,box,whole['certificate']);assert raw==prior
        eq(whole['expected_old_margins'],prior);assert whole['expected_old_statuses']==statuses and whole['certificate']['statuses']==statuses
        r['rows'].append(dict(name=case['name'],kind='whole_box',**dg));save(r)
        domain=dict(leader=case['leader'],second=case['second'],gap=F(1,4));assert set(part['input'])=={'model','box','domain','old_margins'}
        eq(part['input']['domain'],domain);eq(part['input']['old_margins'],prior)
        dg,raw=reconstruct(model,box,part['certificate'],domain,prior);assert not dg['empty']
        eq(part['certificate']['feasibility']['second_interval'],[F(1,2),F(1)]);assert F(3,4) in [F(v['t']) for v in part['certificate']['feasibility']['candidates']]
        changes={};assert set(part['comparison'])==keys
        for key in sorted(keys):
            before=prior[key];new=raw[key];final=[max(before[0],new[0]),min(before[1],new[1])];assert final[0]<=final[1]
            widths=[v[1]-v[0] for v in (before,new,final)];change='narrower' if widths[1]<widths[0] else 'equal' if widths[1]==widths[0] else 'wider';changes[key]=change
            c=part['comparison'][key]
            for field,value in dict(old=before,raw=new,final=final,old_width=widths[0],raw_width=widths[1],final_width=widths[2]).items():eq(c[field],value)
            assert c['raw_width_change']==change and c['raw_contained_in_old']==(before[0]<=new[0]<=new[1]<=before[1]) and c['old_contained_in_raw']==(new[0]<=before[0]<=before[1]<=new[1])
        target=case['second'];expected=dict(old=statuses[target],raw=part['certificate']['raw_statuses'][target],fallback=part['certificate']['statuses'][target]);assert part['target_statuses']==expected
        for b,n in zip(statuses,part['certificate']['statuses'],strict=True):assert b=='unknown' or b==n
        assert out['summary'][k]==dict(name=case['name'],target_statuses=expected,raw_width_changes=changes,accounting=part['certificate']['accounting'])
        r['rows'].append(dict(name=case['name'],kind='restricted',target_statuses=expected,raw_width_changes=changes,**dg));save(r)
    events.append(dict(kind='completed'));assert len(events)==out['events']==18 and [json.loads(v) for v in (a/'output/progress.jsonl').read_text().splitlines()]==events
    cap();r.update(complete=True,pass_all=True,total_rows=8,events=18,exact_certificate_reconstructions=8,trace_rows=sum(v['trace_rows'] for v in r['rows']),readout_contrasts=sum(v['contrasts'] for v in r['rows']),independent_vertex_systems=sum(v['systems'] for v in r['rows']),author_cpu=out['cpu_seconds'],input_pins=d['pins'],scope_sha256=sha(P/'scope.json'),author_calls=0,point_forward_cdf_rng_mass_calls=0,limitations='Exact saved represented-enclosure reconstruction on four fixed H8T4 domains; no point-forward or old candidate replay, no original weight deserialization or mass integration. Selected count formulas verified; internal peak-bit observations not replayed. No full cost, sampling, rung, overall accuracy or generality claim.')
if __name__=='__main__':
    assert not(P/'results.json').exists() and not(P/'results.tmp').exists()
    r=dict(complete=False,pass_all=False,rows=[]);save(r)
    try:run(r)
    except BaseException as e:r.update(complete=False,pass_all=False,error=repr(e),traceback=traceback.format_exc());raise
    finally:save(r)
