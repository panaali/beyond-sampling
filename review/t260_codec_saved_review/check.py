"""Once-only independent saved T260 audit. Never imports producer or predictors."""
from pathlib import Path
import ast,hashlib,json,math,os,re,resource,sys,time,traceback
from reader import Reader,from_bittext,program,array_from_npz,same,NAMES,SCHEMA,TYPES
P=Path(__file__).resolve().parent;OUT=P/'output'
def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def verify(pins):
    assert type(pins)is dict and pins
    for f,h in pins.items():assert re.fullmatch('[0-9a-f]{64}',h) and sha(f)==h,f

def dependencies(fs):
    # Independent union of explicit required step sets, no author function.
    steps=set()
    for f in fs:
        assert type(f)is int and 0<=f<=11
        if f==8:steps.add(9)
        if f==9:steps.update((7,8,9))
        if f>=10:steps.update((1,2,3,4,5,6,7,8,9))
    return sorted(steps)

def control_check(rows):
    assert len(rows)==32
    fixtures=[None,False,True,-100000,0,100000,'αβ',[],[1,2],(1,'x'),0.,-0.,1.25,-.004,ast.parse('x = a + 2\ny = {0, 2, 7}')]
    for i,x in enumerate(fixtures):
        r=rows[i];assert r['kind']=='roundtrip' and r['index']==i and r['same'] is True
        v,m=from_bittext(r['bits']);same(x,v);assert m==r['measure']
    pairs=[(ast.parse('x={0,2}'),ast.parse('x={0,3}')),(ast.parse('x=a+b'),ast.parse('x=a*b')),([1],[1,2])]
    for i,(a,b) in enumerate(pairs):
        r=rows[15+i];assert r['kind']=='mutation' and r['index']==i and r['a']!=r['b']
        same(a,from_bittext(r['a'])[0]);same(b,from_bittext(r['b'])[0])
        if i==2:assert len(r['a'])<len(r['b'])
    cases=[[],[0,7],[8],[9],[10],[11],[8,9],[0,8,10]]
    for i,fs in enumerate(cases):
        r=rows[18+i];same(fs,r['families']);assert r['kind']=='dependency' and r['actual']==r['expected']==dependencies(fs)
    for i,x in enumerate([[-1],[12],[True],[1.0]]):
        r=rows[26+i];same(x,r['input']);assert r['kind']=='guard' and r['error']=='family'
    r=rows[30];assert r['kind']=='route'
    same([[2.,6.],[3.,4.],[6.,5.]],r['actual']);same(r['actual'],r['expected'])
    assert rows[31]=={'kind':'guard','input':'trailing bit','error':'trailing bit rejected'}
    # Independent framing assertion from the stored contract, not author decode.
    q=Reader(b'\x00',4);assert q.value() is None and q.pos==q.end-1
    rejected=False
    try:q.finish()
    except AssertionError:rejected=True
    assert rejected

def main():
    assert not OUT.exists() or not any(OUT.iterdir()),'fresh output required'
    OUT.mkdir(exist_ok=True);start=time.process_time()
    result={'complete':False,'pass_all':False,'rows':[],'events':0,'arrays_checked':0,'author_calls':0,'predictor_calls':0}
    def save():
        result['cpu_seconds']=time.process_time()-start;result['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        f=OUT/'results.tmp';f.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');f.replace(OUT/'results.json')
    def event(name,**kw):
        save();assert time.process_time()-start<95,'internal cap'
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'event':name,'cpu_seconds':time.process_time()-start,**kw})+'\n')
        result['events']+=1;save()
    try:
        save();p=read(P/'protocol.json');g=read(P/'gate.json');completed=read(P/'completed_inputs.json')
        assert g['ready'] is True
        for key in ['review_commit','review_publication','execution_publication']:assert re.fullmatch('[0-9a-f]{8,40}',g[key])
        verify(p['pins']);verify(g['pins']);verify(completed['pins'])
        assert re.fullmatch('[0-9a-f]{8,40}',completed['producer_publication'])
        own=[str(P/x) for x in ['check.py','reader.py','protocol.json','PROTOCOL.md','launch.sh']]
        assert all(x in g['pins'] for x in own+[str(P/'completed_inputs.json'),p['source_review']])
        reviewer=read(p['source_review']);assert reviewer['complete'] is reviewer['pass_all'] is reviewer['execution_allowed'] is True
        verify(reviewer['pins']);assert all(x in reviewer['pins'] for x in own)
        assert resource.getrlimit(resource.RLIMIT_CPU)==(105,110) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
        assert sys.executable==p['python_invoked'] and str(Path(sys.executable).resolve())==p['python_resolved']
        assert sha(sys.executable)==p['python_sha256']
        root=Path(p['producer']);pp=read(root/'protocol.json');pg=read(root/'gate.json');manifest=read(root/'manifest.json')
        assert pg['ready'] is True
        for k in ['source_publication','review_publication','execution_publication']:assert re.fullmatch('[0-9a-f]{8,40}',pg[k])
        for obj in [pp,pg,manifest]:verify(obj['pins'])
        pr=read(pp['source_review']);assert pr['complete'] is pr['pass_all'] is pr['execution_authorized'] is True;verify(pr['pins'])
        assert all(str(root/f) in pr['pins'] for f in p['producer_sources'])
        assert all(str(root/f) in completed['pins'] for f in ['output/results.json','output/progress.jsonl','run1.log','gate.json'])
        old=read(root/'output/results.json');assert old['complete'] is old['pass_all'] is True;verify(old['pins']);assert old['pins']==pg['pins']
        prior=read(pp['pilot_review']);assert prior['complete'] is prior['pass_all'] is True and prior['archives']==168 and prior['models']==21
        contract=read(pp['contract_review']);assert contract['complete'] is contract['pass_all'] is True
        assert old['limits']=={'CPU':[105,110],'AS':[2**31,2**31]}
        assert old['runtime']['numpy']==pp['numpy_version'] and old['runtime']['python']==sys.version
        assert old['grammar_names']==NAMES and old['grammar_schema']==SCHEMA
        result.update(pins={**p['pins'],**completed['pins'],str(P/'gate.json'):sha(P/'gate.json')},limits={'CPU':[105,110],'AS':[2**31,2**31]},runtime={'invoked':sys.executable,'resolved':str(Path(sys.executable).resolve()),'python':sys.version})
        event('prerequisites')
        control_check(old['controls']);assert old['control_calls']=={'encode_attempts':22,'encode_returns':22,'decode_attempts':16,'decode_returns':15}
        assert old['control_counts']=={'roundtrips':15,'mutations':3,'dependencies':8,'family_guards':4,'routes':1,'frame_guards':1,'rows':32,'table_calls':1,'actual_predictor_calls':0}
        result['controls']=32;event('controls_checked')
        pilot=Path(pp['pilot'])/'output';archive=read(pilot/'results.json');assert archive['complete'] is archive['pass_all'] is True
        loaded=set();arrays={}
        def load(name,key):
            ident=(name,key)
            if ident in arrays:return arrays[ident]
            f=pilot/(name+'.npz');rec=archive['archives'][name]
            assert rec['path']==str(f) and manifest['pins'][str(f)]==rec['sha256']==sha(f)
            v,shape,desc=array_from_npz(f,key,rec);arrays[ident]=(v,shape,desc);loaded.add(name);result['arrays_checked']+=1;return arrays[ident]
        weights={}
        for k,shape in [('Wi',(16,)),('Wh',(16,16)),('Wo',(10,16))]:
            v,s,d=load('weights',k);assert s==shape and d=='<f8';weights[k]=v
        selections={}
        for key,maximum in [('choices',12),('no_hinge_choices',8)]:
            v,s,d=load('validation_selection',key);assert s==(16,) and d=='<i8' and all(0<=x<maximum for x in v);selections[key]=v
        same(selections['choices'],old['selection']['choices']);same(selections['no_hinge_choices'],old['selection']['no_hinge_choices']);assert old['inherited_metrics']==prior['metrics']
        # Independently bind all 21 inference records, including unselected overhead.
        models={}
        for name in ['bank_'+str(i) for i in range(1,10)]+['final_'+str(i) for i in range(12)]:
            ids,is_,it=load(name+'_model','ids');pool,ps,pt=load(name+'_model','pooled');coef,cs,ct=load(name+'_model','coef');width=4 if name.startswith('bank') else 16
            assert it=='<i8' and len(is_)==1 and len(ids)>0 and all(a<b for a,b in zip(ids,ids[1:]))
            assert pt==ct=='<f8' and len(ps)==2 and 1<=ps[0]<=64 and ps[1]==width and cs==(len(ids),*ps)
            models[name]=(ids,pool,coef)
        assert len(loaded)==old['loads']==23 and result['arrays_checked']==68
        event('archives_checked')
        names=[k+'_'+t for k in ['all12','no_hinge8','network','HA1'] for t in ['hidden','labels']]
        assert [r['name'] for r in old['rows']]==names
        totals=0;packed=0
        def file_bound(rec,suffix,name):
            f=root/'output'/(name+suffix);assert rec['path']==str(f) and f.stat().st_size==rec['bytes'] and sha(f)==rec['sha256']
            assert completed['pins'][str(f)]==rec['sha256'];return f
        for row in old['rows']:
            name=row['name'];kind,target=name.rsplit('_',1);event('artifact_started',name=name)
            assert row['status']=='complete' and row['exact_roundtrip'] is True
            bf=file_bound(row['binary'],'.bits',name);pf=file_bound(row['payload_file'],'.payload.json',name);af=file_bound(row['ast_file'],'.ast.json',name)
            bits=row['binary']['bits'];assert row['binary']['padding']=='leading zero pad' and bits==row['returned_bits']<=80_000_000
            reader=Reader(bf.read_bytes(),bits);begin=reader.pos;assert reader.take(3)==7 and reader.natural()==3
            tree,tm=reader.piece();payload,pm=reader.piece();reader.finish()
            framed={'bits':reader.pos-begin,'counts':reader.counts.copy()};framed['counts']['tuple']+=1
            assert framed==row['framed'] and tm==row['program'] and pm==row['payload'] and framed['bits']==6+tm['bits']+pm['bits']
            expected_tree=program(root,'tables' if kind in ('all12','no_hinge8') else kind)
            same(expected_tree,tree);assert ast.dump(tree,include_attributes=False)==read(af)
            same(payload,read(pf),json_containers=True)
            readout=None if target=='hidden' else weights['Wo']
            detail={}
            if kind in ('all12','no_hinge8'):
                choice=selections['choices' if kind=='all12' else 'no_hinge_choices'];families=sorted(set(choice));deps=dependencies(families)
                banks=[(s,*models['bank_'+str(s)]) for s in deps]
                finals=[(f,[j for j in range(16) if choice[j]==f],*models['final_'+str(f)]) for f in families]
                expected=(target,1,16,10,banks,finals,readout)
                full=sum((len(models['final_'+str(f)][0])+1)*len(models['final_'+str(f)][1])*16 for f in families)
                used=sum((len(models['final_'+str(f)][0])+1)*len(models['final_'+str(f)][1])*choice.count(f) for f in families)
                detail=dict(selected_families=families,bank_steps=deps,selected_final_full_float_count=full,selected_output_float_count=used,unused_column_float_count=full-used)
                assert all(row[k]==v for k,v in detail.items())
            elif kind=='network':expected=(target,1,weights['Wi'],weights['Wh'],readout)
            else:assert kind=='HA1';expected=(target,1,readout)
            same(expected,payload)
            totals+=bits;packed+=bf.stat().st_size;assert totals<=240_000_000
            result['rows'].append({'name':name,'bits':bits,'program':tm,'payload':pm,'framed':framed,'roundtrip_and_archive_binding':True,**detail});save();event('artifact_checked',name=name)
            del payload,expected,tree,reader
        assert totals==old['total_bits'] and packed==old['packed_bytes']
        assert all(old[k]==8 for k in ['encode_attempts','encode_returns','decode_attempts','decode_returns'])
        events=[read_line(x) for x in (root/'output/progress.jsonl').read_text().splitlines()]
        expected_events=['started','controls_completed','inputs_bound']+sum((['artifact_started','artifact_completed'] for _ in names),[])+['completed']
        assert len(events)==old['events']==20 and [e['event'] for e in events]==expected_events
        times=[e['cpu_seconds'] for e in events];assert all(type(t) in (int,float) and math.isfinite(t) and t>=0 for t in times) and times==sorted(times) and times[-1]<=old['cpu_seconds']<95
        for i,n in enumerate(names):assert events[3+2*i]['name']==events[4+2*i]['name']==n
        assert old['rss_kib']<2097152
        result.update(total_bits=totals,packed_bytes=packed,artifacts=8,producer_events=20,producer_cpu=old['cpu_seconds'],producer_rss_kib=old['rss_kib'],qualification='Exact declared byte grammar/payload consistency only; no predictor, fit, score, optimality or population-rate claim.')
        event('complete');assert result['events']==20
        result['complete']=result['pass_all']=True;save()
    except BaseException:
        result.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise

def read_line(s):return json.loads(s,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
if __name__=='__main__':main()
