"""One fixed T260 codec/control and retained-artifact application, no scoring."""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,re,resource,sys,time,traceback
P=Path(__file__).resolve().parent;OUT=P/'output'
FILES=['run.py','representation.py','inference.py','network.py','ha_wrapper.py','controls.py','launch.sh','PLAN.md','PROTOCOL.md','protocol.json','manifest.json','frozen/codec.py','frozen/features.py','frozen/labels.py']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(pins):
    assert isinstance(pins,dict) and pins
    for f,h in pins.items():assert sha(f)==h,f

def main():
    assert not OUT.exists() or not any(OUT.iterdir());OUT.mkdir(exist_ok=True);start=time.process_time()
    r={'complete':False,'pass_all':False,'rows':[],'controls':[],'events':0,'loads':0,'encode_attempts':0,'encode_returns':0,'decode_attempts':0,'decode_returns':0,'packed_bytes':0,'total_bits':0}
    def save():
        r['cpu_seconds']=time.process_time()-start;r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t=OUT/'results.tmp';t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(OUT/'results.json')
    def cap():assert time.process_time()-start<95,'internal CPU cap'
    def event(kind,**kw):
        cap()
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'event':kind,'cpu_seconds':time.process_time()-start,**kw})+'\n')
        r['events']+=1;save()
    def jsonfile(name,obj):
        f=OUT/name;assert not f.exists();b=(json.dumps(obj,separators=(',',':'),allow_nan=False)+'\n').encode();assert len(b)<=128*2**20
        with f.open('xb') as s:s.write(b)
        return {'path':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    try:
        save();p=read(P/'protocol.json');g=read(P/'gate.json');m=read(P/'manifest.json')
        assert g['ready'] is True
        mandatory=[str(P/f) for f in FILES]+list(m['pins'])+[p['source_review']]
        assert all(f in g['pins'] for f in mandatory)
        for k in ['source_publication','review_publication','execution_publication']:assert re.fullmatch('[0-9a-f]{8,40}',g[k])
        verify(g['pins']);verify(m['pins']);verify(p['pins'])
        rv=read(p['source_review']);assert rv['complete'] and rv['pass_all'] and rv['execution_authorized'];verify(rv['pins']);assert all(str(P/f) in rv['pins'] for f in FILES)
        old=read(p['pilot_review']);assert old['complete'] and old['pass_all'] and old['archives']==168 and old['models']==21
        prior=read(p['contract_review']);assert prior['complete'] and prior['pass_all']
        assert resource.getrlimit(resource.RLIMIT_CPU)==(105,110) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
        assert str(Path(sys.executable).resolve())==p['python_resolved']
        import numpy as np
        assert np.__version__==p['numpy_version']
        import representation as rep,controls
        spec=importlib.util.spec_from_file_location('t25_frozen_codec',P/'frozen/codec.py');codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
        r.update(pins=g['pins'],limits={'CPU':[105,110],'AS':[2**31,2**31]},runtime={'python':sys.version,'numpy':np.__version__},grammar_schema=codec.SCHEMA,grammar_names=codec.NAMES)
        event('started')
        def emit(row):r['controls'].append(row);save();cap()
        r['control_calls']={'encode_attempts':0,'encode_returns':0,'decode_attempts':0,'decode_returns':0}
        class ControlCodec:
            @staticmethod
            def encode(x):
                r['control_calls']['encode_attempts']+=1;save();cap();z=codec.encode(x);r['control_calls']['encode_returns']+=1;save();return z
            @staticmethod
            def decode(x):
                r['control_calls']['decode_attempts']+=1;save();cap();z=codec.decode(x);r['control_calls']['decode_returns']+=1;save();return z
        r['control_counts']=controls.run(ControlCodec,rep,emit);save();assert r['control_calls']=={'encode_attempts':22,'encode_returns':22,'decode_attempts':16,'decode_returns':15};assert len(r['controls'])==32;event('controls_completed')
        pilot=Path(p['pilot']);source=read(pilot/'output/results.json');assert source['complete'] and source['pass_all']
        def load(name,keys):
            cap();f=pilot/'output'/f'{name}.npz';rec=source['archives'][name];assert str(f)==rec['path'] and sha(f)==rec['sha256'] and str(f)in m['pins']
            r['loads']+=1;save()
            with np.load(f,allow_pickle=False) as z:
                assert set(z.files)==set(rec['arrays']);a={k:z[k].copy() for k in keys}
            for k,v in a.items():assert list(v.shape)==rec['arrays'][k]['shape'] and str(v.dtype)==rec['arrays'][k]['dtype'] and hashlib.sha256(v.tobytes()).hexdigest()==rec['arrays'][k]['sha256']
            return a
        weights=load('weights',['Wi','Wh','Wo']);sel=load('validation_selection',['choices','no_hinge_choices'])
        assert weights['Wi'].shape==(16,) and weights['Wh'].shape==(16,16) and weights['Wo'].shape==(10,16)
        assert all(a.dtype==np.float64 and np.isfinite(a).all() for a in weights.values())
        models={}
        for name in ['bank_'+str(s) for s in range(1,10)]+['final_'+str(f) for f in range(12)]:
            a=load(name+'_model',['ids','pooled','coef']);ids=a['ids'];pool=a['pooled'];co=a['coef'];q=4 if name.startswith('bank') else 16
            assert ids.dtype==np.int64 and ids.ndim==1 and len(ids)>0 and np.all(ids[1:]>ids[:-1])
            assert pool.dtype==co.dtype==np.float64 and pool.ndim==2 and pool.shape[1]==q and 1<=pool.shape[0]<=64 and co.shape==(len(ids),*pool.shape) and np.isfinite(pool).all() and np.isfinite(co).all()
            models[name]=(ids.tolist(),pool.tolist(),co.tolist())
        assert r['loads']==23
        r['selection']={k:a.tolist() for k,a in sel.items()};r['inherited_metrics']=old['metrics'];save();event('inputs_bound')
        # Fixed order is independent of all output bit totals or evaluation scores.
        for kind in ['all12','no_hinge8','network','HA1']:
            for target in ['hidden','labels']:
                name=kind+'_'+target;entry={'name':name,'status':'started'};r['rows'].append(entry);event('artifact_started',name=name)
                readout=None if target=='hidden' else weights['Wo'].tolist()
                if kind in ('all12','no_hinge8'):
                    choice=sel['choices' if kind=='all12' else 'no_hinge_choices'];assert choice.shape==(16,) and choice.dtype==np.int64 and np.all((choice>=0)&(choice<(12 if kind=='all12' else 8)))
                    fam=sorted(set(map(int,choice)));deps=rep.dependencies(fam)
                    banks=[(s,*models['bank_'+str(s)]) for s in deps]
                    finals=[(f,[j for j in range(16) if choice[j]==f],*models['final_'+str(f)]) for f in fam]
                    payload=(target,1,16,10,banks,finals,readout);tree=rep.program('tables',codec.Normalize)
                    full=sum((len(models['final_'+str(f)][0])+1)*len(models['final_'+str(f)][1])*16 for f in fam)
                    used=sum((len(models['final_'+str(f)][0])+1)*len(models['final_'+str(f)][1])*int(np.count_nonzero(choice==f)) for f in fam)
                    entry.update(selected_families=fam,bank_steps=deps,selected_final_full_float_count=full,selected_output_float_count=used,unused_column_float_count=full-used)
                elif kind=='network':payload=(target,1,weights['Wi'].tolist(),weights['Wh'].tolist(),readout);tree=rep.program(kind,codec.Normalize)
                else:payload=(target,1,readout);tree=rep.program(kind,codec.Normalize)
                whole=(tree,payload);entry['program']=rep.measure(tree);entry['payload']=rep.measure(payload);entry['framed']=rep.measure(whole);save()
                assert entry['framed']['bits']==6+entry['program']['bits']+entry['payload']['bits']
                assert entry['framed']['bits']<=80_000_000 and r['total_bits']+entry['framed']['bits']<=240_000_000,'fixed encoded work cap'
                entry['payload_file']=jsonfile(name+'.payload.json',payload)
                entry['ast_file']=jsonfile(name+'.ast.json',ast.dump(tree,include_attributes=False))
                r['encode_attempts']+=1;save();cap();bits=codec.encode(whole);r['encode_returns']+=1;entry['returned_bits']=len(bits);save()
                packed=int(bits,2).to_bytes((len(bits)+7)//8,'big');f=OUT/(name+'.bits');assert not f.exists()
                with f.open('xb') as s:s.write(packed)
                entry['binary']={'path':str(f),'bytes':len(packed),'sha256':hashlib.sha256(packed).hexdigest(),'bits':len(bits),'padding':'leading zero pad'};r['packed_bytes']+=len(packed);r['total_bits']+=len(bits);save()
                assert len(bits)==entry['framed']['bits'];assert f'{int.from_bytes(packed,"big"):0{len(bits)}b}'==bits
                r['decode_attempts']+=1;save();cap();decoded=codec.decode(bits);r['decode_returns']+=1;entry['exact_roundtrip']=rep.same(whole,decoded);save();assert entry['exact_roundtrip']
                entry['status']='complete';event('artifact_completed',name=name)
                del whole,payload,tree,decoded,bits,packed
        assert len(r['rows'])==r['encode_attempts']==r['encode_returns']==r['decode_attempts']==r['decode_returns']==8
        r['qualification']='Declared conditional code upper bounds on fixed fresh pilot; no scoring, fits, RNN, RNG, optimality, population rate or historical recovery.'
        event('completed');assert r['events']==20
        r['complete']=r['pass_all']=True;save()
    except BaseException:r.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
