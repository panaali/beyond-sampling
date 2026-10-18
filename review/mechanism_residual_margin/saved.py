"""Fixed saved T201 panels: fitted arithmetic only, no recurrence or fitting."""
from pathlib import Path
import hashlib,json,resource,sys,time,traceback
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
DATA=ROOT/'review/mechanism_claims_review/rebuild';OUT=H/'saved'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    assert not OUT.exists(),'immutable once-only output'
    OUT.mkdir();resource.setrlimit(resource.RLIMIT_CPU,(19,20));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    start=time.process_time();r=dict(complete=False,chunks=[],panels={},counts=dict(lookup_rows=0,readout_rows=0,diagnostic_rows=0,diagnostic_calls=0,archive_reads=0),named_diagnostic_arithmetic={},lookup_readout_expression_arithmetic=dict(mul=0,add=0))
    def save():
        r.update(cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        p=OUT/'results.tmp';p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');p.replace(OUT/'results.json')
    def emit(stage,**kw):
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,**kw),allow_nan=False)+'\n');f.flush()
    def cap():
        if time.process_time()-start>17:raise RuntimeError('17 CPU soft cap; preserve partial, no automatic retry')
    try:
        p=read(H/'saved_protocol.json');g=read(H/'saved_gate.json')
        assert g['passed'] and g['published_commit']!='pending' and g['controls_saved_review']!='pending'
        assert g['protocol_sha256']==sha(H/'saved_protocol.json')
        for path,digest in {**p['pins'],**g.get('additional_pins',{})}.items():assert sha(path)==digest,path
        assert read(H/'controls_v2_results.json')['complete']
        r.update(protocol_sha256=sha(H/'saved_protocol.json'),gate_sha256=sha(H/'saved_gate.json'))
        emit('started');save()
        import numpy as np
        from primitive import analyze,TRANSITIONS
        r['runtime']=dict(executable=sys.executable,prefix=sys.prefix,numpy=np.__version__,numpy_file=np.__file__)
        save();emit('runtime_bound',runtime=r['runtime'])
        assert sys.executable==p['invoked_python'] and sys.prefix==p['venv_prefix']
        assert np.__version__==p['numpy_version'] and np.__file__==p['numpy_file']
        manifest=read(H/'manifest.json')
        for path,digest in manifest['pins'].items():assert sha(ROOT/path)==digest,path
        old=read(DATA/'fullfit/results.json');assert old['complete']
        def load(path,expected):
            assert sha(path)==expected,path
            r['counts']['archive_reads']+=1
            with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
        def inherited(path):return load(ROOT/path,manifest['pins'][path])
        B=inherited('review/mechanism_claims_review/rebuild/fullfit/lookup_coefficients.npz')['coefficients']
        Wo=inherited('review/mechanism_claims_review/rebuild/pilot/weights.npz')['Wo']
        assert B.shape==(216,3,16) and Wo.shape==(10,16)
        assert np.isfinite(B).all() and np.isfinite(Wo).all()
        legal=np.array([(c*9+a)*9+b for c in range(3) for a in range(9) for b in range(9) if a!=b])
        prior={(c['panel'],c['index']):c for c in old['chunks']};assert len(prior)==191
        pp=read(DATA/'FULLFIT_PROTOCOL.json')['pins']
        for panel in ['train','holdout1','holdout2']:
            labels=inherited(f'review/mechanism_claims_review/rebuild/fullfit/{panel}_labels.npz')
            assert all(v.shape==(65536,) for v in labels.values())
            table=np.zeros((3,3,10,5),dtype=np.int64)
            signed=np.zeros((5,16));absolute=np.zeros((5,16))
            stats=dict(n=0,max_abs_closure=0.,closure_l1=0.,zero_term_sums=0,network_tie_rows=0,lookup_tie_rows=0,max_finite_cancellation_ratio=None)
            for index in range(64):
                cap();rec=dict(panel=panel,index=index,complete=False);r['chunks'].append(rec)
                emit('chunk_attempt',panel=panel,index=index);save()
                if panel=='train' and index==0:
                    ix=DATA/'pilot/inputs.npz';ox=DATA/'pilot/candidate.npz'
                    ih=next(v for k,v in pp.items() if k.endswith('/pilot/inputs.npz'))
                    oh=next(v for k,v in pp.items() if k.endswith('/pilot/candidate.npz'))
                else:
                    priorc=prior[(panel,index)];assert priorc['complete']
                    ix=DATA/'fullfit'/Path(priorc['inputs']['path']).name;ox=DATA/'fullfit'/Path(priorc['outputs']['path']).name
                    ih=priorc['inputs']['sha256'];oh=priorc['outputs']['sha256']
                rec['source']=dict(inputs=str(ix.relative_to(ROOT)),inputs_sha256=ih,outputs=str(ox.relative_to(ROOT)),outputs_sha256=oh)
                inp=load(ix,ih);out=load(ox,oh);x=inp['x'];hidden=out['hidden'];logits=out['logits']
                assert x.shape==(1024,10) and hidden.shape==(1024,16) and logits.shape==(1024,10)
                assert all(a.dtype==np.float64 and np.isfinite(a).all() for a in (x,hidden,logits))
                sl=slice(index*1024,(index+1)*1024);ys=labels['true_labels'][sl];key=labels['state_key'][sl]
                case=key//81;p1=(key//9)%9;p2=key%9;row=np.arange(1024)
                assert np.all(p1!=p2) and np.all((key>=0)&(key<243))
                slot=np.searchsorted(legal,key);assert np.array_equal(legal[slot],key)
                X=np.column_stack([x[row,p1],x[row,p2],x[:,9]])
                assert np.array_equal(case,labels['case'][sl])
                assert np.array_equal(ys,np.where(case==0,p1,np.where(case==1,9,p2)))
                positive=np.minimum((x>0).sum(axis=1),2)
                assert np.array_equal(positive,labels['positive_bin'][sl])
                r['counts']['lookup_rows']+=1024;save()
                hh=np.einsum('ni,nij->nj',X,B[slot])
                r['counts']['readout_rows']+=1024;save();ll=hh@Wo.T
                r['counts']['diagnostic_calls']+=1;save()
                a=analyze(hidden,hh,logits,ll,Wo,ys);r['counts']['diagnostic_rows']+=1024
                # Preserve computed output before parity assertions; mismatches stop here.
                arrays={k:v for k,v in a.items() if isinstance(v,np.ndarray)}
                arrays.update(lookup_hidden=hh,lookup_logits=ll,truth=ys,case=case,positive_bin=positive,state_key=key)
                rank=1+(x>x[row,a['network'],None]).sum(axis=1);arrays['network_global_rank']=rank
                dest=OUT/f'{panel}_{index:03d}.npz'
                with dest.open('xb') as f:np.savez_compressed(f,**arrays)
                rec['output']=dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),arrays={k:list(v.shape) for k,v in arrays.items()})
                rec['counts']=a['named_arithmetic'];save()
                assert np.array_equal(a['network'],labels['network'][sl])
                assert np.array_equal(a['lookup'],labels['lookup'][sl])
                assert np.all((rank>=1)&(rank<=10))
                np.add.at(table,(positive,case,rank-1,a['transition']),1)
                np.add.at(signed,a['transition'],a['terms']);np.add.at(absolute,a['transition'],np.abs(a['terms']))
                stats['n']+=1024
                stats['max_abs_closure']=max(stats['max_abs_closure'],float(np.max(np.abs(a['closure_residual']))))
                stats['closure_l1']+=float(np.abs(a['closure_residual']).sum())
                nz=a['term_sum']!=0;stats['zero_term_sums']+=int((~nz).sum())
                if nz.any():
                    ratio=a['term_abs_sum'][nz]/np.abs(a['term_sum'][nz]);assert np.isfinite(ratio).all()
                    stats['max_finite_cancellation_ratio']=max(stats['max_finite_cancellation_ratio'] or 0,float(ratio.max()))
                for prefix in ('network','lookup'):stats[prefix+'_tie_rows']+=int((a[prefix+'_max_ties']>1).sum())
                for k,v in a['named_arithmetic'].items():r['named_diagnostic_arithmetic'][k]=r['named_diagnostic_arithmetic'].get(k,0)+v
                r['lookup_readout_expression_arithmetic']['mul']+=1024*(3*16+10*16)
                r['lookup_readout_expression_arithmetic']['add']+=1024*(2*16+10*15)
                rec['complete']=True;rec['max_abs_closure']=float(np.max(np.abs(a['closure_residual'])))
                emit('chunk_completed',panel=panel,index=index);save();cap()
            counts=table.sum(axis=(0,1,2));assert int(counts.sum())==65536
            assert np.array_equal(table.sum(axis=3),np.array(old['panels'][panel]['network']['cross_tab']['table']))
            assert int(counts[0]+counts[2])==old['panels'][panel]['network']['task_correct']
            assert int(counts[0]+counts[1])==old['panels'][panel]['lookup']['task_correct']
            assert int(counts[0]+counts[3])==old['panels'][panel]['lookup']['network_agree']
            r['panels'][panel]=dict(stats=stats,transition_names=list(TRANSITIONS),transition_counts=counts.tolist(),context_axes=['positive_bin','case','network_global_rank_minus1','transition'],context_counts=table.tolist(),neuron_signed_sum=signed.tolist(),neuron_absolute_sum=absolute.tolist())
            emit('panel_completed',panel=panel);save()
        assert len(r['chunks'])==192 and all(c['complete'] for c in r['chunks'])
        assert r['counts']==dict(lookup_rows=196608,readout_rows=196608,diagnostic_rows=196608,diagnostic_calls=192,archive_reads=389)
        r.update(complete=True,scope=p['scope'],count_scope='Partial expression counts, not native/runtime costs; inherited fits/forwards and advice separate. No new recurrence/fit/RNG/reference.')
        save();emit('completed',results_sha256=sha(OUT/'results.json'))
    except BaseException:
        r['failure']=traceback.format_exc();save();emit('failure');raise

if __name__=='__main__':main()
