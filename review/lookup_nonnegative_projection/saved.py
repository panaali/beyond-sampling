"""Fixed reused T201 arrays, no new input, fit or recurrent network call."""
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'review/mechanism_residual_margin'
OUT=HERE/'saved'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())

def main():
    assert not OUT.exists(), 'immutable one-attempt output'
    assert 0<resource.getrlimit(resource.RLIMIT_CPU)[0]<=20
    assert 0<resource.getrlimit(resource.RLIMIT_AS)[0]<=2147483648
    OUT.mkdir();start=time.process_time()
    r=dict(complete=False,chunks=[],panels={},archive_reads=0,projection_calls=0,
           named_expression_counts={},framework_forward_calls=0,fit_calls=0,random_calls=0,
           interpretation='Post-original-result descriptive follow-up on reused panels; no fresh validation or generalization CI.')
    def save():
        r.update(cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        p=OUT/'results.json.tmp'
        p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
        with p.open('rb') as f: os.fsync(f.fileno())
        os.replace(p,OUT/'results.json')
    def event(phase,**kw):
        with (OUT/'progress.jsonl').open('a') as f:
            f.write(json.dumps(dict(phase=phase,**kw),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    def cap(): assert time.process_time()-start<17, '17CPU soft cap; retain partial result, no automatic retry'
    save();event('attempt',stage='preflight')
    try:
        p=read(HERE/'saved_protocol.json');g=read(HERE/'saved_gate.json')
        assert g['clearance'] is True and g['protocol_sha256']==sha(HERE/'saved_protocol.json')
        for path,digest in {**p['pins'],**g['pins']}.items(): assert sha(path)==digest,path
        reviewed=g['controls_saved_review']
        assert len(reviewed['commit'])>=8 and all(c in '0123456789abcdef' for c in reviewed['commit'])
        assert g['pins'][reviewed['path']]==reviewed['sha256']
        checked=read(reviewed['path']);assert checked['complete'] and checked['pass_all']
        ctrl=read(HERE/'controls/results.json');assert ctrl['status']=='complete' and ctrl['pass_all']
        assert p['panels']==['train','holdout1','holdout2'] and p['chunks_per_panel']==64
        r.update(protocol_sha256=sha(HERE/'saved_protocol.json'),gate=g)
        import numpy as np
        from primitive import project
        r['runtime']=dict(executable=sys.executable,prefix=sys.prefix,numpy_version=np.__version__,numpy_file=np.__file__);save()
        assert r['runtime']==p['runtime']
        manifest=read(PRIOR/'manifest.json')
        for path,digest in manifest['pins'].items(): assert sha(ROOT/path)==digest,path
        old=read(PRIOR/'saved/results.json');assert old['complete'] and len(old['chunks'])==192
        expected=[(panel,j) for panel in p['panels'] for j in range(64)]
        assert [(x['panel'],x['index']) for x in old['chunks']]==expected
        assert all(x['complete'] for x in old['chunks'])
        def load(path,digest,keys):
            assert sha(path)==digest,path
            r['archive_reads']+=1
            with np.load(path,allow_pickle=False) as a: return {k:a[k].copy() for k in keys}
        wp='review/mechanism_claims_review/rebuild/pilot/weights.npz'
        w=load(ROOT/wp,manifest['pins'][wp],['Wo'])['Wo']
        assert w.shape==(10,16) and w.dtype==np.float64 and np.isfinite(w).all()
        event('result',stage='preflight');save()
        prior={(x['panel'],x['index']):x for x in old['chunks']}
        for panel in p['panels']:
            sums=dict(rows=0,negative_rows=0,negative_coordinates=0,max_negative_magnitude=0.,
                      negative_l1=0.,negative_l2_squared=0.,sse_before=0.,sse_after=0.,
                      sse_increase_rows=0,max_sse_increase=0.,projected_tie_rows=0,
                      network_task_correct=0,baseline_task_correct=0,projected_task_correct=0,
                      baseline_network_agree=0,projected_network_agree=0,changed_label_rows=0)
            per_count=np.zeros(16,dtype=np.int64);per_l1=np.zeros(16);per_l2=np.zeros(16)
            task_pairs=np.zeros((2,2),dtype=np.int64);agree_pairs=np.zeros((2,2),dtype=np.int64)
            label_pairs=np.zeros((10,10),dtype=np.int64)
            contexts=np.zeros((3,3,10,2,2,2,2),dtype=np.int64)
            for index in range(64):
                cap();prev=prior[(panel,index)]
                rec=dict(panel=panel,index=index,complete=False,source=prev['source'],prior_output=prev['output'])
                r['chunks'].append(rec);event('attempt',panel=panel,index=index);save()
                a=load(ROOT/prev['output']['path'],prev['output']['sha256'],
                       ['lookup_hidden','lookup_logits','network','lookup','truth','case','positive_bin','network_global_rank'])
                raw=load(ROOT/prev['source']['outputs'],prev['source']['outputs_sha256'],['hidden','logits'])
                h=raw['hidden'];z=a['lookup_hidden'];original_logits=raw['logits']
                assert h.shape==z.shape==(1024,16) and original_logits.shape==a['lookup_logits'].shape==(1024,10)
                assert all(x.dtype==np.float64 and np.isfinite(x).all() for x in (h,z,original_logits,a['lookup_logits']))
                for key in ['network','lookup','truth','case','positive_bin','network_global_rank']:
                    assert a[key].shape==(1024,) and np.issubdtype(a[key].dtype,np.integer)
                assert np.array_equal(original_logits.argmax(axis=1),a['network'])
                assert np.array_equal(a['lookup_logits'].argmax(axis=1),a['lookup'])
                assert np.all((0<=a['truth'])&(a['truth']<10))
                assert np.all((0<=a['case'])&(a['case']<3)) and np.all((0<=a['positive_bin'])&(a['positive_bin']<3))
                assert np.all((1<=a['network_global_rank'])&(a['network_global_rank']<=10))
                r['projection_calls']+=1;save()
                result=project(h,z,w)
                arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
                dest=OUT/f'{panel}_{index:03d}.npz'
                # Save completed numerical outputs before any downstream metric/parity assertions.
                with dest.open('xb') as f: np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
                rec['output']=dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),arrays={k:list(v.shape) for k,v in arrays.items()})
                rec['counts']=result['named_expression_counts'];save()
                pred=result['projected_logits'].argmax(axis=1)
                negative=z<0;mag=np.where(negative,-z,0.)
                assert np.isfinite(mag).all()
                nc=(a['lookup']==a['network']).astype(np.int64);pc=(pred==a['network']).astype(np.int64)
                nt=(a['lookup']==a['truth']).astype(np.int64);pt=(pred==a['truth']).astype(np.int64)
                np.add.at(task_pairs,(nt,pt),1);np.add.at(agree_pairs,(nc,pc),1)
                np.add.at(label_pairs,(a['lookup'],pred),1)
                np.add.at(contexts,(a['positive_bin'],a['case'],a['network_global_rank']-1,nc,pc,nt,pt),1)
                per_count+=negative.sum(axis=0);per_l1+=mag.sum(axis=0);per_l2+=(mag*mag).sum(axis=0)
                delta=result['sse_after']-result['sse_before']
                sums['rows']+=1024;sums['negative_rows']+=int(negative.any(axis=1).sum())
                sums['negative_coordinates']+=int(negative.sum());sums['max_negative_magnitude']=max(sums['max_negative_magnitude'],float(mag.max()))
                sums['negative_l1']+=float(mag.sum());sums['negative_l2_squared']+=float((mag*mag).sum())
                sums['sse_before']+=float(result['sse_before'].sum());sums['sse_after']+=float(result['sse_after'].sum())
                sums['sse_increase_rows']+=int((delta>0).sum());sums['max_sse_increase']=max(sums['max_sse_increase'],float(delta.max()))
                sums['projected_tie_rows']+=int(((result['projected_logits']==result['projected_logits'].max(axis=1)[:,None]).sum(axis=1)>1).sum())
                sums['network_task_correct']+=int((a['network']==a['truth']).sum())
                sums['baseline_task_correct']+=int(nt.sum());sums['projected_task_correct']+=int(pt.sum())
                sums['baseline_network_agree']+=int(nc.sum());sums['projected_network_agree']+=int(pc.sum())
                sums['changed_label_rows']+=int((pred!=a['lookup']).sum())
                for k,v in rec['counts'].items():r['named_expression_counts'][k]=r['named_expression_counts'].get(k,0)+v
                rec['complete']=True;event('result',panel=panel,index=index);save();cap()
            oldc=old['panels'][panel]['transition_counts']
            assert sums['network_task_correct']==oldc[0]+oldc[2]
            assert sums['baseline_task_correct']==oldc[0]+oldc[1]
            assert sums['baseline_network_agree']==oldc[0]+oldc[3]
            assert sums['rows']==65536 and int(task_pairs.sum())==int(agree_pairs.sum())==int(label_pairs.sum())==int(contexts.sum())==65536
            r['panels'][panel]=dict(totals=sums,negative_counts_per_neuron=per_count.tolist(),negative_l1_per_neuron=per_l1.tolist(),
                negative_l2_squared_per_neuron=per_l2.tolist(),task_pairs=task_pairs.tolist(),network_agreement_pairs=agree_pairs.tolist(),
                baseline_projected_label_pairs=label_pairs.tolist(),context_axes=['positive_bin','case','network_rank_minus1','baseline_agrees','projected_agrees','baseline_task_correct','projected_task_correct'],context_counts=contexts.tolist())
            event('panel_complete',panel=panel);save()
        assert len(r['chunks'])==192 and all(x['complete'] for x in r['chunks'])
        assert r['projection_calls']==192 and r['archive_reads']==385
        r['complete']=True;event('complete');save()
    except BaseException as e:
        r.update(failure=repr(e),traceback=traceback.format_exc());event('failure',error=repr(e));save();raise

if __name__=='__main__': main()
