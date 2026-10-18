"""Independent saved projection/readout and descriptive-table audit.

No author imports, recurrent forwards, fitting, RNG or candidate calls. Separate
column-wise readout and fsum reductions check source arithmetic consistency
under standard binary64 rounding bounds; labels use the saved output bits.
"""
from pathlib import Path
from collections import Counter
import hashlib, json, math, time, traceback
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/workspace/work/recover-small_h2_56')
BASE = ROOT / 'review/lookup_nonnegative_projection'
PRIOR = ROOT / 'review/mechanism_residual_margin'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def load(p):
    with np.load(p, allow_pickle=False) as z: return {k:z[k] for k in z.files}
def bits(a,b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
def close(a,b,scale,n):
    a,b,scale = np.asarray(a),np.asarray(b),np.asarray(scale)
    gamma = n*2**-53/(1-n*2**-53)
    err = np.abs(a-b)
    assert np.all(err <= 4*gamma*scale+1e-300), (float(err.max()), n)
    return float(err.max())
def winner(l): return np.argsort(-l, axis=1, kind='stable')[:,0]

def main():
    start=time.process_time();dest=HERE/'t224_full_saved_review.json';assert not dest.exists()
    r=read(BASE/'saved/results.json');p=read(BASE/'saved_protocol.json');g=read(BASE/'saved_gate.json')
    assert sha(BASE/'saved/results.json')=='ed1e7d1266b1ce0c67ce426f3234036b6a02249d387cc84578dd9995452bbe0f'
    assert r['complete'] and 'failure' not in r and g['clearance'] is True
    assert r['gate']==g and r['protocol_sha256']==g['protocol_sha256']==sha(BASE/'saved_protocol.json')
    assert r['runtime']==p['runtime']
    for path,digest in {**p['pins'],**g['pins']}.items(): assert sha(path)==digest, path
    manifest=read(PRIOR/'manifest.json')
    for path,digest in manifest['pins'].items(): assert sha(ROOT/path)==digest,path
    old=read(PRIOR/'saved/results.json');assert old['complete']
    reviewed=read(HERE/'residual_full_saved_review.json');assert reviewed['complete'] and reviewed['pass_all']
    assert reviewed['pins']['saved/results.json']==sha(PRIOR/'saved/results.json')
    prior={(x['panel'],x['index']):x for x in old['chunks']}; assert len(prior)==192
    panels=['train','holdout1','holdout2'];expected=[(a,i) for a in panels for i in range(64)]
    assert [(x['panel'],x['index']) for x in r['chunks']]==expected
    events=[readline for readline in map(json.loads,(BASE/'saved/progress.jsonl').read_text().splitlines())]
    expected_events=[dict(phase='attempt',stage='preflight'),dict(phase='result',stage='preflight')]
    for panel in panels:
        for i in range(64): expected_events += [dict(phase='attempt',panel=panel,index=i),dict(phase='result',panel=panel,index=i)]
        expected_events.append(dict(phase='panel_complete',panel=panel))
    expected_events.append(dict(phase='complete'));assert events==expected_events
    weights=ROOT/'review/mechanism_claims_review/rebuild/pilot/weights.npz'
    assert sha(weights)==manifest['pins'][str(weights.relative_to(ROOT))]
    w=load(weights)['Wo'];assert w.shape==(10,16) and w.dtype==np.float64
    counts=Counter();report={};errors=dict(readout=0.,row_sse=0.,aggregate=0.)
    unchanged_total=0;readout_byte_chunks=0
    for panel in panels:
        group=[c for c in r['chunks'] if c['panel']==panel]
        task=np.zeros(4,dtype=np.int64);agree=np.zeros(4,dtype=np.int64);labels=np.zeros(100,dtype=np.int64);context=np.zeros(1440,dtype=np.int64)
        integers=Counter();float_parts={k:[] for k in ['negative_l1','negative_l2_squared','sse_before','sse_after']}
        per_negative=[[] for _ in range(16)];per_count=np.zeros(16,dtype=np.int64)
        maxmag=0.;maxincrease=0.
        for c in group:
            i=c['index'];prev=prior[(panel,i)];assert c['complete']
            assert c['source']==prev['source'] and c['prior_output']==prev['output']
            for item in [c['output'],c['prior_output']]: assert sha(ROOT/item['path'])==item['sha256']
            for role in ['inputs','outputs']: assert sha(ROOT/c['source'][role])==c['source'][role+'_sha256']
            a=load(ROOT/c['prior_output']['path']);raw=load(ROOT/c['source']['outputs']);v=load(ROOT/c['output']['path'])
            assert {k:list(x.shape) for k,x in v.items()}==c['output']['arrays']
            assert set(v)=={'projected','projected_logits','sse_before','sse_after'}
            h,z,l=raw['hidden'],a['lookup_hidden'],raw['logits'];q=v['projected'];pl=v['projected_logits']
            assert h.shape==z.shape==q.shape==(1024,16) and l.shape==pl.shape==(1024,10)
            assert all(x.dtype==np.float64 and np.isfinite(x).all() for x in [h,z,l,*v.values()]) and np.all(h>=0)
            # Element ownership, including canonical +0 in clipped/signed-zero entries.
            expected_q=z.copy();expected_q[expected_q<=0]=0.;bits(q,expected_q)
            # Different reduction tree from the candidate BLAS readout.
            terms=[q[:,j,None]*w[None,:,j] for j in range(16)]
            recon=terms[0].copy()
            for term in terms[1:]: recon=recon+term
            errors['readout']=max(errors['readout'],close(recon,pl,sum(np.abs(x) for x in terms),32))
            readout_byte_chunks+=int(recon.tobytes()==pl.tobytes())
            for name,zz in [('sse_before',z),('sse_after',q)]:
                squared=(zz-h)*(zz-h)
                independent=np.array([math.fsum(row) for row in squared])
                errors['row_sse']=max(errors['row_sse'],close(independent,v[name],independent,32))
                float_parts[name].extend(v[name].tolist())
            # The exact-real coordinate SSE drop is z^2-2zh>=0 when z<0,h>=0;
            # other coordinates are unchanged. No floating difference is truncated.
            negative=z<0;mag=np.where(negative,-z,0.);unclipped=~np.any(negative,axis=1)
            assert c['unchanged_rows']==int(unclipped.sum()) and c['unchanged_readout_equal'] and c['unchanged_label_changes']==0
            assert np.array_equal(pl[unclipped],a['lookup_logits'][unclipped]);unchanged_total+=int(unclipped.sum())
            pred=winner(pl);base=winner(a['lookup_logits']);network=winner(l);truth=a['truth']
            bits(base,a['lookup']);bits(network,a['network']);assert np.array_equal(pred[unclipped],base[unclipped])
            assert all(x.shape==(1024,) for x in [truth,a['case'],a['positive_bin'],a['network_global_rank']])
            assert np.all((truth>=0)&(truth<10))
            bt,pt,bn,pn=(base==truth),(pred==truth),(base==network),(pred==network)
            task+=np.bincount(2*bt.astype(int)+pt,minlength=4);agree+=np.bincount(2*bn.astype(int)+pn,minlength=4)
            labels+=np.bincount(10*base+pred,minlength=100)
            idx=((a['positive_bin']*3+a['case'])*10+a['network_global_rank']-1)
            for bit in [bn,pn,bt,pt]: idx=2*idx+bit
            context+=np.bincount(idx,minlength=1440)
            per_count+=np.count_nonzero(negative,axis=0)
            for j in range(16): per_negative[j].extend(mag[:,j].tolist())
            float_parts['negative_l1'].extend(mag.ravel().tolist());float_parts['negative_l2_squared'].extend((mag*mag).ravel().tolist())
            d=v['sse_after']-v['sse_before'];maxincrease=max(maxincrease,float(d.max()));maxmag=max(maxmag,float(mag.max()))
            integers.update(rows=1024,negative_rows=int(np.any(negative,axis=1).sum()),negative_coordinates=int(negative.sum()),sse_increase_rows=int((d>0).sum()),projected_tie_rows=int((np.sum(pl==pl[np.arange(1024),pred,None],axis=1)>1).sum()),network_task_correct=int((network==truth).sum()),baseline_task_correct=int(bt.sum()),projected_task_correct=int(pt.sum()),baseline_network_agree=int(bn.sum()),projected_network_agree=int(pn.sum()),changed_label_rows=int((base!=pred).sum()))
            expected_count=dict(comparisons=1024*16,subtractions=2*1024*16,multiplications=1024*(2*16+16*10),additions=1024*(2*15+10*15))
            assert c['counts']==expected_count;counts.update(expected_count)
        saved=r['panels'][panel];total=saved['totals']
        assert all(total[k]==v for k,v in integers.items()) and total['rows']==65536
        assert total['max_negative_magnitude']==maxmag and total['max_sse_increase']==maxincrease
        for k,parts in float_parts.items():
            fs=math.fsum(parts);errors['aggregate']=max(errors['aggregate'],close(fs,total[k],fs,len(parts)))
        assert per_count.tolist()==saved['negative_counts_per_neuron']
        for k,square in [('negative_l1_per_neuron',False),('negative_l2_squared_per_neuron',True)]:
            vals=np.array([math.fsum((x*x if square else x) for x in row) for row in per_negative]);errors['aggregate']=max(errors['aggregate'],close(vals,saved[k],vals,65536))
        assert task.reshape(2,2).tolist()==saved['task_pairs'] and agree.reshape(2,2).tolist()==saved['network_agreement_pairs']
        assert labels.reshape(10,10).tolist()==saved['baseline_projected_label_pairs']
        assert context.reshape(3,3,10,2,2,2,2).tolist()==saved['context_counts']
        assert saved['context_axes']==['positive_bin','case','network_rank_minus1','baseline_agrees','projected_agrees','baseline_task_correct','projected_task_correct']
        oldc=old['panels'][panel]['transition_counts']
        assert total['network_task_correct']==oldc[0]+oldc[2] and total['baseline_task_correct']==oldc[0]+oldc[1] and total['baseline_network_agree']==oldc[0]+oldc[3]
        assert task.sum()==agree.sum()==labels.sum()==context.sum()==65536
        report[panel]=dict(totals=total,task_pairs=saved['task_pairs'],network_agreement_pairs=saved['network_agreement_pairs'],task_net_gain=int(task[1]-task[2]),agreement_net_gain=int(agree[1]-agree[2]))
    assert dict(counts)==r['named_expression_counts']
    assert r['archive_reads']==385 and r['projection_calls']==192
    assert r['framework_forward_calls']==r['fit_calls']==r['random_calls']==0
    result=dict(complete=True,pass_all=True,task='T224 independent full saved projection audit',raw_commit='45ab5489',delegation='ad8043fd',pins={n:sha(BASE/n) for n in ['saved/results.json','saved/progress.jsonl','saved_protocol.json','saved_gate.json','saved.py','primitive.py']},prior_review_sha256=sha(HERE/'residual_full_saved_review.json'),checker_sha256=sha(__file__),rows=196608,chunks=192,coordinates=3145728,progress_events=len(events),unchanged_rows=unchanged_total,panels=report,named_expression_counts=dict(counts),max_consistency_differences=errors,independent_column_readout_byte_equal_chunks=readout_byte_chunks,author_cpu_seconds=r['cpu_seconds'],review_cpu_seconds=time.process_time()-start,author_model_fit_rng_calls=0,scope='All saved array/pin/coverage bindings, projection ownership, separate readout column contraction, SSE fsum consistency, exact first-index saved labels, paired/context tables and clipping sums. Exact coordinate monotonicity follows from h>=0 and z^2-2zh>=0 for z<0; no task-accuracy monotonicity.',limitations=['Existing panels were reused after original results, not fresh validation.','Readout and SSE consistency uses standard rounding bounds, not exact-real equality of rounded computations.','No recurrent network, fit or candidate function rerun.','Named expression categories are partial, not a full cost or T196 comparison.','All task regressions and label changes remain retained.'])
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'complete':True,'pass_all':True,'sha256':sha(dest),'review_cpu':result['review_cpu_seconds'],'panels':{k:(v['agreement_net_gain'],v['task_net_gain']) for k,v in report.items()},'errors':errors}))

if __name__=='__main__':
    try: main()
    except BaseException:
        (HERE/'t224_full_reviewer_failure.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2)+'\n');raise
