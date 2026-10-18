"""Independent saved-array residual/context reconciliation. No author imports or RNN/refit."""
from pathlib import Path
from collections import Counter
import json,hashlib,time,math,traceback
import numpy as np
P=Path('/workspace/work/recover-small_h2_56/review/mechanism_residual_margin');ROOT=P.parents[1];H=Path(__file__).resolve().parent
DATA=ROOT/'review/mechanism_claims_review/rebuild'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def bits(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype
 assert a.tobytes()==b.tobytes()
def close_sum(a,b,scale,n):
 # Conservative standard binary64 sum consistency allowance; not a candidate certificate.
 gamma=(n*2**-53)/(1-n*2**-53)
 assert np.all(np.abs(a-b)<=4*gamma*scale+1e-300)
 return float(np.max(np.abs(a-b)))
def main():
 start=time.process_time();r=read(P/'saved/results.json');p=read(P/'saved_protocol.json');g=read(P/'saved_gate.json');m=read(P/'manifest.json')
 assert r['complete'] and r['protocol_sha256']==sha(P/'saved_protocol.json') and r['gate_sha256']==sha(P/'saved_gate.json')
 assert g['passed'] and g['protocol_sha256']==r['protocol_sha256']
 for path,digest in {**p['pins'],**g['additional_pins']}.items():assert sha(path)==digest
 for path,digest in m['pins'].items():assert sha(ROOT/path)==digest
 old=read(DATA/'fullfit/results.json');assert old['complete']
 prior={(v['panel'],v['index']):v for v in old['chunks']};assert len(prior)==191
 pp=read(DATA/'FULLFIT_PROTOCOL.json')['pins']
 W=load(DATA/'pilot/weights.npz')['Wo'];B=load(DATA/'fullfit/lookup_coefficients.npz')['coefficients'];assert W.shape==(10,16) and B.shape==(216,3,16)
 legal=sorted((c*9+a)*9+b for c in range(3) for a in range(9) for b in range(9) if a!=b);slots={v:i for i,v in enumerate(legal)}
 expected_pairs=[(panel,i) for panel in ['train','holdout1','holdout2'] for i in range(64)]
 assert [(c['panel'],c['index']) for c in r['chunks']]==expected_pairs
 events=[json.loads(s) for s in (P/'saved/progress.jsonl').read_text().splitlines()]
 ee=[dict(stage='started'),dict(stage='runtime_bound',runtime=r['runtime'])]
 for panel in ['train','holdout1','holdout2']:
  for i in range(64):ee.extend([dict(stage='chunk_attempt',panel=panel,index=i),dict(stage='chunk_completed',panel=panel,index=i)])
  ee.append(dict(stage='panel_completed',panel=panel))
 ee.append(dict(stage='completed',results_sha256=sha(P/'saved/results.json')));assert events==ee
 assert r['runtime']==dict(executable=p['invoked_python'],prefix=p['venv_prefix'],numpy=p['numpy_version'],numpy_file=p['numpy_file'])
 totals=Counter();panels={};max_reconstruction={'hidden':0.,'logits':0.};byte_equal={'hidden':0,'logits':0};max_group_error=0.
 for panel in ['train','holdout1','holdout2']:
  labels=load(DATA/f'fullfit/{panel}_labels.npz');table=np.zeros(450,dtype=np.int64);sums=[[[] for _ in range(16)] for _ in range(5)];abss=[[[] for _ in range(16)] for _ in range(5)];closures=[];zeros=0;ties=[0,0];ratio_max=0.;n=0
  for c in [v for v in r['chunks'] if v['panel']==panel]:
   i=c['index'];assert c['complete'];src=c['source']
   for role in ['inputs','outputs']:assert sha(ROOT/src[role])==src[role+'_sha256']
   if panel=='train' and i==0:
    for role,fn in [('inputs','inputs.npz'),('outputs','candidate.npz')]:
     assert src[role]==str((DATA/'pilot'/fn).relative_to(ROOT));assert src[role+'_sha256']==next(v for k,v in pp.items() if k.endswith('/pilot/'+fn))
   else:
    z=prior[(panel,i)]
    for role in ['inputs','outputs']:assert Path(src[role]).name==Path(z[role]['path']).name and src[role+'_sha256']==z[role]['sha256']
   assert sha(ROOT/c['output']['path'])==c['output']['sha256']
   inp=load(ROOT/src['inputs']);out=load(ROOT/src['outputs']);a=load(ROOT/c['output']['path'])
   assert {k:list(v.shape) for k,v in a.items()}==c['output']['arrays']
   assert all(np.isfinite(v).all() for v in list(inp.values())+list(out.values())+list(a.values()))
   x=inp['x'];h=out['hidden'];l=out['logits'];hh=a['lookup_hidden'];ll=a['lookup_logits'];ix=np.arange(1024);sl=slice(i*1024,(i+1)*1024)
   assert x.shape==(1024,10) and h.shape==hh.shape==(1024,16) and l.shape==ll.shape==(1024,10)
   # Input-derived stable sorting independently checks all saved target/state/context labels.
   order=np.argsort(-x,axis=1,kind='stable');head=np.argsort(-x[:,:9],axis=1,kind='stable');p1=head[:,0];p2=head[:,1];truth=order[:,1]
   case=np.where(x[:,9]>x[ix,p1],0,np.where(x[:,9]>x[ix,p2],1,2));key=(case*9+p1)*9+p2;positive=np.minimum(np.count_nonzero(x>0,axis=1),2)
   for name,v in [('truth',truth),('case',case),('state_key',key),('positive_bin',positive)]:bits(a[name],v.astype(a[name].dtype))
   bits(a['truth'],labels['true_labels'][sl]);bits(a['state_key'],labels['state_key'][sl]);bits(a['case'],labels['case'][sl]);bits(a['positive_bin'],labels['positive_bin'][sl])
   coeff=B[[slots[int(k)] for k in key]];feature=np.column_stack((x[ix,p1],x[ix,p2],x[:,9]));products=[feature[:,j,None]*coeff[:,j,:] for j in range(3)];recon=(products[0]+products[1])+products[2];scale=sum(np.abs(z) for z in products)
   max_reconstruction['hidden']=max(max_reconstruction['hidden'],close_sum(recon,hh,scale,6));byte_equal['hidden']+=int(recon.tobytes()==hh.tobytes())
   q=[hh[:,j,None]*W[None,:,j] for j in range(16)];recon=q[0].copy()
   for z in q[1:]:recon=recon+z
   max_reconstruction['logits']=max(max_reconstruction['logits'],close_sum(recon,ll,sum(np.abs(z) for z in q),32));byte_equal['logits']+=int(recon.tobytes()==ll.tobytes())
   # Stable sorts make first-index ties and exclusion explicit, independent of author argmax.
   pred=np.argsort(-l,axis=1,kind='stable')[:,0];look=np.argsort(-ll,axis=1,kind='stable')[:,0];ranked=np.argsort(-l,axis=1,kind='stable');comp=np.where(ranked[:,0]==truth,ranked[:,1],ranked[:,0])
   bits(a['network'],pred);bits(a['lookup'],look);bits(a['competitor'],comp);bits(pred,labels['network'][sl]);bits(look,labels['lookup'][sl])
   trans=np.full(1024,4,dtype=np.int64);trans[pred==look]=3;trans[look==truth]=1;trans[pred==truth]=2;trans[(pred==truth)&(look==truth)]=0;bits(a['transition'],trans)
   terms=(hh-h)*(W[truth]-W[comp]);bits(a['terms'],terms);term_sum=terms[:,0].copy();abs_sum=np.abs(terms[:,0])
   for j in range(1,16):term_sum+=terms[:,j];abs_sum+=np.abs(terms[:,j])
   om=l[ix,truth]-l[ix,comp];lm=ll[ix,truth]-ll[ix,comp];delta=lm-om;closure=delta-term_sum
   for name,v in [('term_sum',term_sum),('term_abs_sum',abs_sum),('original_margin',om),('lookup_margin',lm),('margin_delta',delta),('closure_residual',closure)]:bits(a[name],v)
   rank=1+np.sum(x>x[ix,pred,None],axis=1);bits(a['network_global_rank'],rank)
   for j,(name,logs,preds) in enumerate([('network',l,pred),('lookup',ll,look)]):
    nt=np.sum(logs==logs[ix,preds,None],axis=1);bits(a[name+'_max_ties'],nt);ties[j]+=int(np.sum(nt>1))
   table+=np.bincount((((positive*3+case)*10+(rank-1))*5+trans),minlength=450)
   for t in range(5):
    selected=terms[trans==t]
    for j in range(16):sums[t][j].extend(selected[:,j].tolist());abss[t][j].extend(np.abs(selected[:,j]).tolist())
   closures.extend(np.abs(closure).tolist());zeros+=int(np.sum(term_sum==0));nz=term_sum!=0
   if np.any(nz):ratio_max=max(ratio_max,float(np.max(abs_sum[nz]/np.abs(term_sum[nz]))))
   count=dict(subtractions=1024*36,multiplications=1024*16,additions=1024*30,absolute_values=1024*16);assert c['counts']==count;totals.update(count)
   assert c['max_abs_closure']==float(np.max(np.abs(closure)));n+=1024
  pr=r['panels'][panel];context=table.reshape(3,3,10,5);trans_count=context.sum(axis=(0,1,2));assert context.tolist()==pr['context_counts'] and trans_count.tolist()==pr['transition_counts']
  assert context.sum(axis=3).tolist()==old['panels'][panel]['network']['cross_tab']['table']
  ss=pr['stats'];assert n==ss['n']==65536 and max(closures)==ss['max_abs_closure'] and ss['zero_term_sums']==zeros and ss['network_tie_rows']==ties[0] and ss['lookup_tie_rows']==ties[1] and ss['max_finite_cancellation_ratio']==ratio_max
  close_sum(np.array(math.fsum(closures)),np.array(ss['closure_l1']),np.array(math.fsum(closures)),65536)
  for field,v in [('neuron_signed_sum',sums),('neuron_absolute_sum',abss)]:
   exactish=np.array([[math.fsum(z) for z in row] for row in v]);scales=np.array([[math.fsum(z) for z in row] for row in abss]);max_group_error=max(max_group_error,close_sum(exactish,np.array(pr[field]),scales,65536))
  panels[panel]=dict(rows=n,transitions=trans_count.tolist(),max_closure=max(closures),cancellation_max=ratio_max)
 assert r['counts']==dict(lookup_rows=196608,readout_rows=196608,diagnostic_rows=196608,diagnostic_calls=192,archive_reads=389)
 assert r['named_diagnostic_arithmetic']==dict(totals) and r['lookup_readout_expression_arithmetic']==dict(mul=196608*208,add=196608*182)
 result=dict(complete=True,pass_all=True,raw_commit='88d44ff9',pins={n:sha(P/n) for n in ['saved/results.json','saved/progress.jsonl','saved_protocol.json','saved_gate.json','manifest.json','saved.py','primitive.py']},checker_sha256=sha(__file__),chunks=192,rows=196608,term_entries=196608*16,progress_events=len(events),panels=panels,named_arithmetic=dict(totals),lookup_reconstruction_max_difference=max_reconstruction,lookup_reconstruction_bitwise_chunks=byte_equal,aggregate_fsum_max_difference=max_group_error,review_cpu=time.process_time()-start,author_cpu=r['cpu_seconds'],new_author_model_refit_RNG_calls=0,scope='All pinned saved rows/arrays, source bindings, stable first-index labels/competitor,16 terms and sequential binary64 fold/closure, context histograms/counts/progress independently reconciled. Lookup/readout source consistency uses separate scalar-column products with standard rounding allowance; original recurrent network and fitting are not rerun. Aggregate sums checked against math.fsum under conservative standard summation allowance. No causal, deployable-estimator, noisy-weight transfer, full cost or exact-real closure claim.')
 (H/'residual_full_saved_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
 try:main()
 except BaseException:
  (H/'residual_full_reviewer_failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2)+'\n');raise
