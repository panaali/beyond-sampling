"""Saved target dyadic and planar certificate review; no checkpoint decode/author calls."""
from check_t219_controls_saved import cone_check,history_forms,rays_of,primitive,dot,sha
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,time
import numpy as np
P=Path('/workspace/work/recover-small_h2_34/review/target_prefix_recount');H=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def main():
 start=time.process_time();r=read(P/'target_k2_result.json');p=read(P/'target_k2_protocol.json');g=read(P/'target_k2_gate.json')
 assert r['status']=='complete' and r['gate']==g and r['protocol_sha256']==g['protocol_sha256']==sha(P/'target_k2_protocol.json') and g['clearance']
 for n,h in {**p['pins'],**g['pins']}.items():assert sha(n)==h
 prior=g['control_saved_review'];assert sha(prior['path'])==prior['sha256'] and read(prior['path'])['pass_all']
 meta=read(p['prior_metadata']);assert meta['checkpoint_sha256']==sha(p['checkpoint'])
 with np.load(p['weights_archive'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
 assert set(arrays)=={'Wi','Wh','Wo'}
 mapping={'Wi':('rnn.weight_ih_l0',(16,1)),'Wh':('rnn.weight_hh_l0',(16,16)),'Wo':('linear.weight',(10,16))}
 for name,a in arrays.items():
  assert a.dtype==np.float64 and np.isfinite(a).all()
  digest=hashlib.sha256(a.tobytes()).hexdigest();assert r['archive_arrays'][name]==dict(shape=list(a.shape),dtype='float64',sha256=digest) and meta['archives']['weights.npz']['arrays'][name]['bytes_sha256']==digest
  key,shape=mapping[name];f32=a.astype(np.float32).reshape(shape);assert np.array_equal(f32.astype(np.float64).reshape(a.shape),a)
  digest32=hashlib.sha256(f32.tobytes()).hexdigest();assert r['checkpoint_arrays'][key]==dict(shape=list(shape),dtype='torch.float32',sha256=digest32) and meta['stored_weights'][key]['sha256']==digest32
 weights=r['weights'];D=weights['D'];wi=weights['wi'];W=weights['wh'];assert D==549755813888 and D&(D-1)==0
 frwi=[F(float(v)) for v in arrays['Wi']];frW=[[F(float(v)) for v in row] for row in arrays['Wh']]
 assert weights['wi_ratios']==[[v.numerator,v.denominator] for v in frwi] and weights['wh_ratios']==[[[v.numerator,v.denominator] for v in row] for row in frW]
 assert [F(v,D) for v in wi]==frwi and [[F(v,D) for v in row] for row in W]==frW
 assert D==max(v.denominator for v in frwi+[v for row in frW for v in row])
 assert weights['checkpoint_sha256']==sha(p['checkpoint']) and weights['archive_sha256']==sha(p['weights_archive'])
 case=dict(wi=wi,wh=W,D=D);levels=r['cone_levels'];assert [l['depth'] for l in levels]==[1,2]
 maps=[]
 for lev in levels:
  byhistory={}
  assert len(lev['states'])==len(lev['certificates'])
  for s,c in zip(lev['states'],lev['certificates'],strict=True):
   hist=tuple(map(tuple,s['history']));assert len(hist)==lev['depth'] and hist not in byhistory
   witness=cone_check(s,c['witness']);hidden,forms,zero=history_forms(case,hist)
   assert hidden==s['hidden'] and c['zero_forms']==zero and c['strict_forms']==len(forms) and all(dot(n,witness)>0 for n in forms)
   assert rays_of(forms,lev['depth'])==set(map(tuple,s['rays']));byhistory[hist]=s
  maps.append(byhistory)
 assert len(maps[0])==2 and set(maps[0])=={(tuple(int(s*v>0) for v in wi),) for s in (-1,1)}
 expected_histories=set();checks=[];rootcounts=[]
 for sign,half in zip((-1,1),r['planar'],strict=True):
  assert half['sign']==sign
  # Independently compute slope coefficients from exact stored real weights, then compare integer scale.
  h=[max(sign*v,F(0)) for v in frwi];alpha=[sum((a*b for a,b in zip(row,h)),F(0)) for row in frW];beta=frwi
  assert [F(v,D*D) for v in half['alpha']]==alpha and [F(v,D*D) for v in half['beta']]==beta
  roots=sorted(set(-a/b for a,b in zip(alpha,beta) if b));assert half['roots']==[[v.numerator,v.denominator] for v in roots];assert len(roots)==16;rootcounts.append(len(roots))
  intervals=list(zip([None]+roots,roots+[None]));assert len(half['rows'])==len(intervals)==17
  for row,(lo,hi) in zip(half['rows'],intervals,strict=True):
   assert row['bounds']==[None if v is None else [v.numerator,v.denominator] for v in (lo,hi)]
   v=F(*row['representative']);assert (lo is None or lo<v) and (hi is None or v<hi)
   mask=tuple(int(a+b*v>0) for a,b in zip(alpha,beta));hist=(tuple(int(sign*w>0) for w in wi),mask);assert row['history']==[list(x) for x in hist]
   assert hist not in expected_histories;expected_histories.add(hist)
   rays={ (0,-1) if lo is None else primitive((sign*lo.denominator,lo.numerator)),(0,1) if hi is None else primitive((sign*hi.denominator,hi.numerator))}
   s=maps[1][hist];assert rays==set(map(tuple,row['rays']))==set(map(tuple,s['rays'])) and row['hidden']==s['hidden']
   checks.append(dict(history=[list(x) for x in hist],rays_pass=True,hidden_pass=True))
 assert expected_histories==set(maps[1]) and len(expected_histories)==r['count']==r['author_reported_k2']==34 and r['author_count_agrees'] and r['comparisons']==checks
 assert [r[k] for k in ['checkpoint_decodes','model_calls','mass_calls','random_calls']]==[1,0,0,0]
 events=[json.loads(x) for x in (P/'target_k2_progress.jsonl').read_text().splitlines()]
 assert events==[dict(phase='attempt',stage='provenance'),dict(phase='result',stage='provenance',coefficients=432,recurrence_coefficients=272),dict(phase='attempt',stage='independent_planar'),dict(phase='result',stage='independent_planar',count=34),dict(phase='attempt',stage='cone_k2'),dict(phase='result',stage='cone_k2',count=34,author_count_agrees=True)]
 assert all(type(v) is int and v>=0 for v in r['cone_counts'].values()) and r['cone_counts']['lifts']==2
 out=dict(complete=True,pass_all=True,raw_commit='e08f856f',pins={n:sha(P/n) for n in ['target_k2_result.json','target_k2_progress.jsonl','target_k2_protocol.json','target_k2_gate.json']},helper_sha256=sha(H/'check_t219_controls_saved.py'),checker_sha256=sha(__file__),weight_coefficients=432,exact_recurrence_coefficients=272,common_denominator=D,roots_per_half=rootcounts,k1_histories=2,k2_histories=34,cone_certificates=36,progress_events=6,author_cpu=r['cpu_seconds'],review_cpu=time.process_time()-start,new_checkpoint_decodes=0,new_author_geometry_model_calls=0,reported_named_counts=r['cone_counts'],scope='Pinned saved archive bytes and inherited independently reviewed checkpoint mapping; no checkpoint decode. Exact dyadic ratio/float32 recovery and planar interval coverage, closed-column maps, all cone/extreme-ray/strict-interior certificates checked. Confirms34 positive-volume exact stored-weight gate histories at k2 only; not minimal affine regions, k3/k4/T10 growth, mass or cost. Internal operation/peak-bit execution not replayed.')
 (H/'t219_k2_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
