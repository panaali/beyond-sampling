"""Contract exactly2500 saved blocks into fixed25-family evaluation intervals."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.special import betainc,betaincinv
HERE=Path(__file__).resolve().parent
COST=HERE.parent/'hopf_native_cost'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 dst=HERE/'reference_summary.json';assert not dst.exists();start=time.process_time()
 protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['draws_per_model']==100000000 and protocol['block_size']==1000000 and protocol['blocks_per_model']==100 and protocol['confidence_family']==25
 rows=[];boxes=[];totalcpu=0;pin=sha(HERE/'protocol.json');src=sha(HERE/'reference.py');tail=.05/(2*25)
 for model in protocol['models']:
  s,c=model['seed'],model['copy'];folder=HERE/f's{s}_c{c}';assert len(list(folder.glob('block*.json')))==100
  state=np.random.default_rng(model['rng_seed']).bit_generator.state;hits=0;cpu=0;records=[]
  for block in range(100):
   path=folder/f'block{block:03d}.json';r=json.loads(path.read_text())
   assert r['status']=='complete' and (r['seed'],r['copy'],r['block'])==(s,c,block) and r['n']==1000000
   assert r['protocol_sha256']==pin and r['source_sha256']==src and r['model_sha256']==model['model_sha256'] and r['rng_seed']==model['rng_seed']
   assert r['state_before']==state;state=r['state_after'];assert 0<=r['hits']<=1000000
   hits+=r['hits'];cpu+=r['cpu_seconds'];records.append({'block':block,'sha256':sha(path)})
  n=100000000;lo=0. if hits==0 else float(betaincinv(hits,n-hits+1,tail));hi=1. if hits==n else float(betaincinv(hits+1,n-hits,1-tail))
  residual=max(0. if hits==0 else abs(float(betainc(hits,n-hits+1,lo))-tail),0. if hits==n else abs(float(betainc(hits+1,n-hits,hi))-(1-tail)));assert residual<1e-9
  rows.append(dict(seed=s,copy=c,model_sha256=model['model_sha256'],rng_seed=model['rng_seed'],hits=hits,n=n,accuracy=hits/n,cpu_seconds=cpu,block_records=records));totalcpu+=cpu
  boxes.append(dict(H=4,T=4,seed=s,index=c,hits=hits,n=n,lower=lo,upper=hi,beta_tail_residual=residual))
 assert len(rows)==25 and len({r['rng_seed'] for r in rows})==25
 out=dict(rows=rows,mc_truth_boxes=boxes,family_size=25,nominal_joint_coverage=.95,reference_draws=2500000000,extra_control_evaluations=318,reference_cpu_seconds=totalcpu,protocol_sha256=pin,source_sha256=src,numerical_beta_not_outward_rounded=True,scope='Fresh fixed independent per-model reference streams only. Unchanged original estimator and source costs; no accuracy-adaptive sample count, full real-arithmetic truth or native cost certificate.',contraction_cpu_seconds=time.process_time()-start)
 dst.write_text(json.dumps(out,indent=2)+'\n')
 original=json.loads(Path('/workspace/work/recover-small_h2_78/m16/approaches/bulk_hopf_review/frozen/ev_44.json').read_text())
 for r in rows:original[str(r['seed'])]['rows'][r['copy']]['mc']=r['accuracy']
 fresh=HERE/'fresh_ev_44.json';assert not fresh.exists();fresh.write_text(json.dumps(original,indent=2)+'\n')
 print(json.dumps({'rows':25,'reference_draws':out['reference_draws'],'cpu_seconds':totalcpu,'first_accuracy':rows[0]['accuracy']}))
if __name__=='__main__':main()
