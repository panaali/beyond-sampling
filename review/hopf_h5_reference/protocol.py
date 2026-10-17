"""Declare fixed H5 independent reference identities before any random draws."""
import hashlib,json,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;COST=HERE.parent/'hopf_h5_native'
H4=Path('/workspace/work/recover-small_h4_4/m16/approaches/hopf_reference_precision')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 dst=HERE/'protocol.json';assert not dst.exists()
 files={'models':COST/'models.json','summary':COST/'panel_summary.json','comparison':COST/'panel_comparison.json','forward':COST/'frozen/zoo44.py','h4_protocol':H4/'protocol.json','h4_reference_source':H4/'reference.py'}
 models=json.loads(files['models'].read_text());summary=json.loads(files['summary'].read_text());comp=json.loads(files['comparison'].read_text())
 assert models['complete'] and summary['complete'] and comp['complete'] and len(models['models'])==len(summary['rows'])==25
 h4=json.loads(files['h4_protocol'].read_text());excluded={7,100,101,102,103,104}|{r['rng_seed'] for r in h4['models']}
 rows=[]
 for idx,m in enumerate(models['models']):
  assert (m['seed'],m['copy'])==divmod(idx,5);r=summary['rows'][idx];assert r['model_sha256']==m['model_sha256'] and r['value_hex']==m['saved_h64_hex']
  name=f"T11594-hopf-H5T4-seed{m['seed']}-copy{m['copy']}-reference-v1";seed=zlib.crc32(name.encode('ascii'));assert seed not in excluded
  rows.append(dict(index=idx,seed=m['seed'],copy=m['copy'],model_sha256=m['model_sha256'],estimator_hex=r['value_hex'],conditional_cost=r['conditional'],rng_name=name,rng_seed=seed))
 assert len({r['rng_seed'] for r in rows})==25
 out=dict(claim='b55c0779',H=5,T=4,models=rows,files={k:{'path':str(v),'sha256':sha(v)} for k,v in files.items()},draws_per_model=100000000,block_size=1000000,blocks_per_model=100,confidence_family=25,alpha=.05,first_pilot=[0,0,0],total_projection_gate_forward_cpu_seconds=900,primary_cpu_soft=105,primary_cpu_hard=110,source_cost_reviews=['79f9bef8','d0458af9'],scope='Fresh diagnostic reference only. Fixed25 family/count/streams before any draw; old75-family10M result separate, no count combination, tuning or outcome-based stopping. First1M pilot reused exactly once after source/saved/throughput gate; no H4 truth or cost transfer.')
 dst.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'models':len(rows),'draws_per_model':out['draws_per_model'],'random_draws':0}))
if __name__=='__main__':main()
