"""Pin a fixed independent evaluation panel before any reference input draws."""
import hashlib,json,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
COST=HERE.parent/'hopf_native_cost'
ROOT=Path('/workspace/work/recover-small_h2_78/m16/approaches/bulk_hopf_review')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 dst=HERE/'protocol.json';assert not dst.exists()
 files={'models':COST/'panel_models.json','summary':COST/'panel_summary.json','comparison':COST/'panel_comparison.json','forward':ROOT/'frozen/zoo44.py'}
 models=json.loads(files['models'].read_text());rows=[]
 for m in models['rows']:
  name=f"T11581-hopf-H4T4-seed{m['seed']}-copy{m['copy']}-reference-v1";seed=zlib.crc32(name.encode('ascii'))
  assert seed not in [7,100,101,102,103,104]
  rows.append(dict(seed=m['seed'],copy=m['copy'],model_sha256=m['model_sha256'],rng_name=name,rng_seed=seed))
 assert len({r['rng_seed'] for r in rows})==25
 out=dict(claim='c78c0999',models=rows,files={k:{'path':str(v),'sha256':sha(v)} for k,v in files.items()},draws_per_model=100000000,block_size=1000000,blocks_per_model=100,confidence_family=25,alpha=.05,first_pilot=[0,0,0],total_projection_gate_cpu_seconds=600,scope='Fresh independent diagnostic truth only; unchanged original estimator/cost. Fixed count, no estimate-dependent stopping or reuse of old MC stream; old75-family result remains separate. First pilot included once.')
 dst.write_text(json.dumps(out,indent=2)+'\n');print({'models':len(rows),'draws_per_model':out['draws_per_model']})
if __name__=='__main__':main()
