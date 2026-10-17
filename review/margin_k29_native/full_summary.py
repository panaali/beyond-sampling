"""Frozen per-seed full k29 contraction, never transferring model costs."""
import json,sys,time,hashlib
from pathlib import Path
from fractions import Fraction as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'recipe112_recovery'))
from summarize_direct import summarize

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics(rows,name):
 out={}
 for convention,sampler,costkey in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
  rr=[dict(estimate=r['recipes'][name]['value'],truth=r['truth'],ops=r['recipes'][name][costkey],parity_difference=0.,cell_count_equal=True,parity_pass=True) for r in rows]
  d=summarize(rr,sampler)
  ratios=[(F(r['estimate'])-F(r['truth']))**2/(F(r['truth'])*(1-F(r['truth']))/(r['ops']//sampler)) for r in rr]
  d.update(individual_losses=sum(x>1 for x in ratios),max_individual_ratio=float(max(ratios)),worst_copy_index=ratios.index(max(ratios)))
  out[convention]=d
 return out

def main(partial=False):
 start=time.process_time();pilot=json.loads((HERE/'pilot.json').read_text());assert pilot['complete']
 out=dict(complete=False,partial=partial,groups=[],files={},total_rows=0,scope='All frozen k29 models, two unchanged recipes. Numerical-source111 and historical72 conventions separate. Conditional1000-bootstrap excludes truth/geometry uncertainty; no fullhardware or population guarantee.',numeric_totals={'r85b':0,'r112':0},integer_totals={'r85b':0,'r112':0})
 for seed in range(5):
  paths=[HERE/('pilot.json' if (seed,noise,block)==(0,20,0) else f'panel/s{seed}_n{noise}_b{block}.json') for noise in range(20,24) for block in range(4)]
  if not all(p.exists() and json.loads(p.read_text()).get('complete') for p in paths):
   assert partial,('Incomplete seed',seed);continue
  rows=[]
  for (noise,block),path in zip(((n,b) for n in range(20,24) for b in range(4)),paths):
   d=json.loads(path.read_text());assert d['seed']==seed and d['noise']==noise and d['sampler_numeric']==111 and d['sampler_legacy']==72
   assert [r['copy'] for r in d['rows']]==list(range(64*block,64*(block+1)))
   archive=path.with_suffix('.jsonl.gz');assert sha(archive)==d['archive_sha256'];out['files'][str(path.relative_to(HERE))]=sha(path)
   rows.extend(dict(r,noise=noise) for r in d['rows'])
  assert len(rows)==1024 and len({r['model_sha256'] for r in rows})==1024
  for name in ('r85b','r112'):
   out['numeric_totals'][name]+=sum(r['recipes'][name]['numeric'] for r in rows);out['integer_totals'][name]+=sum(r['recipes'][name]['integer'] for r in rows)
  g=dict(seed=seed,recipes={name:metrics(rows,name) for name in ('r85b','r112')})
  out['groups'].append(g);out['total_rows']+=len(rows)
 clean_path=HERE/'clean.json'
 if clean_path.exists():
  d=json.loads(clean_path.read_text());assert d['complete'] and [r['seed'] for r in d['rows']]==list(range(5));out['clean']=[]
  for r in d['rows']:
   recipes={name:dict(value=v['baseline']['value'],legacy=v['baseline']['legacy'],numeric=v['traced']['ledger']['named_numeric_total']) for name,v in r['recipes'].items()}
   row=dict(truth=r['truth'],recipes=recipes)
   out['clean'].append(dict(seed=r['seed'],truth=r['truth'],independent_truth=r['independent_truth'],recipes={name:metrics([row],name) for name in ('r85b','r112')}))
  out['files']['clean.json']=sha(clean_path)
 out['complete']=len(out['groups'])==5 and len(out.get('clean',[]))==5
 if not partial:assert out['complete']
 out['research_cpu']=sum(json.loads((HERE/p).read_text())['cpu'] for p in out['files'])+sum(json.loads((HERE/f'control_s{s}.json').read_text())['cpu'] for s in (0,4))
 out['research_cpu_scope']='Completed included chunks plus two controls and clean; excludes provenance and earlier source work, and import-before-timer caveats remain. No repeated pilot/control estimates counted as new statistical models.'
 out['contraction_source_sha256']=sha(HERE.parent/'recipe112_recovery/summarize_direct.py');out['cpu_summary']=time.process_time()-start
 target=HERE/('full_summary_partial.json' if partial else 'full_summary.json')
 if not partial:assert not target.exists()
 target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=out['complete'],rows=out['total_rows'],groups=[dict(seed=g['seed'],ratios={name:v['named_numeric']['ratio_integer_budget'] for name,v in g['recipes'].items()}) for g in out['groups']],cpu=out['research_cpu'])))
if __name__=='__main__':main('--partial' in sys.argv)
