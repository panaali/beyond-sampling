"""All fixed H8 groups at each model's native named-arithmetic budget."""
import json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'recipe112_recovery'))
from summarize_direct import summarize

def main(partial=False):
 start=time.process_time();out=dict(complete=False,partial=partial,groups=[],cpu_replay=0.,sampler=351,scope='Named source arithmetic only, conditional on saved truths/floating geometry. No hardware, integer/control/memory, input-generation or model-free setup closure; no other H rungs priced.')
 for z in range(5):
  paths=[HERE/'panel'/f'H8_z{z}_n{noise}.json' for noise in (80,81,82,83)]
  if not all(p.exists() and json.loads(p.read_text())['complete'] for p in paths):
   assert partial,'Missing fixed group';continue
  rows=[]
  for noise,p in zip((80,81,82,83),paths):
   d=json.loads(p.read_text());assert d['zoo']==z and d['noise']==noise and d['sampler_numeric']==351
   assert [r['copy'] for r in d['rows']]==list(range(64))
   for r in d['rows']:
    assert r['parity'];rows.append(dict(r,parity_difference=r['estimate']-r['saved_estimate'],cell_count_equal=True,parity_pass=True))
   out['cpu_replay']+=d['cpu']
  g=summarize(rows,351);g['zoo']=z;out['groups'].append(g)
 out['complete']=len(out['groups'])==5;out['cpu_summary']=time.process_time()-start
 if out['groups']:
  out['worst_ratio']=max(g['ratio_integer_budget'] for g in out['groups']);out['largest_conditional_ci_upper']=max(g['bootstrap95_integer'][1] for g in out['groups'])
 (HERE/('panel_summary_partial.json' if partial else 'panel_summary.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main('--partial' in sys.argv)
