"""Frozen equal-budget contractions, conditional on saved numerical truths."""
import hashlib,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'recipe112_recovery'))
from summarize_direct import summarize

def main():
 d=json.loads((HERE/'pilot.json').read_text());assert d['complete'] and [r['copy'] for r in d['rows']]==list(range(64))
 groups={}
 for name in ('r85b','r112'):
  groups[name]={}
  for convention,sampler,costkey in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
   rows=[dict(estimate=r['recipes'][name]['value'],truth=r['truth'],ops=r['recipes'][name][costkey],parity_difference=0.,cell_count_equal=True,parity_pass=True) for r in d['rows']]
   g=summarize(rows,sampler);g['individual_losses']=sum((r['estimate']-r['truth'])**2 > r['truth']*(1-r['truth'])/(r['ops']//sampler) for r in rows);groups[name][convention]=g
 out=dict(groups=groups,scope='First64 fixed seed0/noise20 copies only; cost/parity gate, not full rung. Conditional intervals exclude truth/geometry error. Named-source units exclude integer/control/memory/library internals/model-free setup.',pilot_sha256=hashlib.sha256((HERE/'pilot.json').read_bytes()).hexdigest(),contraction_source_sha256=hashlib.sha256((HERE.parent/'recipe112_recovery/summarize_direct.py').read_bytes()).hexdigest())
 p=HERE/'pilot_summary.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
