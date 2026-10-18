"""Saved-only first64 matched-budget contraction; no performance gate selection."""
import gzip,hashlib,importlib.util,json,time
from collections import Counter
from pathlib import Path
from provenance import HERE,ROOT,sha
def main():
 start=time.process_time();path=HERE/'pilot_summary.json';assert not path.exists()
 data=json.loads((HERE/'pilot.json').read_text());assert data['complete'] and [r['copy'] for r in data['rows']]==list(range(64))
 archive=HERE/'pilot.jsonl.gz';assert sha(archive)==data['archive_sha256']
 with gzip.open(archive,'rt') as f:raw=[json.loads(line) for line in f]
 assert len(raw)==64
 for row,record in zip(data['rows'],raw):
  assert row['copy']==record['copy'] and row['model_sha256']==record['model_sha256']
  for key in ('hex','legacy','stats','info'):assert record['baseline'][key]==record['traced'][key]
  ledger=record['traced']['ledger'];assert row['numeric']==ledger['named_numeric_total'] and row['integer']==ledger['integer_control_total']
 helper=ROOT/'recipe112_recovery/summarize_direct.py'
 spec=importlib.util.spec_from_file_location('t186_prior_summary',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 groups={}
 for convention,sampler,costkey in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
  rows=[dict(estimate=r['value'],truth=r['truth'],ops=r[costkey],parity_difference=0.,cell_count_equal=True,parity_pass=True) for r in data['rows']]
  group=module.summarize(rows,sampler)
  group['individual_losses']=sum((r['estimate']-r['truth'])**2>r['truth']*(1-r['truth'])/(r['ops']//sampler) for r in rows)
  groups[convention]=group
 out=dict(complete=True,groups=groups,rows=64,numeric=sum(r['numeric'] for r in data['rows']),integer=sum(r['integer'] for r in data['rows']),pilot_cpu=data['cpu'],projected_64_shard_cpu=64*data['cpu'],projection_scope='Finite pilot scaling only; not runtime guarantee or production cost',cpu=time.process_time()-start,pilot_sha256=sha(HERE/'pilot.json'),archive_sha256=sha(archive),helper_sha256=sha(helper),driver_sha256=sha(Path(__file__)),scope='First64 original copies only. Conditional saved-truth/source-unit comparisons, no full panel/runtime or rung claim. Bootstrap1000 RNG0 matches existing summary convention; no outcome-based continuation gate.')
 path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
