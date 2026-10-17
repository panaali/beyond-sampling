"""Independent saved full-seed H5 Hopf site/closure audit; no trace replay."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import ast,gzip,hashlib,json,time,sys
P=Path('/workspace/work/recover-small_h2_910/review/hopf_h5_native')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
helper=Path(__file__).with_name('check_h5_hopf_controls_review.py');tree=ast.parse(helper.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ledger');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(helper),'exec'))
start=time.process_time();seed=int(sys.argv[1]);assert seed in range(5)
target=Path(__file__).with_name(f'h5_hopf_seed{seed}_review_results.json');assert not target.exists()
for name in ('panel_scope.json','pilot_scope.json'):
 for path,h in read(P/name)['pins'].items():assert sha(path)==h
models=read(P/'models.json');assert models['complete'];rows=[]
for copy in range(5):
 i=5*seed+copy;m=models['models'][i];assert (m['seed'],m['copy'])==(seed,copy)
 meta=P/('pilot.json' if i==0 else f'panel/{i:02d}.json');archive=P/('pilot_ledger.json.gz' if i==0 else f'panel/{i:02d}_ledger.json.gz');r=read(meta)
 assert r['complete'] and r['phase']=='complete' and (r['seed'],r['copy'],r['nu'],r['nd'])==(seed,copy,64,128)
 assert r['source_sha256']==sha(P/('pilot.py' if i==0 else 'panel.py')) and r['ledger_sha256']==sha(archive)
 assert r['model_sha256']==m['model_sha256'] and r['value_hex']==r['baseline_hex']==r['saved_hex']==m['saved_h64_hex']==float(r['value']).hex()
 assert r['value']==r['baseline']==r['saved_value']==m['saved_h64'] and r['legacy']==r['baseline_legacy']==r['saved_legacy']==m['saved_legacy']
 if i==0:
  prior=read(Path(__file__).with_name('h5_hopf_pilot_review_results.json'));assert prior['complete'] and prior['pilot_sha256']==sha(meta) and prior['ledger_sha256']==sha(archive)
  explicit=prior['explicit'];closure=prior['closures'];startup=1;count=57344
 else:
  with gzip.open(archive,'rt') as stream:raw=json.load(stream)
  work=ledger(raw['report']);cold=ledger(raw['startup']);explicit=work['explicit']+cold['explicit'];closure=work['closures'];startup=cold['explicit'];count=len(raw['report']['source_closures'])
  assert r['opaque_calls']==raw['report']['opaque_calls']
 assert count==r['source_closure_count']==57344 and explicit==r['explicit_with_constant_setup'] and startup==r['startup_explicit']==1
 conditional=explicit+sum(closure.values());assert conditional==r['conditional_source_with_constant_setup']
 rows.append(dict(index=i,seed=seed,copy=copy,model_sha256=r['model_sha256'],value=r['value'],value_hex=r['value_hex'],legacy=r['legacy'],explicit=explicit,conditional=conditional,closures=closure,closure_count=count,cpu_seconds=r['cpu_seconds'],meta_sha256=sha(meta),ledger_sha256=sha(archive),pilot_reused_once=i==0))
assert len({r['model_sha256'] for r in rows})==5
out=dict(complete=True,seed=seed,rows=rows,totals={k:sum(r[k] for r in rows) for k in ('legacy','explicit','conditional','closure_count','cpu_seconds')},review_cpu=time.process_time()-start,new_evaluator_calls=0,scope='Five original fixed H5 models, all saved site sums and source closure formulas checked; previously reviewed pilot reused once. Conditional category allowance, not full native/runtime, mask repair or accuracy claim. Original uncertainty retained.')
target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
