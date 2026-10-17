import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,time
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/fibre_t4_native_cost');prior=P.parent/'fibre_t4_recovery';start=time.process_time()
def read(p):return json.loads(p.read_text())
summary=read(P/'summary.json');sha=hashlib.sha256((P/'tracer.py').read_bytes()).hexdigest();checked=[]
for name in ['primitive','zero','constant0','plus3_minus0','plus0_minus3','component']:
 r=read(P/('component.json' if name=='component' else f'controls_{name}.json'));assert r['complete'] and r['tracer_sha256']==sha;ledger=r['ledger'];tot=Counter()
 for site,c in ledger['sites'].items():tot.update(c)
 assert dict(tot)==ledger['totals'];inc=sum(tot.values());ints=sum(v for k,v in tot.items() if k.startswith('integer_control'));assert inc==ledger['inclusive_explicit_total'] and ints==ledger['integer_control_total'] and inc-ints==ledger['named_numeric_total'];diag=sum(sum(c.values()) for site,c in ledger['sites'].items() if "stats['traces'] += 1" in site or "stats['pieces'] += 1" in site)
 row_counts=Counter()
 for site,c in ledger['sites'].items():
  if site.startswith('rows_at:'):row_counts.update(c)
 rowints=sum(v for k,v in row_counts.items() if k.startswith('integer_control'));rownum=sum(row_counts.values())-rowints
 saved=summary['component'] if name=='component' else summary['controls'][name];assert inc==saved['inclusive_explicit'] and diag==saved['diagnostic_stats_units_included'] and inc-diag==saved['inclusive_after_diagnostic_stats'];assert rownum==saved['rows_at_named'] and rowints==saved['rows_at_integer']
 if name=='primitive':assert rownum==892 and rowints==16 and len(r['rows'])==12
 else:
  old=read(prior/('noise_preflight.json' if name=='component' else f'guard_{name}.json'))
  if name=='component':assert r['raw_error_component']==old['raw_error_component'] and r['model_sha256']==old['model_sha256'];expected=old
  else:expected=old['arms'][0];assert r['value']==expected['estimate']
  assert r['legacy_ops']==expected['legacy_ops'] and r['stats']==expected['stats'];assert diag==sum(r['stats'].values())
 if name=='component':assert inc==59028138 and diag==4794 and inc-diag==59023344;assert rowints==4*2595 and rownum==2095*2595;assert not ledger['opaque_calls']
 checked.append(dict(name=name,inclusive=inc,diagnostic_stats=diag,visible_without_diagnostic=inc-diag,rows_at_numeric=rownum,rows_at_integer=rowints))
# Independent direct dense expansion of rows_at.
def dense(H,T):return (T-1)*H*T*(2*H-1)+(T-1)*H+T*H*(2*T-1)+T*H+T*H*T+T*T*(2*H-1)+H-1
assert dense(2,4)==223 and dense(8,4)==2095
out=dict(checked=checked,trained_components=1,visible_to_legacy=59023344/37704799,review_cpu=time.process_time()-start,scope='Saved source-ledger arithmetic and baseline parity; no trace rerun. Only one fixed noisy component, no whole-model multiplier. Includes explicit integer expressions; diagnostics removed separately. Cold/native/library/control/memory exclusions remain, no complete-cost claim.')
Path(__file__).with_name('fibre_native_component_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
