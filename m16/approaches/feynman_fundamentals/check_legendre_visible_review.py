"""Saved explicit Legendre site inventory; no eigenvalue evaluations."""
from pathlib import Path
from collections import Counter
import hashlib,json,resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/fibre_cold_native')
a=json.loads((P/'spectral/legendre_visible.json').read_text())
b=json.loads((P/'capture.json').read_text())
assert a['complete'] and len(a['rows'])==6
assert a['runner_sha256']==hashlib.sha256((P/'spectral/legendre_visible.py').read_bytes()).hexdigest()
manifest=json.loads((P/'spectral/capture_manifest.json').read_text())
assert a['source_sha256'] in manifest['files'].values()
numeric=[];integer=[];byfn=Counter();calls=Counter()
for r in a['rows']:
    old=b['rows'][r['saved_row']]
    assert old['name']=='leggauss' and old['args']['items'][0]==r['order']
    assert r['parity'] and r['output_hex']==[v['hex'] for v in old['output']['items']]
    ledger=r['ledger']; totals=Counter()
    for site,c in ledger['sites'].items():
        assert all(isinstance(v,int) and v>=0 for v in c.values())
        totals.update(c);byfn[site.split(':')[0]]+=sum(c.values())
    assert dict(totals)==ledger['totals']
    ni=sum(v for k,v in totals.items() if not k.startswith('integer_control_'))
    ii=sum(v for k,v in totals.items() if k.startswith('integer_control_'))
    assert ni==ledger['named_numeric_total'] and ii==ledger['integer_control_total']
    assert ledger['complete_implementation_cost'] is False
    numeric.append(ni);integer.append(ii);calls.update(x['name'] for x in r['calls'])
    n=r['order']
    assert totals['special_np.sqrt']==n and totals['special_np.abs']==2*n
    assert totals['reduce_.max']==2*(n-1) and totals['reduce_.sum']==n-1
assert numeric==[570,150,150,570,1550,3920]
assert sum(numeric)==6910 and sum(integer)==1018
assert calls['np.linalg.eigvalsh']==6
out=dict(complete=True,rules=6,numeric_by_order=numeric,integer_by_order=integer,
         numeric_sum=sum(numeric),integer_sum=sum(integer),inclusive_sum=sum(numeric+integer),
         inclusive_by_function=dict(byfn),helper_call_census=dict(calls),
         raw_sha256=hashlib.sha256((P/'spectral/legendre_visible.json').read_bytes()).hexdigest(),
         scope='Explicit transformed source expressions only. Integer-operand divisions can yield floating values, so the separate integer category is not discarded. Eigenvalue, polynomial utility, reduction internals, interpreter/control/memory remain unclosed.',
         new_rule_eigenvalue_model_calls=0)
(HERE/'legendre_visible_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
