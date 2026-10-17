"""Independent saved-only clean-model audit; no author imports or model calls."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,time
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_910/review/margin_k29_native')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.process_time()
d=read(P/'clean.json');pin=read(P/'clean_pin.json')
assert d['complete'] and d['source_sha256']==sha(P/'clean.py')
assert d['pin_sha256']==sha(P/'clean_pin.json') and pin['sha256']==sha(Path(pin['source']))
assert (d['sampler_numeric'],d['sampler_legacy'])==(111,72)
original=read(Path(pin['source']))
assert all(r in original['rows'] for r in pin['rows'])
summary=read(P/('full_summary.json' if (P/'full_summary.json').exists() else 'full_summary_partial.json'))
assert len(d['rows'])==len(pin['rows'])==len(summary['clean'])==5
results=[]
for seed,(row,prior,reported) in enumerate(zip(d['rows'],pin['rows'],summary['clean'])):
    assert row['seed']==prior['training_seed']==reported['seed']==seed and prior['power']==29 and prior['copy']==-1
    assert row['model_sha256']==prior['model_sha256']
    assert row['truth']==prior['truth']['value']==reported['truth']
    assert row['independent_truth']==prior['independent']['value']==reported['independent_truth']
    p=F(row['truth']);pq=p*(1-p);rec={}
    for name,pair in row['recipes'].items():
        b,t=pair['baseline'],pair['traced']
        for k in ('hex','legacy','stats','info','recipe'):assert b[k]==t[k]
        assert b['hex']==float(b['value']).hex()==float(t['value']).hex()
        ledger=t['ledger'];total=Counter()
        for cats in ledger['sites'].values():total.update(cats)
        assert total==Counter(ledger['totals']) and not ledger['complete_implementation_cost']
        integer=sum(v for k,v in total.items() if k.startswith('integer_control'))
        numeric=sum(total.values())-integer
        assert (numeric,integer)==(ledger['named_numeric_total'],ledger['integer_control_total'])
        err=(F(b['value'])-p)**2;ratios={}
        for convention,sampler,ops in [('named_numeric',111,numeric),('historical',72,b['legacy'])]:
            g=reported['recipes'][name][convention];n=ops//sampler
            ratio=err*n/pq;continuous=err*ops/(pq*sampler)
            assert g['ratio_integer_budget']==float(ratio)
            assert g['ratio_continuous_budget']==g['ratio_mean_budget']==float(continuous)
            assert g['mse']==float(err) and g['mean_sampling_variance_integer']==float(pq/n)
            assert g['min_samples']==g['max_samples']==n and g['mean_ops']==ops
            assert g['individual_losses']==int(ratio>1) and g['max_individual_ratio']==float(ratio)
            assert g['worst_copy_index']==0
            # A one-row bootstrap is degenerate; allow only floating mean roundoff.
            assert all(abs(x-float(ratio))<=max(1e-30,abs(float(ratio))*2e-15) for x in g['bootstrap95_integer'])
            ratios[convention]=float(ratio)
        rec[name]=dict(error=float(F(b['value'])-p),numeric=numeric,integer=integer,ratios=ratios)
    results.append(dict(seed=seed,model_sha256=row['model_sha256'],recipes=rec))
out=dict(complete=True,models=5,ledgers=10,results=results,
         clean_sha256=sha(P/'clean.json'),pin_sha256=sha(P/'clean_pin.json'),
         review_cpu=time.process_time()-start,new_model_truth_calls=0,
         scope='Saved site sums, baseline parity, existing independently reviewed truth provenance and per-model budget arithmetic. Source111/historical72 remain separate; no complete cost or population guarantee.')
(HERE/'k29_clean_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
