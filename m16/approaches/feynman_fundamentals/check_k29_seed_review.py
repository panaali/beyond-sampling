"""Incremental saved-only audit of one completed frozen1024-model seed."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import gzip,hashlib,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_910/review/margin_k29_native')
D=Path('/workspace/work/recover-small_h2_56/m16/approaches/training_axis_ladder')
seed=int(sys.argv[1]);assert 0<=seed<5
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ledger(l):
    total=Counter()
    for cats in l['sites'].values():total.update(cats)
    assert total==Counter(l['totals']) and not l['complete_implementation_cost']
    integer=sum(v for k,v in total.items() if k.startswith('integer_control'))
    numeric=sum(total.values())-integer
    assert (numeric,integer)==(l['named_numeric_total'],l['integer_control_total'])
    return numeric,integer
start=time.process_time();rows=[];pins={};totals=Counter()
for noise in range(20,24):
    old=read(D/f'block_s{seed}_n{noise}.json')
    for block in range(4):
        pilot=(seed,noise,block)==(0,20,0)
        path=P/('pilot.json' if pilot else f'panel/s{seed}_n{noise}_b{block}.json')
        d=read(path);assert d['complete'] and d['seed']==seed and d['noise']==noise
        archive=path.with_suffix('.jsonl.gz');assert d['archive_sha256']==sha(archive)
        assert d['provenance_sha256']==sha(P/'provenance.json')
        if not pilot:
            assert d['driver_sha256']==sha(P/'shard.py') and d['control_driver_sha256']==sha(P/'control.py')
            assert d['control4_sha256']==sha(P/'control_s4.json')
        pins[str(path.relative_to(P))]=sha(path)
        assert [r['copy'] for r in d['rows']]==list(range(block*64,(block+1)*64))
        with gzip.open(archive,'rt') as f:
            raw=[json.loads(line) for line in f]
        assert len(raw)==64
        for small,r in zip(d['rows'],raw):
            copy=small['copy'];truth=dict(zip(old['schema'],old['rows'][copy]))
            assert r['copy']==copy and r['model_sha256']==small['model_sha256']==truth['model_sha256']
            assert small['truth']==truth['truth']
            for name in ('r85b','r112'):
                pair=r['recipes'][name];b,t=pair['baseline'],pair['traced'];q=small['recipes'][name]
                for k in ('hex','legacy','stats','info','recipe'):assert b[k]==t[k]
                num,integer=ledger(t['ledger'])
                assert (num,integer)==(q['numeric'],q['integer'])
                assert q['hex']==float(q['value']).hex()==b['hex'] and q['value']==b['value']
                assert q['legacy']==b['legacy'] and q['cells']==b['stats']['cells']
                totals[name]+=num
                if (seed,noise,copy) in ((0,20,0),(4,23,255)):
                    assert pair==read(P/f'control_s{seed}.json')['recipes'][name]
            rows.append(small)
assert len(rows)==1024 and len({r['model_sha256'] for r in rows})==1024
summary_path=P/('full_summary.json' if (P/'full_summary.json').exists() else 'full_summary_partial.json')
summary=read(summary_path);reported=next(g for g in summary['groups'] if g['seed']==seed)
ratios={}
for name in ('r85b','r112'):
    ratios[name]={}
    ps=[F(r['truth']) for r in rows];err=[(F(r['recipes'][name]['value'])-p)**2 for r,p in zip(rows,ps)]
    mean=lambda xs:sum(xs,F(0))/1024
    for convention,sampler,cost in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
        g=reported['recipes'][name][convention];ops=[r['recipes'][name][cost] for r in rows]
        ns=[o//sampler for o in ops];pq=[p*(1-p) for p in ps];v=[p/n for p,n in zip(pq,ns)]
        rat=mean(err)/mean(v);individual=[e/d for e,d in zip(err,v)]
        assert float(rat)==g['ratio_integer_budget'] and float(mean(err))==g['mse']
        assert float(mean(v))==g['mean_sampling_variance_integer']
        assert float(mean(err)/mean(p*sampler/o for p,o in zip(pq,ops)))==g['ratio_continuous_budget']
        assert float(mean(err)/(mean(pq)*sampler/mean(map(F,ops))))==g['ratio_mean_budget']
        assert sum(v>1 for v in individual)==g['individual_losses']
        assert float(max(individual))==g['max_individual_ratio'] and individual.index(max(individual))==g['worst_copy_index']
        assert min(ns)==g['min_samples'] and max(ns)==g['max_samples'] and float(mean(map(F,ops)))==g['mean_ops']
        ix=np.random.default_rng(0).integers(0,1024,(1000,1024))
        bootstrap=np.array(list(map(float,err)))[ix].mean(1)/np.array(list(map(float,v)))[ix].mean(1)
        assert np.array_equal(np.percentile(bootstrap,[2.5,97.5]),g['bootstrap95_integer'])
        ratios[name][convention]=dict(ratio=float(rat),individual_losses=g['individual_losses'])
out=dict(complete=True,seed=seed,models=1024,recipes=2,numeric_totals=dict(totals),ratios=ratios,pins=pins,
         model_hashes=[r['model_sha256'] for r in rows],review_cpu=time.process_time()-start,
         scope='All saved2048 ledgers, parity/model/truth pins, exact own-budget contractions and frozen bootstrap. Existing numerical truths reused; source111/historical72 separate, no full-cost or population guarantee.',new_model_truth_calls=0)
(HERE/f'k29_seed{seed}_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('pins','model_hashes')}))
