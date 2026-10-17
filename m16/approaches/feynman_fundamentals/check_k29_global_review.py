"""Compose five completed independent seed audits and clean audit without replay."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,time
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_910/review/margin_k29_native')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.process_time();s=read(P/'full_summary.json');assert s['complete'] and not s['partial'] and s['total_rows']==5120
hashes=set();files={};num=Counter();integer=Counter();cpu=0;max_shard=0;reused=0
for seed in range(5):
    r=read(HERE/f'k29_seed{seed}_review_results.json');assert r['complete'] and r['seed']==seed and r['models']==1024
    assert not hashes.intersection(r['model_hashes']);hashes.update(r['model_hashes'])
    assert len(r['pins'])==16
    for path,digest in r['pins'].items():
        assert path not in files and sha(P/path)==digest==s['files'][path]
        files[path]=digest;d=read(P/path);cpu+=d['cpu'];max_shard=max(max_shard,d['cpu'])
        assert len(d['rows'])==64 and d['complete']
        if path=='pilot.json':reused+=1
        for row in d['rows']:
            for name in ('r85b','r112'):
                num[name]+=row['recipes'][name]['numeric'];integer[name]+=row['recipes'][name]['integer']
    reported=next(g for g in s['groups'] if g['seed']==seed)
    for name in ('r85b','r112'):
        for convention in ('named_numeric','historical'):
            a=r['ratios'][name][convention];b=reported['recipes'][name][convention]
            assert a['ratio']==b['ratio_integer_budget'] and a['individual_losses']==b['individual_losses']
assert len(hashes)==5120 and len(files)==80 and reused==1
assert dict(num)==s['numeric_totals'] and dict(integer)==s['integer_totals']
clean=read(HERE/'k29_clean_review_results.json');assert clean['complete'] and clean['models']==5
assert clean['clean_sha256']==sha(P/'clean.json')==s['files']['clean.json']
assert len({r['model_sha256'] for r in clean['results']})==5 and not hashes.intersection(r['model_sha256'] for r in clean['results'])
assert set(s['files'])==set(files)|{'clean.json'}
cpu+=read(P/'clean.json')['cpu']
for seed in (0,4):cpu+=read(P/f'control_s{seed}.json')['cpu']
assert abs(cpu-s['research_cpu'])<1e-10
assert s['contraction_source_sha256']==sha(P.parent/'recipe112_recovery/summarize_direct.py')
out=dict(complete=True,noisy_models=5120,clean_models=5,noisy_ledgers=10240,clean_ledgers=10,
         shards=80,reused_pilot_once=True,control_rows_reused_once=True,
         numeric_totals=dict(num),integer_totals=dict(integer),research_cpu=cpu,max_shard_cpu=max_shard,
         final_summary_sha256=sha(P/'full_summary.json'),
         seed4_named_ratios={n:s['groups'][4]['recipes'][n]['named_numeric']['ratio_integer_budget'] for n in ('r85b','r112')},
         seed4_named_losses={n:s['groups'][4]['recipes'][n]['named_numeric']['individual_losses'] for n in ('r85b','r112')},
         review_cpu=time.process_time()-start,new_model_truth_calls=0,
         scope='Composition of all five independent 2048-ledger seed audits plus ten clean ledgers, global hashes/coverage and saved CPU. Named111/historical72 conventions stay separate; finite truth and conditional-bootstrap limitations remain. No hardware/full-cost or universal win claim.')
(HERE/'k29_global_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
