import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
P=Path('/workspace/work/recover-small_h2_910/review/recipe112_recovery');start=time.process_time();orig=json.loads((P/'snapshot/out/count.json').read_text());additional=json.loads((P/'additional_evidence.json').read_text());out=[]
for H in [4,5,6,8,16]:
 d=json.loads((P/f'counter_H{H}.json').read_text());assert d['complete'] and len(d['rows'])==25 and d['all_parity_pass'];assert {(r['zoo'],r['copy']) for r in d['rows']}=={(z,c) for z in range(5) for c in range(-1,4)}
 for p,h in d['source_hashes'].items():assert hashlib.sha256((P/p).read_bytes()).hexdigest()==h
 for p,h in d['checkpoint_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 for i,r in enumerate(d['rows']):
  assert 'failure' not in r and r['saved_original']==orig[str(H)]['rows'][i];assert r['author_counter']==r['saved_original']['charge'];assert r['parity_difference']==r['repaired']-r['float_reimplementation'];assert abs(r['parity_difference'])<=1e-12
 for field,rowfield in [('original_count_sum',None),('repaired_count_sum','repaired_counter'),('equality_count_sum','equality_operations')]:
  total=sum(r['saved_original']['auto'] if rowfield is None else r[rowfield] for r in d['rows']);assert total==d[field]
 charge=sum(r['author_counter'] for r in d['rows']);mult=F(d['repaired_count_sum'])*(3*H*(H+1)+3*H)/(d['sampler_original']*charge);saved=next(r for r in additional['count_repair'] if r['H']==H);assert abs(float(mult)-saved['corrected_multiplier'])<5e-16
 out.append(dict(H=H,rows=25,max_parity=max(abs(r['parity_difference']) for r in d['rows']),historical_sampler_multiplier=float(mult),repaired_count=d['repaired_count_sum'],equality_count=d['equality_count_sum']))
# Read source dependency evidence: sensitivity calls counted; fresh estimator calls original author float implementation.
sens=(P/'snapshot/sens.py').read_text();assert '.counted(' in sens
result=dict(rows=125,panels=out,sensitivity_wrapper_dependency_confirmed=True,review_cpu=time.process_time()-start,scope='Saved source/checkpoint hashes, counters and finite output parity only. Multipliers are historical panel transfers, not direct fresh-panel counts or full native cost. Sensitivity summaries inherit broken wrapper values.')
Path(__file__).with_name('recipe_counter_repair_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
