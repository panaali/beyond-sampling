"""Independent raw471-instance expression ledger and matched-budget audit."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction as F
import hashlib,json,math,time
start=time.process_time();here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h2_34/m16/approaches/fibre_suite_cost');summary=json.loads((src/'summary.json').read_text());assert summary['complete'];oldsrc=Path('/workspace/clean-neurons/m16/approaches/fibre_constraint_review');reference=json.loads((oldsrc/'suite_reference.json').read_text());manifest=json.loads((src/'source_manifest.json').read_text())
for item in manifest['files']:
 p=src/item['local'] if 'local' in item else Path(item['source']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
seen=set();groups=defaultdict(list);maxima=defaultdict(list);costchecks=0
for file in sorted(src.glob('h[456]_*.json')):
 chunk=json.loads(file.read_text());assert chunk['complete'];H=chunk['H']
 for row in chunk['rows']:
  key=(H,row['index']);assert key not in seen;seen.add(key);assert all(row['parity'].values());r=row['native'];ref=reference[str(H)]['rows'][row['name']];assert r['value_hex']==float(ref['reference']['est']).hex();assert r['original_n']==r['current_n']-r['current_helper_X']==ref['reference']['ops']==row['original_counter'];assert row['truth']==ref['truth']
  for ledger in (r['ledger'],row['sampler']):
   c=Counter()
   for site in ledger['sites'].values():c.update(site)
   assert dict(c)==ledger['totals'];integer=sum(v for k,v in c.items() if k.startswith('integer_control_'));numeric=sum(c.values())-integer;assert integer==ledger['integer_control_total'] and numeric==ledger['named_numeric_total'];costchecks+=1
  sample=4*H*H+12*H;assert sample==row['sampler_total']==sum(row['sampler']['totals'].values());cost=sum(r['ledger']['totals'].values());assert cost==r['expression_total'];N=F(cost,sample);oldN=F(r['original_n'],3*H*(H+2));p=F(float(row['truth']));bias=F(float(r['value']))-p;var=p*(1-p)/N;ratio=bias*bias/var;mult=N/oldN
  for actual,saved in [(float(ratio),row['squared_bias_ratio']),(float(mult),row['equivalent_budget_multiplier']),(float(var),row['sampling_variance'])]:assert math.isclose(actual,saved,rel_tol=2e-12,abs_tol=1e-20)
  group='clean' if row['name'].startswith('seed') else row['name'].rsplit('_',1)[0];groups[(H,group)].append((bias*bias,var,p*(1-p)/oldN));maxima[H].append(float(ratio))
assert seen=={(h,i) for h in (4,5,6) for i in range(157)}
results=[]
for (H,name),values in sorted(groups.items()):
 sq=sum((x[0] for x in values),F(0));v=sum((x[1] for x in values),F(0));oldv=sum((x[2] for x in values),F(0));reported=summary['by_H'][str(H)]['groups'][name];ratio=float(sq/v);assert len(values)==reported['models'];assert math.isclose(ratio,reported['ratio_of_means'],rel_tol=2e-12,abs_tol=1e-20);assert math.isclose(float(sq/oldv),reported['old_ratio_of_means'],rel_tol=2e-12,abs_tol=1e-20);results.append(dict(H=H,group=name,models=len(values),ratio=ratio))
out=dict(models=len(seen),site_ledgers_checked=costchecks,source_manifest_files=len(manifest['files']),groups=results,group_count=len(results),maximum_individual_by_H={str(h):max(v) for h,v in maxima.items()},all_group_ratios_below_one=all(r['ratio']<1 for r in results),review_cpu=time.process_time()-start,scope='Existing fixed471 suite and saved floating numerical truths; explicit-source-expression units, not complete hardware/implementation cost, independent truth certificate, new MC or a general ARC claim.')
(here/'fibre_suite_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='groups'},indent=2))
