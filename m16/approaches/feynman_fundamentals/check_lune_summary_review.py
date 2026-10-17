import json,time,hashlib
from pathlib import Path
start=time.process_time()
p=Path('/workspace/work/recover-small_h4_4/m16/approaches/anisotropic_harmonic/lune_cycle/ensemble_results.json')
d=json.loads(p.read_text());out={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'families':[]}
for family in ('H2','H4'):
 rows=[r for r in d['rows'] if r['family']==family];n=len(rows)
 old=sum(r['hemisphere']['error']**2*r['hemisphere']['stats']['cpu_seconds'] for r in rows)/n
 new=sum(r['lune']['error']**2*r['lune']['stats']['cpu_seconds'] for r in rows)/n
 out['families'].append(dict(family=family,n=n,all_absolute_errors_improve=all(abs(r['lune']['error'])<abs(r['hemisphere']['error']) for r in rows),ratio_mean_error_squared_cpu=new/old,maximum_refinement=max(r['refinement']['change'] for r in rows),reference_error_consistency=max(abs(r['lune']['estimate']-r['reference']-r['lune']['error']) for r in rows)))
out['cpu_seconds']=time.process_time()-start
Path(__file__).with_name('lune_summary_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
