from pathlib import Path
import json,time
from check_time_shift_cost_saved import P,O,sha,audit
start=time.process_time();p=json.loads((P/'protocol.json').read_text());g=json.loads((P/'run_scope.json').read_text());r=json.loads((P/'results.json').read_text())
assert r['complete'] and r['new_model_geometry_mass_MC_calls']==0 and r['protocol_sha256']==sha(P/'protocol.json') and r['run_scope_sha256']==sha(P/'run_scope.json')
for path,h in p['pins'].items():assert sha(Path(path))==h
assert g['controls_sha256']==sha(P/'controls.json') and g['controls_review'] and g['source_gate']
a=json.loads(Path(p['attribution']).read_text());assert a['complete'] and a['source_sha256']==sha(Path(p['geometry'])) and a['checkpoint_sha256']==r['checkpoint_sha256']==p['checkpoint_sha256']
words=a['sets']['gate']['history_keys'];assert set(words)==set(a['histories']) and len(set(words))==450
c=r['census'];assert c['input_rows']==len(words) and c['unique_histories']==sorted(set(words));audit(c)
result=dict(complete=True,input_rows=len(words),unique_histories=450,prefix_counts=[len(x) for x in c['parent_prefixes']],baseline=sum(c['baseline'].values()),cuts=[dict(cut=x['cut'],warm=sum(x['warm'].values()),cold=sum(x['cold'].values()),combined=sum(x['combined'].values()),named_saving=x['named_saving']) for x in c['cuts']],author_sha256=sha(P/'results.json'),author_cpu=r['cpu_seconds'],review_cpu=time.process_time()-start,scope='Independent saved history/prefix/suffix identities and closed-form category contractions. No trained matrix/cache or geometry execution. No strict cold dense arithmetic saving here; sparse/reset/other reuse and actual runtime unresolved.')
(O/'time_shift_census_saved_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
