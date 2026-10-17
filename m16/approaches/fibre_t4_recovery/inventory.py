"""Recover existing artifacts and source history, without estimator execution."""
from pathlib import Path
import hashlib,json,subprocess,time
HERE=Path(__file__).resolve().parent
REPO=Path('/workspace/work/admin-recover-h4-t67');SOURCE=REPO/'m16/approaches/understand_ladder';start=time.process_time()
names=['README.md','run_t4.py','summary_t4.py','fibre4.py','fibre4b.py','fiber3.py','zoo.py','check4.py','check4_exact.py','results_H8_T4.json','results_H4_T4.json','run_t4_H8_noisy.out','run_t4_H8_seeds.out','run_t4_F2L3.out']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[]
for name in names:
 p=SOURCE/name;rel=str(p.relative_to(REPO));hist=subprocess.check_output(['git','log','-5','--format=%H %s','--',rel],cwd=REPO,text=True).splitlines()
 files.append(dict(name=name,path=str(p),sha256=sha(p),bytes=p.stat().st_size,history=hist))
variants={}
paths=list(Path('/workspace/work').glob('*/m16/approaches/understand_ladder/results_H8_T4.json'))+[Path('/workspace/clean-neurons/m16/approaches/understand_ladder/results_H8_T4.json')]
for p in paths:
 if not p.exists():continue
 h=sha(p);obj=json.loads(p.read_text());rows=obj.get('rows',{})
 item=variants.setdefault(h,dict(sha256=h,bytes=p.stat().st_size,rows=list(rows),estimated=[k for k,v in rows.items() if 'F2L3' in v],paths=[]));item['paths'].append(str(p))
last_result=next(x for x in files if x['name']=='results_H8_T4.json')['history'][0].split()[0]
contemporaneous=[]
for name in ['fibre4.py','fibre4b.py','run_t4.py','zoo.py','fiber3.py']:
 rel='m16/approaches/understand_ladder/'+name;b=subprocess.check_output(['git','show',last_result+':'+rel],cwd=REPO)
 contemporaneous.append(dict(name=name,last_result_commit=last_result,snapshot_sha256=hashlib.sha256(b).hexdigest(),matches_current=hashlib.sha256(b).hexdigest()==sha(SOURCE/name)))
refs=[]
for path in ['/workspace/work/recover-small_h2_910/m16/approaches/plackett_rule_doubling/summary.json','/workspace/work/recover-small_h2_910/m16/approaches/frozen_recipe_holdout/summary.json']:
 p=Path(path);refs.append(dict(path=path,sha256=sha(p),bytes=p.stat().st_size))
result=dict(source_root=str(SOURCE),repo_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),files=files,result_variants=list(variants.values()),last_result_snapshot=contemporaneous,references=refs,cpu_seconds=time.process_time()-start,no_estimator_executed=True)
(HERE/'inventory.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(files=len(files),result_variants=[dict(rows=len(x['rows']),estimated=len(x['estimated']),copies=len(x['paths'])) for x in variants.values()],source_snapshot=contemporaneous,cpu=result['cpu_seconds']),indent=2))
