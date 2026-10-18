"""Read-only ordered Git patch and scoped tree conservation."""
from pathlib import Path
import json,subprocess,hashlib,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
O=Path(__file__).resolve().parent;repo='/workspace/work/integrate-reviewed-78'
def git(*args):return subprocess.check_output(['git','-C',repo,*args])
m=json.loads(Path('/tmp/recover78_integration_manifest.json').read_text());claimed=json.loads(Path('/tmp/recover78_integration_check.json').read_text());base='16be5ad4';target=claimed['target_head'];source=m['head']
new=git('rev-list','--reverse',base+'..'+target).decode().splitlines();assert len(new)==len(m['commits'])==193
folders=claimed['folders'];paths=set();mapping=[]
for old,fresh in zip(m['commits'],new,strict=True):
 oldpaths=git('diff-tree','--no-commit-id','--name-only','-r',old).decode().splitlines();newpaths=git('diff-tree','--no-commit-id','--name-only','-r',fresh).decode().splitlines();assert oldpaths==newpaths
 assert all(any(p.startswith(f+'/') for f in folders) for p in oldpaths);paths.update(oldpaths)
 a=git('diff',old+'^',old,'--binary','--no-ext-diff');b=git('diff',fresh+'^',fresh,'--binary','--no-ext-diff');assert a==b,(old,fresh)
 mapping.append({'source':old,'target':fresh,'patch_sha256':hashlib.sha256(a).hexdigest()})
for f in folders:assert git('ls-tree','-r',source,'--',f)==git('ls-tree','-r',target,'--',f),f
assert not git('diff',base,target,'--','.',*[f':(exclude){f}' for f in folders])
assert not git('status','--porcelain')
out={'pass':True,'source_head':source,'target_head':target,'base':git('rev-parse',base).decode().strip(),'commits':193,'touched_files':len(paths),'folders':folders,'mapping':mapping,'review_cpu':time.process_time()-start,'scope':'Git ordered patch bytes, scoped path/tree conservation and clean worktree only; no scientific execution or claim reclassification. Future rebase/push must preserve these trees and latest remote shared changes.'}
p=O/'recover78_integration_review_results.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('mapping','folders')}))
