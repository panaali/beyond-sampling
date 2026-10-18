"""Read-only text/Git provenance check; never import author modules."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/workspace/work/recover-small_h2_56')
OWN=Path(__file__).resolve().parent
BASE=ROOT/'review/correct_error_complexity_contract'
def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    m=json.loads((BASE/'manifest.json').read_text())
    checks=[]
    for f in m['files']:
        local=(ROOT/f['local']).read_bytes()
        remote=git('show',f["head"]+':'+f['source_path'])
        blob=git('rev-parse',f["head"]+':'+f['source_path']).decode().strip()
        assert local==remote and sha(local)==f['sha256'] and blob==f['blob']
        checks.append({'path':f['local'],'sha256':sha(local),'blob':blob,'bytes':len(local)})
    for path,h in m['pins'].items():
        assert sha((ROOT/path).read_bytes())==h
    for name,folder in [('width','width16_probe'),('gpu','gpu_cost_law')]:
        assert git('ls-tree','-r','--name-only',m['heads'][name],'m16/approaches/'+folder)==(BASE/(name+'_tree.txt')).read_bytes()
    directive=(BASE/'T239_DIRECTIVE.txt').read_text().strip()
    assert directive in git('show','9628bc97:TASKS.md').decode().splitlines()
    for commit in ['c338207e','991a5a66','86de6778']:
        subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/ccr-6a61f25c-wn80ii'],cwd=ROOT,check=True)
    pins={str(p):sha(p.read_bytes()) for p in sorted(BASE.rglob('*')) if p.is_file()}
    # Additional read-only collector binding: this is not execution/geometry validation.
    extra={}
    for name in ['estG.py','estT_base.py']:
        path='m16/approaches/gpu_cost_law/'+name
        b=git('show',m['heads']['gpu']+':'+path)
        extra[path]={'head':m['heads']['gpu'],'sha256':sha(b),'bytes':len(b)}
    out={'task':'T242','reviewer':'recover_34','complete':True,'pass_all':True,
         'execution_authorized':False,'execution_allowed':False,
         'author_commits':['c338207e','991a5a66','86de6778'],
         'review_scope':'written source/provenance and mathematical contract only',
         'pins':pins,'source_checks':checks,'additional_text_bindings':extra,
         'checks':['mixed activation-leaf versus output-region schema','conditional coverage and positive denominator','fixed-subset interval ratios and uncertain ordering','subdivision counterexample','gap intersection and blind law','selection and probabilistic joint coverage'],
         'exclusions':['no author imports or calls','no archive/checkpoint decode','no numerical mass or geometry validation','no current/future T239 result validation','no experiment execution clearance']}
    (OWN/'source_review.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'complete':True,'source_blobs':len(checks),'pins':len(pins)}))
if __name__=='__main__':main()
