"""Text-only Git/SHA binding; no author imports or archive decode."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/workspace/work/recover-small_h2_910')
BASE=ROOT/'review/neuron_account_contract'
OWN=Path(__file__).resolve().parent
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    m=json.loads((BASE/'manifest.json').read_text()); checks=[]
    for r in m['records']:
        b=(BASE/r['snapshot']).read_bytes()
        assert b==git('show',r['source_commit']+':'+r['source_path'])
        assert len(b)==r['bytes'] and sha(b)==r['sha256']
        assert git('rev-parse',r['source_commit']+':'+r['source_path']).decode().strip()==r['git_blob']
        checks.append({'path':r['snapshot'],'bytes':len(b),'sha256':sha(b),'blob':r['git_blob']})
    tree=git('ls-tree','-r',m['shared_snapshot'],m['author_tree_path'])
    assert tree==(BASE/'AUTHOR_TREE.txt').read_bytes()==b''
    assert sha(tree)==m['author_tree_listing_sha256']
    directive=(BASE/'T241_DIRECTIVE.md').read_bytes()
    assert sha(directive)==m['directive_sha256']
    assert directive.decode().strip() in git('show',m['shared_snapshot']+':TASKS.md').decode().splitlines()
    for commit in [m['claim'],'19549761']:
        subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/ccr-6a61f25c-wn80ii'],cwd=ROOT,check=True)
    pins={str(p):sha(p.read_bytes()) for p in sorted(BASE.rglob('*')) if p.is_file()}
    out={'task':'T244','reviewer':'recover_34','complete':True,'pass_all':True,'execution_allowed':False,'execution_authorized':False,
         'author_commit':'19549761','pins':pins,'source_checks':checks,
         'checks':['five-way outcome partition','accuracy/agreement inequality','paired interval union bound','intervention/donor/rollout semantics','mode metric and conjugate reconstruction','source indexing and retrospective features','selection/case/dependence qualifications'],
         'exclusions':['no author imports or calls','no array/checkpoint decode','no new T241 results','no universal causal-role or capacity inference','no execution clearance']}
    (OWN/'source_review.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'complete':True,'text_payloads':len(checks),'pins':len(pins)}))
if __name__=='__main__':main()
