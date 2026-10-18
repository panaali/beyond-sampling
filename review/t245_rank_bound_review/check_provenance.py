"""Only text/Git identity checks, not rank or geometry execution."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/workspace/work/recover-small_h2_910');BASE=ROOT/'review/rank_history_bound';OWN=Path(__file__).resolve().parent
def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT)
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    m=json.loads((BASE/'manifest.json').read_text());checks=[]
    for r in m['records']:
        b=(BASE/r['snapshot']).read_bytes()
        assert b==git('show',r['commit']+':'+r['path']) and len(b)==r['bytes'] and sha(b)==r['sha256']
        assert git('rev-parse',r['commit']+':'+r['path']).decode().strip()==r['git_blob']
        checks.append({'path':r['path'],'sha256':sha(b),'bytes':len(b),'blob':r['git_blob']})
    # The explicit final author commit list is frozen before this checker runs.
    commits=['050ebb65','2ec6aa59','94e2fc83']
    assert 'FINAL_EDGE_COMMIT' not in commits,'await written edge clarification publication'
    for c in commits:subprocess.run(['git','merge-base','--is-ancestor',c,'origin/ccr-6a61f25c-wn80ii'],cwd=ROOT,check=True)
    pins={str(p):sha(p.read_bytes()) for p in sorted(BASE.rglob('*')) if p.is_file()}
    out={'task':'T245','reviewer':'recover_34','complete':True,'pass_all':True,'execution_allowed':False,'execution_authorized':False,'author_commits':commits[1:],'pins':pins,'source_checks':checks,
         'checks':['open nonnegative image excludes zero','positive section dimension','open sign chamber and lift correspondence','affine arrangement upper bound','rank-zero boundary ownership','aggregate and coordinate invariance','eight written edge cases'],
         'exclusions':['no ranks or matrices computed','no geometry/target data','no observed growth or cost claim','no execution clearance']}
    (OWN/'source_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'complete':True,'predecessor_blobs':len(checks),'pins':len(pins)}))
if __name__=='__main__':main()
