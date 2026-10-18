"""Written-review text binding only; no scientific function execution."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/workspace/work/recover-small_h2_78')
BASE=ROOT/'review/easy_correct_certificate';OWN=Path(__file__).resolve().parent
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    m=json.loads((BASE/'manifest.json').read_text());checked=[]
    for r in m['records']:
        p=Path(r['path']);b=p.read_bytes()
        assert len(b)==r['bytes'] and sha(b)==r['sha256']
        # Each local owner path is checked against the same shared snapshot.
        rel=p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.relative_to('/workspace/work/recover-small_h2_34')
        assert b==git('show',m['snapshot_shared_commit']+':'+str(rel))
        checked.append({'path':str(p),'sha256':sha(b),'bytes':len(b)})
    for name,h in m['snapshots'].items():assert sha((BASE/name).read_bytes())==h
    for c in ['f0e38a58','ba46c4f3','3776d1b3','dc183c14',*m['prior_reviews'].values()]:
        subprocess.run(['git','merge-base','--is-ancestor',c,'origin/ccr-6a61f25c-wn80ii'],cwd=ROOT,check=True)
    pins={str(p):sha(p.read_bytes()) for p in sorted(BASE.iterdir()) if p.is_file()}
    out={'task':'T240','reviewer':'recover_34','complete':True,'pass_all':True,'execution_allowed':False,'execution_authorized':False,
         'author_commits':['3776d1b3','dc183c14'],'pins':pins,'inherited_source_checks':checked,
         'checks':['ordered-gap probability identity','Cartesian containment and intersection','first-index correctness predicates','conditional and unresolved mass intervals','corrected true-event subset relation','restricted affine ranges and parent inheritance','prefix versus full-input correctness'],
         'preserved_issue':'3776d1b3 displayed lower bound was not a numeric upper ceiling; corrected dc183c14 without numerical calls',
         'exclusions':['no archive or checkpoint decode','no author calls','no actual mass/cost improvement','no inherited full recurrence re-audit','no future execution clearance']}
    (OWN/'source_review.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'complete':True,'source_texts':len(checked),'pins':len(pins)}))
if __name__=='__main__':main()
