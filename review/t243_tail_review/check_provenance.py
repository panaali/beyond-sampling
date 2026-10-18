"""Check proof/predecessor text bindings only."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/workspace/work/recover-small_h2_56');BASE=ROOT/'review/conditional_residual_tail';OWN=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m=json.loads((BASE/'manifest.json').read_text())
    for path,h in m['pins'].items():assert sha(ROOT/path)==h
    for c in [m['claim'],'1106dbb4','77e5f473',*m['predecessor_reviews']]:
        subprocess.run(['git','merge-base','--is-ancestor',c,'origin/ccr-6a61f25c-wn80ii'],cwd=ROOT,check=True)
    pins={str(p):sha(p) for p in sorted(BASE.iterdir()) if p.is_file()}
    pins.update({str(ROOT/p):h for p,h in m['pins'].items()})
    result={'task':'T243','reviewer':'recover_34','complete':True,'pass_all':True,'execution_allowed':False,'execution_authorized':False,'author_commits':['1106dbb4','77e5f473'],'pins':pins,
        'checks':['oriented strict residual tail and zero atom','second-moment antiderivative and cubic bound','fixed-root small-band asymptotics','all-pairs deterministic margin coefficient','gap/residual union inclusion and K0','past-measurable thresholds','positive-diagonal invariance'],
        'scope':'written exact-real theorem and predecessor provenance only; no numerical author or Gaussian calls',
        'exclusions':['no target gap distribution','no full floating certificate','no covariance-only third moment','no execution or performance claim']}
    (OWN/'source_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':True,'pins':len(pins)}))
if __name__=='__main__':main()
