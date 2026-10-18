"""Text/hash/Git inventory only; no author imports, arrays or model calls."""
from pathlib import Path
import hashlib,json,subprocess
B=Path('/workspace/work/recover-small_h2_56/review/capacity_study_contract');O=Path(__file__).with_name('t237_capacity_contract_review.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)
def main():
 assert not O.exists();m=json.loads((B/'manifest.json').read_text());repo=B.parents[1];deps=Path('/workspace/dependencies/alg-zoo')
 assert len(m['files'])==3 and m['capacity_tree_entries']==[]
 for f in m['files']:
  local=repo/f['local'];original=Path(f['source']);rel=original.relative_to(deps).as_posix()
  assert sha(local)==sha(original)==f['sha256']
  blob=git(deps,'rev-parse',m['installed_alg_zoo_head']+':'+rel).decode().strip();assert blob==f['git_blob']
  assert git(deps,'cat-file','blob',blob)==local.read_bytes()
 assert git(deps,'status','--porcelain')==b'' and git(deps,'rev-parse','HEAD').decode().strip()==m['installed_alg_zoo_head']
 assert git(repo,'ls-tree','-r','--name-only',m['author_head'],'--','m16/approaches/capacity_ceiling')==b''
 assert sha(B/'T234_DIRECTIVE.txt')==m['directive_sha256']
 tasktext=git(repo,'show',m['shared_head']+':TASKS.md').decode();directive=(B/'T234_DIRECTIVE.txt').read_text().strip();assert directive in tasktext
 stats=(B/'STATISTICAL_CONTRACT.md').read_text();assert 'deterioration' in stats and 'two-sided' in stats
 names=['README.md','SOURCE_REVIEW.md','STATISTICAL_CONTRACT.md','manifest.json','T234_DIRECTIVE.txt','capacity_tree_inventory.txt','snapshot/training.py','snapshot/architectures.py','snapshot/tasks.py']
 result={'complete':True,'pass_all':True,'task':'T237 independent written source/theorem contract review','claim':'e823d1d3','pins':{str(B/n):sha(B/n) for n in names},'source_snapshots_verified':3,'directive_verified':True,'empty_capacity_tree_at':m['author_head'],'local_alg_zoo_head':m['installed_alg_zoo_head'],'checker_sha256':sha(Path(__file__)),'written_review_sha256':sha(Path(__file__).with_name('T237_CAPACITY_CONTRACT_REVIEW.md')),'new_import_checkpoint_array_training_model_rng_mc_calls':0,'execution_authorized':False,'scope':'Source/provenance and written statistical implications only; T234 owner authorization and plan unchanged.','qualifications':['Three manifest source entries plus separately bound directive; initial seven-bindings caption corrected.','Longer default n_train changes relative-horizon LR schedule; weights-only state_dict is not exact optimizer continuation.','Best-found model confidence bounds do not upper-bound architecture capacity; MC, training seeds and selection uncertainty distinct.','One-sided no-material-gain test is not two-sided equivalence; corrected contract checked.','Future Colab driver/runtime and study outcomes are unverified; absence claim limited to frozen Git tree.']}
 O.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'complete':True,'sha256':sha(O)}))
if __name__=='__main__':main()
