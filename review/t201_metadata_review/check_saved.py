"""T201 complementary file/source metadata only; no author module imports."""
import ast,hashlib,json,subprocess,time
from fractions import Fraction as F
from pathlib import Path
HERE=Path(__file__).resolve().parent
A=Path('/workspace/work/recover-small_h2_78/review/mechanism_claims_review')
REPO=A.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO)
def main():
 start=time.process_time();out=HERE/'results.json';assert not out.exists()
 pins=json.loads((A/'pins.json').read_text());assert pins['snapshot_commit']=='cf62f4ab' and len(pins['files'])==13
 checked=[];trees={}
 for name,pin in pins['files'].items():
  path=A/'frozen'/name;raw=path.read_bytes();assert sha(path)==pin['sha256']
  blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest();assert blob==pin['git_blob']
  assert git('show',pins['snapshot_commit']+':'+name)==raw
  checked.append(dict(path=name,sha256=pin['sha256'],git_blob=blob))
  if name.endswith('.py'):trees[name]=ast.parse(raw)
 functions={name:[x.name for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))] for name,tree in trees.items()}
 assert functions['m16/state.py']==['discrete_state']
 assert functions['m16/separable.py']==['design','fit']
 assert functions['m16/features.py']==['loo_max','prefix_max','dictionary']
 assert functions['m16/frozen.py']==['gates','run_frozen','canonical']
 assert functions['m16/exact.py']==['contribution_map','logit_map']
 for name in ['m16/state.py','m16/separable.py','m16/features.py','m16/frozen.py','m16/exact.py']:
  assert all(isinstance(x,(ast.Import,ast.ImportFrom,ast.FunctionDef)) or (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str)) for x in trees[name].body)
 st=ast.unparse(trees['m16/state.py']);assert 'key=(case * 9 + p1) * 9 + p2' in st
 sep=ast.unparse(trees['m16/separable.py']);assert 'np.linalg.lstsq(X[mask], h[mask], rcond=None)[0]' in sep
 assert 'np.eye(9)[s[posvar]]' in sep and 'np.concatenate(cols, 1)' in sep
 co=ast.unparse(trees['m16/common.py']);assert '_m = example_2nd_argmax().eval()' in co and 'np.argsort(x, 1)[:, -2]' in co
 for name in ('m16/estimate.py','m16/arc_eval.py'):
  tree=trees[name];refs=[n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ref' for t in n.targets)]
  assert len(refs)==1 and 'standard_normal((400000, T))' in ast.unparse(refs[0])
 origins={c:git('diff-tree','--no-commit-id','--name-only','-r',c).decode().splitlines() for c in pins['history'].values()}
 assert origins['6cc72d88']==['m16/frozen.py','m16/state.py']
 assert origins['0c2ec2e1']==['m16/exact.py','m16/modes.py','m16/separable.py']
 assert origins['dd8b65b9']==['m16/FINDINGS.md']
 raw=json.loads((A/'frozen/results/algzoo_16_10.json').read_text());assert raw['replacement_acc']==.54133 and raw['model_acc']==.9535 and raw['n_neurons']==16
 findings=(A/'frozen/m16/FINDINGS.md').read_text();results=(A/'frozen/RESULTS.md').read_text()
 for text in ['0.83%','2.17%','0.52%','0.96%','4.69%','90-91%']:assert text in findings
 for text in ['0.82%','2.14%','0.52%','0.27%','98%']:assert text in results
 # Exact decimals contract prose only; no empirical denominator is inferred.
 finding_total=sum(map(F,['0.83','2.17','0.52','0.96','0.2']));result_total=sum(map(F,['0.82','2.14','0.52','0.27']))
 assert finding_total==F('4.68') and result_total==F('3.75') and F('4.69')-result_total==F('.94')
 result=dict(complete=True,assignment='754e7160',snapshot_commit=pins['snapshot_commit'],checked_snapshots=checked,function_inventory=functions,origin_changed_files=origins,prose_sums_percent=[str(finding_total),str(result_total)],prose_gap_percent=str(F('4.69')-result_total),pins={str(p):sha(p) for p in [A/'pins.json',A/'SOURCE_INVENTORY.md',A/'README.md',HERE/'check_saved.py']},cpu_seconds=time.process_time()-start,author_imports=0,model_fit_random_truth_calls=0,scope='Thirteen snapshot/helper/cited-origin metadata and rounded prose only. No global absence assertion, fit/test validation, blind/crossing theorem review, numeric reconstruction or replay gate.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,snapshots=13,cpu=result['cpu_seconds'],author_imports=0)))
if __name__=='__main__':main()
