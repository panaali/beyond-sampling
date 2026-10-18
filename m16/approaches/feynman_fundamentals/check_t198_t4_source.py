"""Incremental T4 saved-checker source gate: metadata/AST only, no row contraction."""
from pathlib import Path
import ast,hashlib,json,subprocess,time
HERE=Path(__file__).resolve().parent
OWNER=Path('/workspace/work/recover-small_h4_4')
BASE=OWNER/'review/split_attribution_review/t4_increment'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.process_time();out=HERE/'t198_t4_source_review.json';assert not out.exists()
 m=json.loads((BASE/'manifest.json').read_text());assert len(m['records'])==22
 payloads=[]
 for r in m['records']:
  data=(BASE/'frozen'/r['original']).read_bytes()
  assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
  assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==r['git_blob']
  assert subprocess.check_output(['git','-C',str(OWNER),'show',m['author_commit']+':'+r['original']])==data
  payloads.append(dict(path=r['original'],sha256=r['sha256'],bytes=len(data)))
 unchanged=[]
 for r in m['unchanged_from_completed_phase']:
  a=subprocess.check_output(['git','-C',str(OWNER),'rev-parse',m['author_commit']+':'+r['path']]).decode().strip()
  b=subprocess.check_output(['git','-C',str(OWNER),'rev-parse',m['previous_author']+':'+r['path']]).decode().strip()
  assert a==b==r['git_blob'];unchanged.append(r['path'])
 assert len(m['new_T4_paths'])==15 and not m['T5_prune_paths'] and m['rendered_prune_report_present'] is False
 source=(BASE/'check_saved.py').read_text();tree=ast.parse(source)
 imports=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):imports.extend(a.name.split('.')[0] for a in node.names)
  if isinstance(node,ast.ImportFrom):imports.append(node.module.split('.')[0])
 assert set(imports)<={'pathlib','fractions','hashlib','json','math','time','traceback'}
 prior=HERE/'revised_cap_saved_review.json';review=json.loads(prior.read_text());assert review['complete'] and review['pass_all']
 r=dict(complete=True,pass_all=True,task='T198 incremental T4 saved checker source gate',source_commit='57018bce',initial_source_commit='01533562',delegation='52f06ead',pins={str(BASE/n):sha(BASE/n) for n in ['check_saved.py','PROTOCOL.md','manifest.json','README.md']},prior_review_sha256=sha(prior),payloads=payloads,unchanged_source_and_T3_paths=unchanged,approved_scope=dict(new_T4_rows=15,partial_groups=10,prior_T3_rows_reused=15,author_calls=0,max_process_cpu_seconds=20,address_space_bytes=2147483648,attempts=1,review_lock=True),required_execution_gate='Bind this completed review, all current checker/protocol/manifest files, completed prior T3 result and its independent review. Publish gate before the sole immutable run.',findings=['Fraction ratios use the exact stored binary64 reference, continuous FLOPs/682 and p(1-p)/draws; reported bound-squared ratio is separately retained.','Historical FLOPs/(2000/3) residual is only a source-allowance consistency diagnostic.','Seed0 ten rows plus seed1 five rows are explicitly fixed; missing seeds and thresholds remain missing.','Cell growth uses matching seed subsets, not an incomplete T4 mean versus all-five T3 mean.','Prune count conservation and padded step residuals do not establish outward masses or total estimator error.','Immutable outputs, preserved failures, stdlib-only reads and source prerequisites checked; external process limit remains required.'],limitations=['No universal cap, exact truth, CI, integer-budget, full-cost or rung adoption claim.','T5 attribution archive metadata does not authorize a load or pruning replay.','No rendered report exists for text-parity checks.','Completed saved output needs a separate independent audit.'],review_cpu_seconds=time.process_time()-start,checker_sha256=sha(__file__))
 out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(complete=True,pass_all=True,sha256=sha(out),cpu=r['review_cpu_seconds'])))
if __name__=='__main__':main()
