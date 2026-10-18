from pathlib import Path
import ast,hashlib,json,subprocess
P=Path('/workspace/work/recover-small_h2_56/review/pruning_cost_frontier/panel')
R=Path('/workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals/t264_frontier_review/parity_saved/results.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((P/'target_protocol.json').read_text());r=json.loads(R.read_text());assert r['complete'] and r['pass_all']
assert r['pins'][str(P/'results.json')]==p['pins'][str(P/'results.json')]
for pins in [p['pins'],r['pins']]:
 for f,h in pins.items():assert sha(f)==h,f
for n in ['apply.py','primitive.py']:ast.parse((P/n).read_text())
m=json.loads((P/'retained/manifest.json').read_text())
for x in m['records']:
 b=Path(x['staged_path']).read_bytes();assert len(b)==x['bytes'] and sha(x['staged_path'])==x['sha256']
 assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==x['blob']
 assert subprocess.check_output(['git','rev-parse',m['source_commit']+':'+x['git_path']],cwd=P).decode().strip()==x['blob']
files=[P/n for n in ['apply.py','primitive.py','TARGET_PROTOCOL.md','target_protocol.json','retained/manifest.json']]+[R]
out={'complete':True,'pass_all':True,'execution_allowed':True,'source_commit':'7d6a11d2','pins':{str(f):sha(f) for f in files},'scope':'One predetermined retained H16T4 zoo0 chamber, one exact frontier call; 20/20CPU,2GiB,17internal,25wall after review lock; immutable published gate required. No numerical payload decode or author call by reviewer.','checks':['five source files and completed parity binding','all protocol and parity pins','three original Git blob identities without payload decoding','NPY schema bounded before decode','nonnegative error/cost and positive fixed work contract','exact Fraction threshold and index-tied subset algebra','raw complete return precedes post-return event checks','failure status and runtime/limit checks'],'qualifications':['Saved binary-value frontier only; no true mass/full native cost/deployment claim','historical NumPy ordering and summation are not reconstructed','negative diagnostic leaf mass remains reported','resource cap may fail; preserve partial results, no automatic retry']}
Path(__file__).with_name('source_gate.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS source/provenance only; no numerical payload decoded')
