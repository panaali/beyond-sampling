"""Source/resource gate only; does not execute the proposed contraction."""
from pathlib import Path
import ast,hashlib,json,time
HERE=Path(__file__).resolve().parent
BASE=Path('/workspace/work/recover-small_h2_78/review/tie_tail_evidence')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.process_time();out=HERE/'t227_checker_source_review.json';assert not out.exists()
 protocol=json.loads((BASE/'protocol.json').read_text())
 assert len(protocol['pins'])==31
 for path,digest in protocol['pins'].items():assert sha(path)==digest,path
 tree=ast.parse((BASE/'check_saved.py').read_text());imports=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):imports.extend(a.name.split('.')[0] for a in n.names)
  if isinstance(n,ast.ImportFrom):imports.append(n.module.split('.')[0])
 assert set(imports)<={'pathlib','fractions','collections','math','hashlib','json','time','resource','traceback','sys'}
 prior=json.loads((HERE/'t227_source_review.json').read_text())
 assert prior['complete'] and prior['pass_all'] and prior['manifest_sha256']==sha(BASE/'source_manifest.json')
 assert protocol['models']==['z6s0','z6s1','z6s2','z6s3','z6s4','orig','s005','s010']
 assert protocol['schemes']==['mech','pos','pos_flips','nopos','relpos','coarse'] and protocol['L']==[10,30,100]
 r=dict(complete=True,pass_all=True,task='T227 fixed saved contraction source/resource gate',source_commit='fe130a60',initial_source='e7b2b358',pins={str(BASE/n):sha(BASE/n) for n in ['check_saved.py','protocol.json','PROTOCOL.md','launcher.sh','source_manifest.json']},prior_source_review_sha256=sha(HERE/'t227_source_review.json'),checker_sha256=sha(__file__),approved_scope=dict(models=8,rules_per_model=2,tail_rows_per_rule=1639,schemes=6,L=[10,30,100],within_statistics=48,cross_copy_scores=36,author_calls=0,attempts=1,internal_cpu_seconds=17,soft_cpu_seconds=19,hard_cpu_seconds=20,address_space_bytes=2147483648,review_lock=True),findings=['Exact Fraction of each stored finite binary64 value is used for group sums, deterministic selections, ratios and prefix-summary arithmetic.','Stable exact ties use first encounter order, matching the source dictionary/order convention; any disagreement caused by historical rounding must fail, not induce retuning.','Top1 uses floor(.01*32768)=327 and top5 ceil(.05*32768)=1639.','All retained and missing-kind point counts/mass, duplicate/ordered flip diagnostics and fixed six schemes are retained.','Within-label statistics remain conditioned on the selected global tail.','Fixed absolute 5e-12 compares saved display/summation consistency only; no integration or model error bound.','Half-tail and random-label assignments are absent and not regenerated; complete all-face masses and labels/walks are not independently certified.','Atomic partial result and progress preserve failure; immutable audit directory, launcher log and completed/pinned prerequisite checks precede the sole run.'],required_gate='Publish immutable gate with both complete/pass_all source review artifacts and mandatory checker/protocol/launcher/manifest hashes before execution.',limitations=['Finite-rule kill is the original orig/s005/s010 mech L100 threshold only.','No population CI, model identity, exact walk/LP, complete-cost or method-family impossibility claim.','Saved results require independent review after completion; source clearance is not a result.'],review_cpu_seconds=time.process_time()-start)
 out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(complete=True,pass_all=True,sha256=sha(out),cpu=r['review_cpu_seconds'])))
if __name__=='__main__':main()
