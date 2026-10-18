"""T228 independent source/algebra and saved-checker gate; no archive decoding."""
from pathlib import Path
from fractions import Fraction as F
import ast,hashlib,itertools,json,subprocess,time
HERE=Path(__file__).resolve().parent
OWNER=Path('/workspace/work/recover-small_h2_34')
BASE=OWNER/'review/width16_structure_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.process_time();out=HERE/'t228_source_review.json';assert not out.exists()
 m=json.loads((BASE/'manifest.json').read_text());assert len(m['files'])==37
 checked=[]
 for r in m['files']:
  b=(BASE/r['snapshot']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
  assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==r['git_blob']
  assert subprocess.check_output(['git','-C',str(OWNER),'show',m['author_revision']+':'+r['path']])==b
  checked.append(dict(path=r['path'],sha256=r['sha256'],bytes=len(b),git_blob=r['git_blob']))
 # Exact algebraic counterexamples, not neural inputs or author code.
 cycle=((0,1,0),(0,0,1),(1,0,0))
 def relabel(p):return tuple(tuple(cycle[p[i]][p[j]] for j in range(3)) for i in range(3))
 assert relabel((1,2,0))==cycle
 for i,j in itertools.combinations(range(3),2):
  p=list(range(3));p[i],p[j]=p[j],p[i];assert relabel(p)!=cycle
 def cos2(a,b):return F(sum(x*y for x,y in zip(a,b))**2,sum(x*x for x in a)*sum(y*y for y in b))
 assert cos2((1,1,0),(1,0,1))==F(1,4)
 assert cos2((2,1,0),(1,0,1))==F(2,5)
 protocol=json.loads((BASE/'protocol.json').read_text())
 for rel,digest in protocol['pins'].items():assert sha(BASE/rel)==digest,rel
 tree=ast.parse((BASE/'audit.py').read_text());imports=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):imports.extend(a.name.split('.')[0] for a in n.names)
  if isinstance(n,ast.ImportFrom):imports.append(n.module.split('.')[0])
 assert set(imports)<={'pathlib','fractions','collections','hashlib','itertools','json','math','os','resource','subprocess','sys','time','traceback','numpy'}
 assert protocol['expected_arrays']==9 and protocol['expected_progress']==22 and F(protocol['absolute_comparison_tolerance'])==F(1,2000000000)
 r=dict(complete=True,pass_all=True,task='T228 independent source/provenance/algebra and fixed saved-checker gate',delegation='37d8695d',source_notes_commit='9d480f7b',pins={str(BASE/n):sha(BASE/n) for n in ['manifest.json','SOURCE_NOTES.md','README.md','audit.py','PROTOCOL.md','protocol.json']},payloads=checked,algebra=dict(three_cycle_preserved=True,all_three_transpositions_rejected=True,gauge_original_cosine_squared='1/4',gauge_transformed_cosine_squared='2/5'),findings=['The pair scanner does not exhaust all permutation cycles. A three-cycle recurrent matrix with constant input/readout columns has a cyclic automorphism and no transposition automorphism.','Positive gauge D transforms Wi,Wh,Wo as DWi,DWhD^-1,WoD^-1; Euclidean augmented-row cosines are not invariant. Exact collinearity and near-collinearity scores are distinct.','SVD row projection keeps Wo but changes entries of recurrent outgoing columns; this intervention is not a function-preserving quotient. Zero forms have both ReLUs OFF.','Near/top-five summaries do not reconstruct all120 scan rows; norm-based dead tests do not prove reachable inactivity. Missing variant driver/raw outcomes/checkpoint hashes remain missing.','Floating exact-mode geometry and fixed numerical cone quadrature do not certify individual leaves or masses. Independent T219 target counts do not transfer those certificates.','For nonnegative normalized joint masses, sum_e max_g p(e,g) is the optimal entire-history classification success from event e. It is not output prediction accuracy or an estimator lower bound.','Entropy effective counts and event-weighted99% covers are not worst-event or computational lower bounds.','Missing SX and id-to-history arrays prevent reconstructing the sign0 variant; numeric IDs do not identify signs.','Checker uses exact stored-float unsigned mass grouping and cover thresholds, independent entropy decomposition and iid strict-order permutation marginals. Signed/nonfinite/malformed outcomes remain explicit; signed masses invalidate probability interpretation.','Paired MC summary mean/second-moment constraints are arithmetic consistency only, not reconstruction of outcomes or confidence coverage.'],approved_scope=dict(arrays=9,prefix_summaries=6,symmetry_summaries=3,comparison_absolute_tolerance='1/2000000000',attempts=1,internal_cpu_seconds=17,hard_cpu_seconds=20,address_space_bytes=2147483648,threads=1,review_lock=True),gate_requirement='Publish immutable gate explicitly pinning completed source/provenance and checker review roles (this combined artifact may fill both), checker/protocol/manifest/PROTOCOL and all payload pins. One saved-only run; independent result review follows.',limitations=['No author import, archive decode, model/geometry/mass/RNG call in this review.','Snapshot identity is not historical execution/model identity.','No universal symmetry exclusion, exact probability certificate, full-cost result or family impossibility.','Fixed5e-10 tolerance is saved arithmetic consistency, not integration error or CI.'],checker_sha256=sha(__file__),review_cpu_seconds=time.process_time()-start)
 out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(complete=True,pass_all=True,sha256=sha(out),cpu=r['review_cpu_seconds'])))
if __name__=='__main__':main()
