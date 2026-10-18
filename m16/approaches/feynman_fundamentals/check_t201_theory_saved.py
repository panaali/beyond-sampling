"""Check archived T138 premises and sign-count algebra, without model execution."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,subprocess,time
H=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_78/review/mechanism_claims_review')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 t=time.process_time();pins=json.loads((P/'t138_pins.json').read_text())
 for original,row in pins['entries'].items():
  snapshot=P/'prior_t138'/Path(original).name
  assert sha(snapshot)==row['sha256']
  archived=subprocess.check_output(['git','show',row['commit']+':'+original],cwd=P)
  assert archived==snapshot.read_bytes()
 c=json.loads((P/'prior_t138/certificate.json').read_text());m=json.loads((P/'prior_t138/margin_results.json').read_text());review=json.loads((P/'prior_t138/gate_symbol_review_results.json').read_text())
 assert c['model_sha256']=='498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'
 cert=c['certificate'];q=list(map(F,m['crossing_input']));v=list(map(F,m['margin_coefficients']));lam=F(m['crossing_parameter'])
 assert len(q)==len(v)==10 and 0<lam<1 and v[1]>0 and q[9]<0
 a,b=[list(map(F,r['x'])) for r in cert['pair']]
 assert q==[(1-lam)*x+lam*y for x,y in zip(a,b,strict=True)] and sum(x*y for x,y in zip(q,v,strict=True))==0
 assert sorted(range(10),key=lambda i:q[i],reverse=True)==cert['rank_order']==[0,3,6,9,8,7,5,4,2,1]
 assert len(set(q))==10 and [q[i] for i in [0,3,6]]==[F(1),F(63,64),F(31,32)]
 assert m['truth']==3 and m['competitor']==6 and F(m['other_output_min_margin'])>0 and m['gate_constraints']==160 and m['rank_constraints']==9
 assert review['reverse_adjoint_coefficients_match'] and review['crossing_parameter']==float(lam) and review['other_output_min_margin']==float(F(m['other_output_min_margin']))
 assert all(r['gate_constraints']==160 and F(r['gate_abs_min'])>0 and F(r['rank_min'])>0 for r in cert['pair'])
 blind=F(1,1024)+F(10,1024);ideal=(F(9,10)+10*F(8,9))/11;rounded=(F(889,1000)+10*F(768,1000))/11
 assert blind==F(11,1024) and ideal==F(881,990) and rounded==F(779,1000)
 result=dict(complete=True,pins={str(P/n):sha(P/n) for n in ['BLIND_ARGUMENT.md','EXACT_STATE_LIMIT.md','t138_pins.json','frozen/m16/state.py']},reused_independent_review='8f90409',premises='Pinned original bytes, exact interpolation/rank/margin-zero/positive derivative and gaps checked; gate/recurrence correctness reused from independent T138 review, not rerun.',blind_mass=str(blind),rank_blind_combined_error=str(ideal),rounded_claim_conditional=str(rounded),rounded_claim_unconditional=str((F(889,1000)+10*F(768,1000))/1024),theorem_findings=['The sign mass follows independent fair signs; exchangeable target-label idealization additionally uses the iid/rank-blind hypotheses.','Constant first-index prediction has error9/10 on each sign-count sector under iid inputs, refuting universal architecture-only8/9.','Positive x1 margin derivative and finitely many strict gate/rank/other-logit slacks ensure a positive epsilon preserving the advertised summary while winners differ.','Truth remains3 on both strict sides; first-index ownership at the crossing is irrelevant.','Exact hidden reconstruction from this summary would force equal readouts, contradicting the two-side implication.'],limitations=['Pointwise exact-real fixed-weight non-sufficiency only; no new epsilon/radius/probability or useful error bound.','No statement that arbitrarily tiny floating perturbations flip the implemented network.','No rejection of approximate held-out lookup performance, weighted omitted-input summaries or general compression.','The13-snapshot/helper/historical-fit provenance review belongs to h4; missing provenance does not imply leakage.','Rounded displayed rates are arithmetic inputs, not independently measured rates or confidence intervals.'],new_model_forward_search_calls=0,review_cpu=time.process_time()-t,checker_sha256=sha(Path(__file__)))
 (H/'t201_theory_saved_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,review_cpu=result['review_cpu'])))
if __name__=='__main__':main()
