"""Complementary T198 source/provenance gate. No author function execution."""
from pathlib import Path
import json,hashlib,subprocess,time
P=Path('/workspace/work/recover-small_h4_4/review/split_attribution_review/revised_cap');H=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
 start=time.process_time();m=json.loads((P/'manifest.json').read_text());repo=P.parents[2]
 assert len(m['records'])==73 and len(m['prune_paths'])==51 and not m['rendered_prune_report_present']
 for r in m['records']:
  b=(P/'frozen'/r['original']).read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256']
  orig=subprocess.check_output(['git','show',m['author_commit']+':'+r['original']],cwd=repo);assert b==orig
  assert subprocess.check_output(['git','rev-parse',m['author_commit']+':'+r['original']],cwd=repo).decode().strip()==r['git_blob']
 out=dict(complete=True,pass_all=True,source_findings_commit='7a8c9404',saved_checker_source='5a2d1445',author_commit=m['author_commit'],pins={n:sha((P/n).read_bytes()) for n in ['README.md','SOURCE_REVIEW.md','manifest.json','check_saved.py','SAVED_PROTOCOL.md']},checked_payloads=73,new_author_cap_geometry_mass_model_calls=0,review_cpu=time.process_time()-start,
 findings=['Ideal circular cone containment and half-beta probability hold for unit generators and a normalized axis with positive minimum dot product.','Replacing1-cos² removes the original cancellation mechanism, but fixedSIN_PAD/BETA_REL/MASS_PAD constants alone lack a documented universal rounding theorem. No actual pointed trained failure is inferred.','Zero-sum ±coordinate rays yield NaN axis under ordinary nontrapping semantics; Python min(1,nan) gives1, so the source returns about1/2 rather than hullmass1. This lies outside the stated pointed caller contract and is a general-domain gap only.','Printed834checks omit most ray/reference records and skip NaN references; no independent all834 numerical confirmation is claimed.','Frozen records contain51T3 pruning rows and no T4/T5 replay. Canonical511 comparison is distinct from historicalN=flops/500. flops=2ops is only declared convention.','Pruning-only bounds require domain/cover/error premises and do not certify underlying recipe or numerical wrong-centroid masses; by_step excludes mass pad and global accumulation is not outward.'],
 clearance='One saved-only check_saved.py run over frozen51T3 rows/10groups and printed metadata; external20CPU/2GiB,one-thread reviewlock,immutable completed source gate and all input pins published first. No author imports or reference/cap replay.',
 checker_review=['Exact stored-float Fraction point ratios and continuous canonical draws; no CI or integer-floor adoption','Exact fixed coverage and nonnegative integer counters, historicalN ulp residual and unpadded step sums','Pad residual retained as diagnostic, not asserted to certify individual rounding','Immutable outputs and durable failure record; preflight failures retained by exclusive launch log'],
 limitations='Source findings and checker clearance only; completed saved arithmetic result requires separate review. No shared implementation repair or cap certification granted.')
 (H/'revised_cap_source_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,payloads=73,review_cpu=out['review_cpu'])))
if __name__=='__main__':main()
