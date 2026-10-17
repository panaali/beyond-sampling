"""Run as a repository module to verify package imports and failure charges."""
import json,time
from pathlib import Path
import numpy as np
from . import estimator
from .bounded_rank import AmbiguousRank
HERE=Path(__file__).resolve().parent
start=time.process_time();rows=[];real_enumerate=estimator.enumerate_correct
def deliberate_failure(model,path,audit,**kwargs):
 audit.add('deliberate_test_work',123)
 raise AmbiguousRank('deliberate ambiguity propagation check')
estimator.enumerate_correct=deliberate_failure
try:
 try:estimator.calculate(None)
 except AmbiguousRank as exc:
  assert exc.partial_stats['cost']['known_arithmetic_upper_bound']==123
  assert exc.partial_stats['failure_kind']=='numerical_ambiguity'
  rows.append(dict(case='injected_rank_ambiguity',error_type=type(exc).__name__,partial_stats=exc.partial_stats))
 else:raise AssertionError('expected explicit ambiguity')
finally:estimator.enumerate_correct=real_enumerate
try:estimator.est((np.zeros(2),np.eye(2),np.ones((4,2))),1,None)
except ValueError as exc:
 assert exc.partial_stats['cost']['known_arithmetic_upper_bound']==684
 assert exc.partial_stats['failure_kind']=='invalid_or_unsupported_input'
 assert exc.partial_stats['advisory_budget']==1
 rows.append(dict(case='unsupported_zero_input',error_type=type(exc).__name__,partial_stats=exc.partial_stats))
else:raise AssertionError('expected explicit unsupported-case failure')
out=dict(checks=rows,cpu_seconds=time.process_time()-start,package_import_succeeded=True)
(HERE/'failure_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
