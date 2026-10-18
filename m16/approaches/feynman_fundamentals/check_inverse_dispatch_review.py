"""Independent saved passthrough parity/event-count audit; no native rerun."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,time
start=time.process_time();here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h2_34/m16/approaches/inverse4_cost_review');base=json.loads((src/'baseline.json').read_text());spy=json.loads((src/'instrumented.json').read_text());kernel=json.loads((src/'kernel_controls.json').read_text());assert len(base['rows'])==len(spy['rows'])==len(kernel['rows'])==11;out=[]
for b,s,k in zip(base['rows'],spy['rows'],kernel['rows']):
 assert b['name']==s['name']==k['name'];assert {a:v for a,v in s.items() if a!='spy'}==b
 for key in ('value_hex','failure'):
  if key in b:assert k[key]==b[key]
 n=2 if b['name']=='batch' else 1;assert all(s['spy'][key]==n for key in ('calls','n4','nrhs4','lda4','ldb4','identity_rhs','finite_input'));events=k['events'];counts=Counter(e['kernel'] for e in events);assert not counts['gemm']
 dot=sum(2*e['args'][0]+1 for e in events if e['kernel']=='dot');outer=counts['dot'];gemv=sum(2*m*q+2*m for e in events if e['kernel']=='gemv' for m,q in [e['args'][:2]] if q);scal=sum(e['args'][0] for e in events if e['kernel']=='scal');reciprocal=4*n # Conservative normal-pivot count, singular/subnormal branches can skip.
 triangular=sum(m*q+q*m*(m-1) for e in events if e['kernel'] in ('upper','lower') for m,q in [e['args'][:2]]);pack=4*counts['upack'];envelope=dot+outer+gemv+scal+reciprocal+triangular+pack;assert envelope<=188*n
 if b['name'] not in ('zero','duplicate_row','subnormal_diagonal','batch'):
  assert (dot,outer,gemv,scal,reciprocal,triangular,pack)==(11,3,32,6,4,128,4)
 out.append(dict(name=b['name'],inverse_calls=n,float_expression_envelope=envelope,gemm_calls=counts['gemm']))
result=dict(controls=out,normal_single_inverse_float_expression_envelope=188,review_cpu=time.process_time()-start,scope='Installed small-size event shapes and pinned-source floating expression envelope, FMA as multiply+add. Excludes abs/comparisons/index/control/packing-copy/runtime; not exact machine instruction count or complete inverse/hardware cost.')
(here/'inverse_dispatch_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
