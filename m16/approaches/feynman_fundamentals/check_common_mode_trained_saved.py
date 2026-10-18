from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import gzip,json,time
from check_common_mode_saved import P,O,sha,dec,validate
start=time.process_time();r=json.loads((P/'trained.json').read_text());gate=json.loads((P/'trained_gate.json').read_text())
assert r['complete'] and gate['complete'] and r['gate_sha256']==sha(P/'trained_gate.json')
for p,h in gate['pins'].items():assert sha(Path(p))==h,p
src=Path('/workspace/work/recover-small_h2_34/m16/approaches/fading_influence/trained.json');assert r['source_sha256']==sha(src)
s=json.loads(src.read_text());assert s['complete'] and len(s['models'])==2
raw=Path(r['raw']);assert r['raw_sha256']==sha(raw) and r['rows_completed']==768 and r['new_trained_forward_calls']==r['new_reference_draws']==0
rows=[dec(json.loads(x)) for x in gzip.decompress(raw.read_bytes()).decode().splitlines()];assert len(rows)==768
reports=[]
for mi,(sm,rm) in enumerate(zip(s['models'],r['models'],strict=True)):
 assert rm['complete'] and rm['rows']==384 and sm['name']==rm['name'] and sm['model']==rm['model'] and sm['checkpoint_sha256']==rm['checkpoint_sha256']
 m=sm['model'];H,T,D=m['H'],m['T'],m['D'];wi,wh,wo=m['wi'],m['wh'],m['wo'];counts=Counter();obstructions=changes=probes=0;first=None;bits=0
 for i,(w,row) in enumerate(zip(sm['witnesses'],rows[384*mi:384*(mi+1)],strict=True)):
  assert w['strict'] and w['local_derivative_certified'] and not w['boundary_gates']
  assert row['model']==sm['name'] and row['index']==w['index']==i and row['x']==w['x'] and row['history']==w['history'] and row['saved_radius']==dec(w['radius'])
  z=row['reconstructed'];assert z['L']==w['logit_map_numerators'] and z['denominator']==w['logit_denominator']==D**(T+1)
  # Rebuild integer affine coefficient rows from fixed masks, not input forwards.
  J=[[0]*T for _ in range(H)];expected=[];zeros=[]
  for t,mask in enumerate(w['history']):
   B=[[sum(wh[a][k]*J[k][j] for k in range(H))+(wi[a]*D**t if j==t else 0) for j in range(T)] for a in range(H)]
   for a,b in enumerate(B):
    value=sum(v*x for v,x in zip(b,w['x']));slope=sum(b)
    if not any(b):assert not mask[a];zeros.append([t,a])
    else:
     assert value and mask[a]==(value>0);expected.append(dict(t=t,i=a,row=b,active=mask[a],intercept=value,slope=slope))
   J=[b if mask[a] else [0]*T for a,b in enumerate(B)]
  assert expected==z['gate_rows'] and zeros==w['identically_zero_gates']
  assert [[sum(wo[a][k]*J[k][j] for k in range(H)) for j in range(T)] for a in range(T)]==z['L']
  q=row['result'];validate(w['x'],z,q)
  probes+=len(q['open_probes'])+len(q['crossing_points']);changes+=q['line_open_change'];counts.update(q['counts']);bits=max(bits,q['max_integer_bits'])
  if q['robust_opposite_pair'] is not None:
   obstructions+=1
   if first is None:
    pair=[q['open_probes'][j] for j in q['robust_opposite_pair']]
    first=dict(index=i,x=w['x'],shifts=[str(v['c']) for v in pair],radii=[str(v['radius']) for v in pair],predictions=[v['prediction'] for v in pair],target=pair[0]['target'],correct=[v['correct'] for v in pair])
 assert obstructions==rm['robust_obstructions'] and changes==rm['line_open_changes'] and probes==rm['affine_probe_evaluations'] and dict(counts)==rm['counts'] and bits==rm['max_integer_bits'] and (None if first is None else first['index'])==rm['first_obstruction_index']
 reports.append(dict(name=sm['name'],rows=384,robust_obstructions=obstructions,line_open_changes=changes,affine_probes=probes,first_exact_witness=first))
result=dict(complete=True,models=reports,author_output_sha256=sha(P/'trained.json'),raw_sha256=sha(raw),author_cpu=r['cpu'],review_cpu=time.process_time()-start,scope='All768 exact saved coefficient recurrences, strict gate intervals/all pair roots, probe logits/first-index ties, positive cube radii, first opposite witnesses and raw-count sums verified. No neural input, author evaluator, geometry or reference calls. Counts are not probabilities; excludes only natural common-mode invariance for these models, not arbitrary projections or conditional representations. Peak-bit/source operation ledger values conserved, not independently replayed.')
(O/'t191_common_mode_trained_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
