"""Recount saved small-kernel shapes; no inverse or trained-model evaluation."""
from pathlib import Path
import hashlib,json,time
P=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 t=time.process_time()
 controls=json.loads((P/'kernel_controls.json').read_text())
 assert controls['all_baseline_outputs_bitwise_unchanged']
 rows=[]
 for row in controls['rows']:
  counts=dict(dot_and_outer_subtraction=0,gemv=0,scal=0,lu_reciprocal_upper=0,triangular=0,upper_pack_reciprocal=0)
  for e in row['events']:
   k=e['kernel']; a,b,c,d=e['args']
   if k=='dot':
    assert a in (1,2) and b==4 and c==1
    counts['dot_and_outer_subtraction']+=2*a+2
   elif k=='gemv':
    assert (a,b,c,d) in [(4,0,1,1),(3,1,1,1),(2,2,1,1),(1,3,1,1)]
    counts['gemv']+=2*a*b+2*a if b else 0
    # One possible normal nonzero-pivot reciprocal for each LU column.
    counts['lu_reciprocal_upper']+=1
   elif k=='scal':
    assert a in (1,2,3) and b==1
    counts['scal']+=a
   elif k in ('lower','upper'):
    assert (a,b,c,d)==(4,2,4,0)
    counts['triangular']+=a*b + b*a*(a-1)
   elif k=='upack':
    assert (a,b,c,d)==(4,4,4,0)
    counts['upper_pack_reciprocal']+=a
   else: assert k in ('iamax','swap','copy','lpack'),k
  n=2 if row['name']=='batch' else 1
  total=sum(counts.values())
  assert total<=188*n
  rows.append(dict(name=row['name'],categories=counts,floating_expression_upper=total))
 parent=Path('/workspace/work/recover-small_h4_4/m16/approaches')
 compositions=[]
 for label,folder,expected in [('exact','plackett_cost_review',2659906),('recipe','recipe_cost_review',2138536)]:
  p=parent/folder/'summary.json'; data=json.loads(p.read_text())
  inv=[x for x in data['opaque_calls'] if x['key'][1]=='np.linalg.inv']
  assert all(x['key'][2]=='(4, 4)' for x in inv)
  count=sum(x['count'] for x in inv); assert count==expected
  compositions.append(dict(label=label,parent_path=str(p),parent_sha256=sha(p),inverse_sites=inv,inverse_calls=count,inverse_floating_expression_upper=count*188,parent_visible_explicit=data['total_visible_explicit_with_cold_setup'],visible_plus_inverse_fp_subset=data['total_visible_explicit_with_cold_setup']+count*188,complete_cost=False))
 result=dict(per_inverse_envelope=188,lu=56,packed_solves=132,rows=rows,parent_compositions=compositions,controls_sha256=sha(P/'kernel_controls.json'),cpu=time.process_time()-t,scope='FP expression envelope only: FMA=2, division=1; abs, comparisons, integer/control, allocation/copy, wrapper/runtime excluded; no full-cost/MSE claim')
 (P/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(per_inverse=188,controls=len(rows),compositions=[(x['label'],x['inverse_floating_expression_upper']) for x in compositions],cpu=result['cpu'])))
if __name__=='__main__': main()
