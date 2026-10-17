"""Independent exhaustive saved IAMAX digest and numeric subset arithmetic."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
import json,hashlib,itertools,math,time
from pathlib import Path
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_34/m16/approaches/inverse_numeric_cost');D=json.loads((P/'controls.json').read_text());S=json.loads((P/'summary.json').read_text());O=P.parent/'inverse4_cost_review'
assert hashlib.sha256((P/'controls.json').read_bytes()).hexdigest()==S['controls_sha256']
for name,digest in S['source_pins'].items():assert hashlib.sha256((O/name).read_bytes()).hexdigest()==digest
# Independent scalar-register transcription: MAX chooses right operand for NaN/tie.
def maximum(a,b):return a if a>b else b
def simulate(x):
 n=len(x);a=abs(x[0]);pairs=[[a,a] for _ in range(4)];ab=2;cm=0
 if n in (3,4):
  pairs[2]=[maximum(a,abs(x[1])),maximum(a,abs(x[2]))];ab+=2;cm+=2
 if n in (2,4):
  pairs[3]=[maximum(a,abs(x[-1]))]*2;ab+=2;cm+=2
 lo=[maximum(pairs[0][i],pairs[1][i]) for i in range(2)]
 hi=[maximum(pairs[2][i],pairs[3][i]) for i in range(2)]
 v=[maximum(lo[i],hi[i]) for i in range(2)];v=maximum(v[1],v[0]);cm+=7
 scanned={1:0,2:2,3:2,4:4}[n];ab+=2*scanned
 for j in range(scanned):
  cm+=1;z=abs(x[j])
  if math.isnan(z) or math.isnan(v) or z==v:return j+1,ab,cm
 return scanned+1,ab,cm
alphabet=[float.fromhex(x) for x in D['alphabet']];digest=hashlib.sha256();count=0;rows=[]
for n in range(1,5):
 maxima=[0,0];nr=0
 for x in itertools.product(alphabet,repeat=n):
  idx,a,c=simulate(x);assert 1<=idx<=n
  digest.update(json.dumps(([v.hex() for v in x],idx,a,c)).encode());nr+=1;maxima=[max(maxima[0],a),max(maxima[1],c)]
 row=dict(n=n,cases=nr,max_abs_lanes=maxima[0],max_cmp_lanes=maxima[1]);assert row==D['iamax_cases'][n-1];rows.append(row);count+=nr
assert count==4680 and digest.hexdigest()==D['iamax_result_sha256']
assert D['baseline_parity'] and D['dispatch_restored'] and len(D['inverse_controls'])==8
for row in D['inverse_controls']:
 e=row['events'];assert [z['args'][0] for z in e if z['kernel']=='iamax']==[4,3,2,1];assert not any(z['kernel']=='gemm' for z in e)
assert sum(r['max_abs_lanes'] for r in rows)==32 and sum(r['max_cmp_lanes'] for r in rows)==44
assert sum(S['added_numeric_categories'].values())==102 and S['combined_numeric_units']==188+102==290
for row,visible,calls in zip(S['parent_compositions'],[553400768,455976715],[2659906,2138536]):
 assert row['calls']==calls and row['floating_expression_upper']==calls*188 and row['numeric_guard_upper']==calls*102 and row['combined_numeric_upper']==calls*290
 assert row['visible_plus_inverse_numeric_subset']==visible+calls*290 and not row['complete_cost']
out=dict(exhaustive_saved_native_digest_matches=count,iamax_maxima=rows,inverse_event_controls=8,numeric_subset_upper=290,parent_numeric_subtotals=[r['visible_plus_inverse_numeric_subset'] for r in S['parent_compositions']],review_cpu=time.process_time()-start,scope='Independent register transcription reproduces saved native digest; raw parity assertions inspected, no native rerun. Numeric subset only; no complete implementation bound or matched-budget win.')
(H/'inverse_numeric_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
