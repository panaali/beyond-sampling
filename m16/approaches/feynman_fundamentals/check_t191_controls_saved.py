from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import ast,hashlib,itertools,json,math,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/weight_circles');OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'controls.json').read_text());g=json.loads((P/'controls_gate.json').read_text())
assert r['complete'] and g['complete'] and r['gate_sha256']==sha(P/'controls_gate.json') and r['source_sha256']==sha(P/'controls.py')
for p,h in g['pins'].items():assert sha(Path(p))==h,p
assert r['new_reference_draws']==r['new_trained_calls']==0
assert (r['circle_calls'],r['deterministic_forward_calls'],r['frame_success_calls'],r['frame_rejection_calls'])==(32,180,14,2)
# Independent expected matrix construction, no recurrent/evaluator calls.
wi=[1,-1,0,0,1,-1,1,-1,0,0];wh=[[0]*10 for _ in range(10)]
for i,j,v in [(2,0,1),(3,1,1),(4,0,-1),(4,1,1),(5,0,1),(5,1,-1),(6,2,-1),(6,3,1),(7,2,1),(7,3,-1),(8,4,1),(9,5,1)]:wh[i][j]=v
wo=[[0]*10 for _ in range(3)]
for i,cols in enumerate([(6,7,8,9),(4,5,8,9),(4,5,6,7)]):
 for j in cols:wo[i][j]=-1
assert r['models']['median']==[wi,wh,wo]
assert r['models']['cycle']==[wi,wh,[wo[1],wo[2],wo[0]]]
assert r['models']['difference']==[[1,-1,1,-1],[[0,0,0,0],[0,0,0,0],[-1,1,0,0],[1,-1,0,0]],[[0,0,1,0],[0,0,0,1],[0,0,0,0]]]
assert r['models']['rank_failure']==[[0],[[0]],[[0],[0],[0],[0]]]
seen=set()
for x in r['forward']:
 name=x['model'];a,b,c=x['x'];key=(name,a,b,c);assert key not in seen;seen.add(key)
 assert len({a,b,c})==3 and all(-2<=v<=2 for v in (a,b,c))
 median=next(i for i,v in enumerate((a,b,c)) if min(a,b,c)<v<max(a,b,c))
 pred=(0 if c>b else 1) if name=='difference' else median if name=='median' else (median+2)%3
 assert x['state_equal'] and x['pred']==x['expected']==pred and x['target']==median
assert seen=={(n,*x) for n in ('difference','median','cycle') for x in itertools.permutations(range(-2,3),3)}
seen=set();maxw=0.
for x in r['wedges']:
 key=(x['orientation'],x['k']);assert key not in seen;seen.add(key)
 target=F(8-x['k'],16);maxw=max(maxw,abs(x['value']-float(target)))
 assert abs(x['value']-float(target))<=5e-14 and x['error']==x['value']-x['expected']
 a,b=x['a'],x['b'];dot=lambda u,v:sum(F(y)*F(z) for y,z in zip(u,v,strict=True))
 assert abs(dot(a,a)-1)<F(1,10**14) and abs(dot(b,b)-1)<F(1,10**14) and abs(dot(a,b))<F(1,10**14)
 n1,n2=x['n1'],x['n2'];cosang=float(dot(n1,n2))/math.sqrt(float(dot(n1,n1)*dot(n2,n2)))
 assert abs(x['value']-(math.pi-math.acos(cosang))/(2*math.pi))<5e-14
 assert x['frame_kernel_numeric']==39 and x['frame_positive_norm_comparisons']==2
assert seen=={(n,k) for n in ('identity','hadamard') for k in range(1,8)}
assert {x['sign'] for x in r['degenerate_frames']}=={-1,1} and all(x['rejected'] for x in r['degenerate_frames'])
expected={'difference':F(1,3),'median':F(1),'cycle':F(0),'rank_failure':F(1,2)};seen=set();total=0;opaques=0;maxcircle=0.
for x in r['circles']:
 name=x['model'];key=(name,x['phase']);assert key not in seen;seen.add(key)
 assert x['baseline']==x['traced'] and x['baseline_hex']==x['traced_hex']==float(x['traced'][0]).hex() and x['legacy_baseline']==x['legacy_traced']
 err=abs(F(x['traced'][0])-expected[name]);assert err<=F(5,10**13);maxcircle=max(maxcircle,float(err))
 assert x['expected_circle']==float(expected[name]) and x['gaussian_expected']==(.25 if name=='rank_failure' else float(expected[name]))
 H=len(r['models'][name][0]);T=len(r['models'][name][2]);a,b=x['a'],x['b']
 assert len(a)==len(b)==T and abs(sum(a))<1e-14 and abs(sum(b))<1e-14
 if T==4:assert a[0]==b[0]==0
 for ledger in (x['ledger'],x['startup']):
  counts=Counter()
  for site,cs in ledger['sites'].items():
   assert site
   for cat,n in cs.items():assert type(n) is int and n>=0;counts[cat]+=n
  assert dict(counts)==ledger['totals'];assert sum(counts.values())==ledger['inclusive_explicit_total']==ledger['named_numeric_total']+ledger['integer_control_total']
  assert not ledger['complete_implementation_cost']
  for family in ('native_calls','opaque_calls','memory_calls'):
   keys=set()
   for event in ledger[family]:
    k=tuple(event['key']);assert k not in keys;keys.add(k);assert type(event['count']) is int and event['count']>0
 L=x['ledger'];native=L['native_calls'];mul=add=nc=0
 for e in native:
  if e['key'][0]=='np.einsum':
   shape=ast.literal_eval(e['key'][2]);assert len(shape)==3 and shape[2]==2 and shape[1] in (H,T)
   elements=math.prod(shape)*e['count'];mul+=elements*H;add+=elements*(H-1);nc+=e['count']
 assert nc==T+1 and L['totals']['einsum_mul']==mul and L['totals']['einsum_add']==add
 assert x['startup']['named_numeric_total']==1
 assert sum(e['count'] for e in L['opaque_calls'] if e['key'][1]=='np.argsort')==1
 assert not any('closure' in c for c in L['totals'])
 total+=L['inclusive_explicit_total']+x['startup']['inclusive_explicit_total'];opaques+=sum(e['count'] for e in L['opaque_calls'])
assert seen=={(n,p) for n in expected for p in (0.,math.pi/7,math.pi/4,math.pi/2)}
result=dict(complete=True,source_gate=g['review'],author_output_sha256=sha(P/'controls.json'),wedges=14,degenerate_rejections=2,saved_forward_records=180,circle_pairs=16,source_calls=32,inclusive_explicit_plus_startup=total,opaque_calls=opaques,max_wedge_error=maxw,max_circle_error=maxcircle,author_cpu=r['cpu'],review_cpu=time.process_time()-start,scope='Saved model definitions/analytic outputs/pins/raw ledgers and shape-based einsum counts independently checked. Forward state_equal flags retained with source algebra review; raw hidden states were not serialized. No evaluator/forward/circle replay, generic correctness/full-native or trained claim.')
(OUT/'t191_controls_saved_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
