"""Independent saved H4T3 incidence/chain/scalar audit. No producer imports."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
from collections import defaultdict
import gzip,hashlib,json,math,os,resource,sys,time,traceback

P=Path(__file__).resolve().parent
START=time.process_time()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def gz(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def verify(pins):
 assert isinstance(pins,dict) and pins
 for p,h in pins.items():assert sha(p)==h,p
def cap():assert time.process_time()-START<17,'saved review CPU cap'
def pair(q):return [str(q.numerator),str(q.denominator)]
def frac(p):
 assert len(p)==2 and all(type(x)is str and str(int(x))==x for x in p)
 a,b=map(int,p);assert b>0;return F(a,b)
def det(a):
 x,y,z=a
 return x[0]*(y[1]*z[2]-y[2]*z[1])-x[1]*(y[0]*z[2]-y[2]*z[0])+x[2]*(y[0]*z[1]-y[1]*z[0])
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b):return sum(x*y for x,y in zip(a,b,strict=True))
def normray(r):
 assert len(r)==3 and all(type(x)is int for x in r)
 g=math.gcd(*r);assert g;return tuple(x//g for x in r)
def normal(v):
 d=math.lcm(*(x.denominator for x in v));a=[x.numerator*(d//x.denominator) for x in v]
 return normray(a) if any(a) else (0,0,0)
def simplex(x):
 rr=[normray(r) for r in x['rays']];assert len(rr)==3 and len(set(rr))==3
 v=F(abs(det(rr)),math.prod(sum(map(abs,r)) for r in rr));assert v>0 and v==frac(x['expected_det'])
 return rr,v
def cone(c):
 assert set(c)=={'rays','tight','constraints'}
 rr=[normray(r) for r in c['rays']];assert rr==[tuple(r) for r in c['rays']] and len(set(rr))==len(rr)
 constraints={int(k):tuple(v) for k,v in c['constraints'].items()}
 assert all(str(k) in c['constraints'] and k>=0 and len(v)==3 and all(type(x)is int for x in v) for k,v in constraints.items())
 assert any(det(m) for m in combinations(rr,3));assert len(rr)==len(c['tight'])
 for r,t in zip(rr,c['tight'],strict=True):
  assert type(t)is int and t>=0;values={k:dot(v,r) for k,v in constraints.items()};assert min(values.values())>=0
  assert t==sum(1<<k for k,v in values.items() if v==0)
  active=[constraints[k] for k,v in values.items() if v==0]
  assert any(any(z:=cross(a,b)) and (normray(z)==r or tuple(-x for x in normray(z))==r) for a,b in combinations(active,2))
 return rr,constraints
def dyadic(rows):
 s=1<<128;lo=hi=ab=0
 for r in rows:
  q=frac(r['det'])*r['weight']*s;lo+=q.numerator//q.denominator;hi+=-((-q.numerator)//q.denominator)
  q=abs(q);ab+=-((-q.numerator)//q.denominator)
 return dict(bits=128,signed_lower_units=str(lo),signed_upper_units=str(hi),absolute_upper_units=str(ab),term_count=len(rows))
def angle_algebra(row,z):
 """Check saved positive angle algebra without calling atan2 or a mass kernel."""
 vv=[[float(F(x,sum(map(abs,r)))) for x in r] for r in row['rays']]
 assert z['normalized_vertices_hex']==[[x.hex() for x in r] for r in vv]
 def fd(a,b):return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2]
 a,b,c=vv;ns=[fd(a,a),fd(b,b),fd(c,c)];ds=[fd(a,b),fd(b,c),fd(c,a)]
 assert z['norm2_hex']==[x.hex() for x in ns] and z['dot_hex']==[x.hex() for x in ds]
 assert min(ns)>0 and min(ds)>=0
 na,nb,nc=map(math.sqrt,ns);ab,bc,ca=ds;den=((na*nb*nc+ab*nc)+bc*na)+ca*nb
 assert den>0 and z['denominator_hex']==den.hex() and z['determinant_hex']==float(frac(row['det'])).hex()
 ang=float.fromhex(z['angle_hex']);assert math.isfinite(ang) and 0<ang<=math.pi/2
 unsigned=ang/(2*math.pi);val=row['weight']*unsigned
 assert z['unsigned_hex']==unsigned.hex() and z['value_hex']==val.hex() and z['weight']==row['weight'];return val
def save(r):
 r.update(cpu_seconds=time.process_time()-START,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 p=P/'results.tmp';p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');p.replace(P/'results.json')

def main(r):
 s=read(P/'scope.json');g=read(P/'gate.json');ci=read(P/'completed_inputs.json');rv=read(g['source_review'])
 own={str(P/n) for n in ['check.py','scope.json','README.md']}
 assert own<=set(rv['pins']) and own|{str(P/'completed_inputs.json'),g['source_review']}<=set(g['pins'])
 assert all(rv[k] is True for k in ['complete','pass_all','execution_allowed']) and g['ready'] is True
 for k in ['review_publication','execution_publication']:
  q=g[k];assert isinstance(q,str) and 8<=len(q)<=40 and all(x in '0123456789abcdef' for x in q)
 for pins in [s['pins'],g['pins'],ci['pins'],rv['pins']]:verify(pins)
 a=Path(s['author']);o=a/'output';p=read(a/'protocol.json');ag=read(a/'gate.json');ar=read(ag['review_path'])
 mandatory={str(a/n) for n in ['run.py','angle.py','PROTOCOL.md','protocol.json','launch.sh','gate.json','run1.log']}|{str(o/n) for n in ['results.json','progress.jsonl','model.json','gates.json.gz','capture.json.gz','chain.json.gz','baseline.jsonl','boundary.jsonl']}|{ag['review_path']}
 assert mandatory<=set(ci['pins']) and ci['raw_publication']
 assert all(ar[k] is True for k in ['complete','pass_all','execution_allowed'])
 for pins in [p['pins'],ag['pins'],ar['pins']]:verify(pins)
 old=read(p['prior_saved_review']);assert old['complete'] and old['pass_all'];verify(old['input_pins'])
 assert str(Path(sys.executable).resolve())==p['resolved_python'] and resource.getrlimit(resource.RLIMIT_CPU)==(20,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
 assert all(os.environ.get(k)=='1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'])
 d=read(o/'results.json');assert d['complete'] and d['pass_all'] and d['angle_controls_passed']
 assert d['forward_inputs']==d['rng_calls']==0 and d['new_geometry_captures']==1
 assert d['prior_chain_review']==dict(sha=sha(p['prior_saved_review']),controls_reused=8,guards_reused=6)
 assert d['runtime']['cpu_limit']==[20,20] and d['runtime']['as_limit']==[2**31,2**31] and d['runtime']['resolved']==p['resolved_python'] and d['numpy']==p['numpy']
 e=[[1,0,0],[0,1,0],[0,0,1]];m=[1,1,0]
 controls=[('octant',e,F(1,8)),('negative',[[-x for x in q] for q in e],F(1,8)),('permuted',e[::-1],F(1,8)),('scaled',[[3,0,0],[0,5,0],[0,0,7]],F(1,8)),('split_left',[e[0],m,e[2]],F(1,16)),('split_right',[m,e[1],e[2]],F(1,16))]
 expected=[]
 def event(phase,kind,data):expected.append(dict(phase=phase,kind=kind,data=data))
 event('prerequisites','prerequisites_passed',dict(protocol_sha=sha(a/'protocol.json'),gate_sha=sha(a/'gate.json')))
 for saved,(name,rays,truth) in zip(d['controls'],controls,strict=True):
  q=F(abs(det(rays)),math.prod(sum(map(abs,v)) for v in rays));row=dict(rays=rays,det=pair(q),weight=1)
  assert saved['name']==name and saved['input']==row and saved['exact_truth']==pair(truth)
  value=angle_algebra(row,saved['result']);assert abs(F.from_float(value)-truth)<=F(1,10**14)
  event('controls','angle_control_attempt',dict(name=name));event('controls','angle_control_result',saved)
 assert d['controls_work']==dict(determinants=6,integer_multiplications=108,integer_accumulations=36)
 model=read(o/'model.json');assert model['weights_sha']==sha(p['weights']) and model['shapes']==[[4],[4,4],[3,4]] and all(x in ['float16','float32','float64'] for x in model['dtypes'])
 wi=[frac(x) for x in model['model'][0]];wh=[[frac(x) for x in row] for row in model['model'][1]];wo=[[frac(x) for x in row] for row in model['model'][2]]
 assert len(wi)==4 and len(wh)==4 and len(wo)==3 and all(len(x)==4 for x in wh+wo)
 assert all(q.denominator&(q.denominator-1)==0 for q in wi+[q for row in wh+wo for q in row])
 gates=gz(o/'gates.json.gz');capture=gz(o/'capture.json.gz');chain=gz(o/'chain.json.gz');N=len(gates['cells'])
 assert d['capture']['stages']==gates['stages'] and gates['stages'][0]==2 and gates['stages'][-1]==N and len(gates['stages'])==3
 assert d['capture']['gate_cells']==N and d['capture']['sha']==sha(o/'capture.json.gz')
 assert {str(o/f'gate_{i:04d}_correct_cones.json.gz') for i in range(N)}<=set(ci['pins'])
 assert set(capture)=={'gate_simplices','correct_simplices','correct_cones'}
 event('capture','weight_decode_attempt',dict(path=p['weights']));event('capture','weight_decode_result',dict(model_sha=sha(o/'model.json')))
 event('capture','gate_capture_attempt',dict(H=4,T=3));event('capture','gate_capture_result',dict(cells=N,stages=gates['stages'],sha=sha(o/'gates.json.gz')))
 gs=defaultdict(list);cs=defaultdict(list)
 for x in capture['gate_simplices']:assert type(x['gate_id'])is int and 0<=x['gate_id']<N;gs[x['gate_id']].append(x)
 for x in capture['correct_simplices']:assert type(x['gate_id'])is int and 0<=x['gate_id']<N;cs[x['gate_id']].append(x)
 allcorrect=[];gt=ct=F();ng=nc=0
 for i,row in enumerate(gates['cells']):
  cap();rr,con=cone(row['cone']);A=[[frac(x) for x in v] for v in row['map']];assert len(A)==4 and all(len(v)==3 for v in A)
  assert all(dot(v,r)>=0 for v in A for r in rr)
  logits=[[sum(wo[j][h]*A[h][k] for h in range(4)) for k in range(3)] for j in range(3)]
  file=gz(o/f'gate_{i:04d}_correct_cones.json.gz');assert file['gate_id']==i;good=[]
  for c in file['cones']:
   rays,cc=cone(c);assert all(cc[k]==v for k,v in con.items())
   js=[j for j in range(3) if 63+j not in cc and all(63+k in cc for k in range(3) if k!=j)];assert len(js)==1;j=js[0]
   want=dict(con);above=0
   for k in range(3):
    if k==j:continue
    margin=normal([x-y for x,y in zip(logits[j],logits[k],strict=True)])
    if any(margin):want[60+k]=margin
    else:assert j<k,'permanent tie must belong to earlier output'
    v=tuple(int(t==j)-int(t==k) for t in range(3));got=cc[63+k];assert got in [v,tuple(-x for x in v)];above+=got!=v;want[63+k]=got
   assert above==1 and cc==want;good.append(cc);allcorrect.append(dict(gate_id=i,cone=c))
  event('capture','gate_geometry_attempt',dict(index=i))
  assert gs[i]
  for x in gs[i]:
   rays,q=simplex(x);assert all(dot(v,r)>=0 for v in con.values() for r in rays);gt+=q;ng+=1
  for x in cs[i]:
   rays,q=simplex(x);assert any(all(dot(v,r)>=0 for v in cc.values() for r in rays) for cc in good);ct+=q;nc+=1
  event('capture','gate_geometry_result',dict(index=i,gate_simplices=ng,correct_simplices=nc,correct_cones=len(allcorrect)))
 assert allcorrect==capture['correct_cones'] and d['capture']['correct_cones']==len(allcorrect)
 assert gt==8 and 0<ct<8 and d['capture']['gate_simplices']==ng and d['capture']['correct_simplices']==nc and d['capture']['gate_det_total']==pair(gt)
 scaled=ct*(1<<128);assert d['capture']['correct_det_bracket']==dict(bits=128,lower_units=str(scaled.numerator//scaled.denominator),upper_units=str(-((-scaled.numerator)//scaled.denominator)),exact_numerator_bits=ct.numerator.bit_length(),exact_denominator_bits=ct.denominator.bit_length())
 r['capture_checked']=dict(gates=N,correct_cones=len(allcorrect),gate_simplices=ng,correct_simplices=nc);save(r)
 event('chain','chain_attempt',dict(simplices=nc));facets={};originals=[];sourcechecks=[]
 for i,x in enumerate(capture['correct_simplices']):
  rays=tuple(sorted(normray(v) for v in x['rays']));dd=det(rays);q=frac(x['expected_det']);orth=[]
  for j in range(3):
   signs={1 if v[j]>0 else -1 for v in rays if v[j]};assert len(signs)==1;orth.append(next(iter(signs)))
  orth=tuple(orth);ids=[];expanded=F()
  for k in range(3):
   face=rays[:k]+rays[k+1:];key=orth,face
   if key not in facets:facets[key]=dict(id=len(facets),orthant=list(orth),rays=[list(v) for v in face],coefficient=0,occurrences=0,signed_det=pair(F(det((orth,)+face),3*math.prod(sum(map(abs,v)) for v in face))))
   f=facets[key];co=(1 if dd>0 else -1)*((-1)**k);f['coefficient']+=co;f['occurrences']+=1;ids.append([f['id'],co]);expanded+=co*frac(f['signed_det'])
  assert expanded==q;record=dict(index=i,det=pair(q),expanded_det=pair(expanded),facets=ids);sourcechecks.append(record)
  event('chain','simplex_attempt',dict(index=i));event('chain','simplex_result',record);originals.append(dict(rays=[list(v) for v in rays],weight=1,det=pair(q)))
 records=list(facets.values());terms=[]
 for f in records:
  q=frac(f['signed_det'])
  if f['coefficient'] and q:terms.append(dict(facet_id=f['id'],rays=[f['orthant']]+f['rays'],weight=f['coefficient']*(1 if q>0 else -1),det=pair(abs(q))))
 assert chain['originals']==originals and chain['facets']==records and chain['terms']==terms and chain['source_checks']==sourcechecks
 nd=nc+len(records);counts=dict(determinants=nd,integer_multiplications=18*nd,integer_accumulations=6*nd,gcd_calls=9*nc,integer_ray_divisions=9*nc,facet_dictionary_lookups=3*nc,facet_coefficient_adds=3*nc,rational_identity_terms=3*nc)
 assert chain['counts']==counts and type(chain['observed_max_integer_bits'])is int and chain['observed_max_integer_bits']>0
 dc=dict(sha=sha(o/'chain.json.gz'),original_terms=nc,boundary_terms=len(terms),unique_facets=len(records),cancelled_facets=sum(x['coefficient']==0 for x in records),live_facets=sum(x['coefficient']!=0 for x in records),zero_anchor_facets=sum(x['coefficient']!=0 and frac(x['signed_det'])==0 for x in records),counts=counts,original_totals=dyadic(originals),reduced_totals=dyadic(terms))
 assert dc==d['chain'];event('chain','chain_result',dc)
 assert max(int(dc[k]['signed_lower_units']) for k in ['original_totals','reduced_totals'])<=min(int(dc[k]['signed_upper_units']) for k in ['original_totals','reduced_totals'])
 for name,rows in [('baseline',originals),('boundary',terms)]:
  cap();raw=[json.loads(x) for x in (o/(name+'.jsonl')).read_text().splitlines()];vals=[]
  for i,(row,z) in enumerate(zip(rows,raw,strict=True)):
   assert z['index']==i;v=angle_algebra(row,z['result']);vals.append(v)
   event('integration','angle_attempt',dict(method=name,index=i));event('integration','angle_result',dict(method=name,index=i,value_hex=v.hex()))
  saved=d['integrations'][name];total=math.fsum(vals)
  assert saved['calls']==len(rows) and saved['value']==total and saved['value_hex']==total.hex() and saved['absolute_sum']==math.fsum(map(abs,vals)) and saved['sha']==sha(o/(name+'.jsonl'))
  assert saved['named_kernel_categories_per_call']==45 and saved['rational_coordinate_conversions_per_call']==9
 a0,b=d['integrations']['baseline'],d['integrations']['boundary'];comparison=dict(difference=b['value']-a0['value'],term_ratio=b['calls']/a0['calls'],integration_cpu_saved=a0['cpu']-b['cpu'],floating_evaluation_certified=False,shared_new_discovery_cpu=d['decode_cpu']+d['gate_capture_cpu']+d['classification_triangulation_recording_cpu'],chain_cpu_includes_recording=True)
 assert d['comparison']==comparison;event('integration','complete',comparison)
 actual=[json.loads(x) for x in (o/'progress.jsonl').read_text().splitlines()];last=0
 for row,ex in zip(actual,expected,strict=True):
  t=row.pop('cpu');assert math.isfinite(t) and last<=t<=d['cpu_seconds'];last=t;assert row==ex
 assert d['geometry_work']['gate_cells']==N and d['geometry_work']['correct_cones']==len(allcorrect) and d['geometry_work']['simplices']==ng+nc
 assert all(type(v)is int and v>=0 for v in d['geometry_work'].values()) and all(type(v)is int and 0<=v<=4096 for v in d['geometry_observed_bits'].values())
 for t in [d[k] for k in ['decode_cpu','gate_capture_cpu','classification_triangulation_recording_cpu','chain_cpu']]+[a0['cpu'],b['cpu']]:assert math.isfinite(t) and 0<=t<=d['cpu_seconds']
 cap();r.update(complete=True,pass_all=True,chain_terms_before=nc,chain_terms_after=len(terms),progress_events=len(actual),comparison=comparison,integrations=d['integrations'],producer_cpu=d['cpu_seconds'],input_pins=ci['pins'],producer_calls=0,atan2_calls=0,model_forward_calls=0,geometry_captures=0,qualification='Saved incidence/readout-map/chain/scalar consistency; global map recurrence, completeness/disjointness, libm atan2 values and full native cost remain source-qualified.')

if __name__=='__main__':
 assert not (P/'results.json').exists() and not (P/'results.tmp').exists();r=dict(complete=False,pass_all=False);save(r)
 try:main(r)
 except BaseException:r.update(complete=False,pass_all=False,failure=traceback.format_exc());raise
 finally:save(r)
