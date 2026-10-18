"""Saved finite controls: exhaustive masks, exact intervals, independent MP formulas."""
from pathlib import Path
from fractions import Fraction as Q
from itertools import product
import hashlib,json,resource,time,traceback,math
P=Path(__file__).resolve().parent;START=time.process_time()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verify(pins):
 assert isinstance(pins,dict) and pins
 for p,h in pins.items():assert sha(p)==h,p
def unpack(x):
 if isinstance(x,dict):
  if set(x)=={'q'}:
   a=x['q'];assert len(a)==2 and all(type(v)is str and str(int(v))==v for v in a);assert int(a[1])>0
   z=Q(int(a[0]),int(a[1]));assert [str(z.numerator),str(z.denominator)]==a;return z
  return {k:unpack(v) for k,v in x.items()}
 if isinstance(x,list):return [unpack(v) for v in x]
 return x
def save(r):
 r.update(cpu_seconds=time.process_time()-START,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 t=P/'results.tmp';t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(P/'results.json')
def cap():assert time.process_time()-START<19

def masks(model,y,depth):
 wi,wh,wo=model;answers=[]
 for bits in product((0,1),repeat=2*depth):
  h=[(Q(0),Q(0))]*2;lo=Q(-2);hi=Q(2);valid=True
  for t in range(depth):
   z=[(wi[i]*y[t]+sum(wh[i][j]*h[j][0] for j in range(2)),wi[i]*[1,-1][t]+sum(wh[i][j]*h[j][1] for j in range(2))) for i in range(2)]
   for i,(c,s) in enumerate(z):
    on=bits[2*t+i]
    if not s:
     if (c>0)!=bool(on):valid=False
    else:
     root=-c/s
     if (s>0)==bool(on):lo=max(lo,root)
     else:hi=min(hi,root)
   h=[z[i] if bits[2*t+i] else (Q(0),Q(0)) for i in range(2)]
  if valid and lo<hi:
   logits=[(sum(w*h[j][0] for j,w in enumerate(row)),sum(w*h[j][1] for j,w in enumerate(row))) for row in wo]
   answers.append([lo,hi,[list(x) for x in logits]])
 answers.sort();assert answers[0][0]==-2 and answers[-1][1]==2
 assert all(a[1]==b[0] for a,b in zip(answers,answers[1:]));return answers

def points(model,y,x,words=None):
 wi,wh,wo=model;h=[Q(0),Q(0)];ww=[]
 for t in range(2):
  z=[wi[i]*(y[t]+[1,-1][t]*x)+sum(wh[i][j]*h[j] for j in range(2)) for i in range(2)]
  w=[v>0 for v in z] if words is None else words[t];ww.append(w);h=[z[i] if w[i] else Q(0) for i in range(2)]
 return [sum(w*h[j] for j,w in enumerate(row)) for row in wo],ww

def frozen(model,y,words):
 wi,wh,wo=model;h=[(Q(0),Q(0))]*2
 for t in range(2):
  z=[(wi[i]*y[t]+sum(wh[i][j]*h[j][0] for j in range(2)),wi[i]*[1,-1][t]+sum(wh[i][j]*h[j][1] for j in range(2))) for i in range(2)]
  h=[z[i] if words[t][i] else (Q(0),Q(0)) for i in range(2)]
 return [[Q(-2),Q(2),[[sum(w*h[j][k] for j,w in enumerate(row)) for k in range(2)] for row in wo]]]

def coalesce(parts):
 out=[]
 for a,b in sorted(parts):
  if out and out[-1][1]==a:out[-1][1]=b
  else:out.append([a,b])
 return out

def error_sets(cells):
 errors=[[],[]]
 for a,b,ls in cells:
  cuts={a,b};c=ls[0][0]-ls[1][0];s=ls[0][1]-ls[1][1]
  if s and a<-c/s<b:cuts.add(-c/s)
  cuts=sorted(cuts)
  for l,u in zip(cuts,cuts[1:]):
   m=(l+u)/2;winner=0 if c+s*m>=0 else 1;errors[1-winner].append([l,u])
 return [coalesce(x) for x in errors]

def main(r):
 s=read(P/'scope.json');g=read(P/'gate.json');ci=read(P/'completed_inputs.json');rv=read(g['source_review']);a=Path(s['author'])
 own={str(P/n) for n in ('check.py','scope.json','README.md')}
 assert own<=set(rv['pins']) and own|{str(P/'completed_inputs.json'),g['source_review']}<=set(g['pins'])
 assert g['ready'] is True and g['source_review_publication'] and g['execution_publication']
 assert all(rv[k] is True for k in ('complete','pass_all','execution_authorized'))
 for pp in (s['pins'],g['pins'],ci['pins'],rv['pins']):verify(pp)
 required={str(a/n) for n in ('controls.py','intervals.py','gaussian.py','controls_protocol.json','controls_launch.sh','CONTROLS.md','PLAN.md','controls_gate.json','controls_run1.log','controls_output/results.json','controls_output/progress.jsonl')}
 assert required<=set(ci['pins']) and ci['raw_publication']
 ap=read(a/'controls_protocol.json');ag=read(a/'controls_gate.json');ar=read(ap['review_path'])
 assert ag['ready'] is True and all(ar[k] is True for k in ('complete','pass_all','execution_authorized'))
 for pp in (ap['pins'],ag['pins'],ar['pins']):verify(pp)
 assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,)*2
 d=unpack(read(a/'controls_output/results.json'));assert d['complete'] is True and d['pass_all'] is True
 assert d['pins']==ag['pins'] and d['runtime']['resolved']==ap['python_resolved']
 assert {k:d[k] for k in ('events','partition_calls','direct_synthetic_calls','masked_synthetic_calls','mask_setup_calls','guard_attempts','target_calls','rng_calls')}==dict(events=38,partition_calls=8,direct_synthetic_calls=28,masked_synthetic_calls=28,mask_setup_calls=4,guard_attempts=6,target_calls=0,rng_calls=0)
 assert math.isfinite(d['cpu_seconds']) and 0<=d['cpu_seconds']<20 and type(d['maxrss_kib'])is int and d['maxrss_kib']>0
 rows=d['rows'];assert len(rows)==18 and all(x['passed'] is True for x in rows)
 z=[[Q(0),Q(0)],[Q(0),Q(0)]];ident=[[Q(1),Q(0)],[Q(0),Q(1)]]
 fixtures=[('zero',[[Q(0),Q(0)],z,ident],[Q(0),Q(0)]),('sign',[[Q(1),Q(-1)],z,ident],[Q(0),Q(0)]),('shifted',[[Q(1),Q(-1)],z,ident],[Q(0),Q(1,2)]),('recurrent',[[Q(1),Q(-1)],ident,ident],[Q(1,2),Q(-1,4)])]
 probes=list(map(Q,[-1.5,-.75,-.25,0,.25,.75,1.5]))
 for row,(name,model,y) in zip(rows[:4],fixtures,strict=True):
  cap();assert row['kind']=='network' and row['name']==name and row['model']==model and row['y']==y and row['direction']==[1,-1] and row['window']==[-2,2]
  cells=masks(model,y,2);assert row['cells']==cells and row['layers']==[len(masks(model,y,1)),len(cells)]
  _,words=points(model,y,Q(0));assert row['frozen_words']==words
  fr=frozen(model,y,words);assert row['frozen']==fr and row['frozen_layers']==[1,1] and row['errors']==error_sets(cells)
  assert len(row['probes'])==7
  for check,x in zip(row['probes'],probes,strict=True):
   logits,_=points(model,y,x);masked,_=points(model,y,x,words);winner=0 if logits[0]>=logits[1] else 1
   assert check==dict(x=x,logits=logits,affine=logits,winner=winner,masked=masked,frozen_affine=masked)
   for l,u,ls in cells:
    if l<=x<=u:assert [c+b*x for c,b in ls]==logits
  assert row['ledger'] and all(type(v)is int and v>=0 for v in row['ledger'].values()) and all(type(v)is int and v>=0 for v in row['max_returned_bits'])
  r['checked'].append(name);save(r)
 diffs=[('overlap',[(1,2),(3,4)],[(1,5)]),('actual_negative',[(-2,0)],[]),('frozen_negative',[],[(-2,0)]),('alternating',[(0,1),(2,3)],[(1,2)])]
 for row,(name,aa,bb) in zip(rows[4:8],diffs,strict=True):
  a0=[list(map(Q,p)) for p in aa];b0=[list(map(Q,p)) for p in bb];assert row['kind']=='difference' and row['name']==name and row['actual']==a0 and row['frozen']==b0
  # Endpoint jumps integrate the signed indicator, independently of midpoint membership.
  jumps={}
  for sign,parts in ((1,a0),(-1,b0)):
   for l,u in parts:jumps[l]=jumps.get(l,0)+sign;jumps[u]=jumps.get(u,0)-sign
  cuts=sorted(jumps);level=0;expected=[]
  for l,u in zip(cuts,cuts[1:]):
   level+=jumps[l]
   if level:
    if expected and expected[-1][1]==l and expected[-1][2]==level:expected[-1][1]=u
    else:expected.append([l,u,level])
  assert row['value']==expected;r['checked'].append(name)
 import mpmath as mp
 mp.mp.dps=80
 gauss=[('near_positive',Q(0),Q(1,4)),('near_negative',Q(-1,4),Q(0)),('wide_positive',Q(1),Q(3)),('wide_negative',Q(-3),Q(-1))]
 errors=[]
 for row,(name,l,u) in zip(rows[8:12],gauss,strict=True):
  assert row['kind']=='gaussian' and row['name']==name and row['band']==[l,u,1]
  lo=mp.mpf(l.numerator)/l.denominator;hi=mp.mpf(u.numerator)/u.denominator;c=lo if lo>=0 else hi;delta=hi-lo if lo>=0 else lo-hi;sign=1 if delta>0 else -1
  density=mp.exp(-c*c)/mp.sqrt(mp.pi)
  ex=dict(first=sign*density*delta,second=sign*density*delta*(1-c*delta),direct=(mp.erf(hi)-mp.erf(lo))/2,remainder1=mp.exp(-mp.mpf(1)/2)*delta**2/mp.sqrt(2*mp.pi),remainder2=abs(delta)**3/(3*mp.sqrt(mp.pi)))
  v=row['value'];assert set(v)==set(ex)|{'detail'} and len(v['detail'])==1
  for k,q in ex.items():
   assert type(v[k])is float and math.isfinite(v[k]);err=abs(mp.mpf(v[k])-q);assert err<mp.mpf('1e-14');errors.append(float(err))
   assert v['detail'][0]['mass' if k=='direct' else k]==v[k]
  assert abs(v['direct']-v['first'])<=v['remainder1']+3e-15 and abs(v['direct']-v['second'])<=v['remainder2']+3e-15
  if name.startswith('wide'):assert v['second']<0
  r['checked'].append(name)
 guards=[('shape','model shape'),('nonfraction','exact Fraction inputs required'),('window','window must straddle or touch zero'),('truth_boolean','truth'),('cross_zero','one-sided signed band'),('bad_sign','one-sided signed band')]
 for row,(name,msg) in zip(rows[12:],guards,strict=True):assert row==dict(kind='guard',name=name,passed=True,rejection=msg)
 events=[{'event':'started'}]+[dict(event=e,index=i) for i in range(18) for e in ('started_row','completed_row')]+[{'event':'completed'}]
 assert [json.loads(x) for x in (a/'controls_output/progress.jsonl').read_text().splitlines()]==events
 cap();r.update(complete=True,pass_all=True,rows=18,events=38,mask_assignments=80,exact_point_reconstructions=56,gaussian_max_abs_difference=max(errors),gaussian_precision=80,gaussian_tolerance='1e-14 absolute, finite diagnostic only',operation_and_peak_bit_internals_replayed=False,guards='saved reasons and source only; no rejection replay',producer_calls=0,target_face_rng_calls=0,author_cpu_seconds=d['cpu_seconds'],input_pins=ci['pins'])
if __name__=='__main__':
 assert not (P/'results.json').exists() and not (P/'results.tmp').exists();r=dict(complete=False,pass_all=False,checked=[]);save(r)
 try:main(r)
 except BaseException:r.update(complete=False,pass_all=False,failure=traceback.format_exc());raise
 finally:save(r)
