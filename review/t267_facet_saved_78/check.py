"""Independent saved incidence audit; no producer imports, density or mass evaluation."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from math import gcd,prod,factorial,isfinite,pi
import gzip,hashlib,json,resource,time,traceback
P=Path(__file__).resolve().parent;START=time.process_time()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def compressed(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def verify(d):
 assert isinstance(d,dict) and d
 for p,h in d.items():assert sha(p)==h,p
def save(r):
 r.update(cpu_seconds=time.process_time()-START,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (P/'results.tmp').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');(P/'results.tmp').replace(P/'results.json')
def cap():assert time.process_time()-START<19,'review internal CPU cap'
def frac(z):
 assert isinstance(z,list) and len(z)==2 and all(type(x)is str and str(int(x))==x for x in z)
 assert int(z[1])>0
 q=F(int(z[0]),int(z[1]));assert z==[str(q.numerator),str(q.denominator)];return q
def pair(q):return [str(q.numerator),str(q.denominator)]
def det(rows):
 # Fraction-free pivoted Bareiss, unlike the producer permutation sum.
 a=[list(row) for row in rows];n=len(a);sign=1;previous=1
 assert all(len(row)==n and all(type(x)is int for x in row) for row in a)
 for k in range(n-1):
  candidates=[i for i in range(k,n) if a[i][k]]
  if not candidates:return 0
  pivot=candidates[0]
  if pivot!=k:a[k],a[pivot]=a[pivot],a[k];sign=-sign
  pivot=a[k][k]
  for i in range(k+1,n):
   for j in range(k+1,n):
    value=pivot*a[i][j]-a[i][k]*a[k][j];assert value%previous==0;a[i][j]=value//previous
   a[i][k]=0
  previous=pivot
 return sign*a[-1][-1]
def ray(raw,d):
 assert len(raw)==d and all(type(x)is int for x in raw);g=gcd(*raw);assert g>0;return tuple(x//g for x in raw)
def normalized_det(rows):return F(det(rows),prod(sum(abs(x) for x in row) for row in rows))
def counters(n,unique,d):
 k=n+unique
 return {'gcd_calls':n*d*d,'integer_ray_divisions':n*d*d,'determinants':k,'integer_multiplications':k*d*factorial(d),'integer_accumulations':k*factorial(d),'facet_dictionary_lookups':d*n,'facet_coefficient_adds':d*n,'rational_identity_terms':d*n}
def audit_chain(source,ans,d,r,label):
 assert set(ans)=={'originals','facets','terms','source_checks','counts','observed_max_integer_bits'}
 assert len(ans['originals'])==len(ans['source_checks'])==len(source)
 # Group all occurrences independently, then compare producer ID/order separately.
 groups={};checks=[];originals=[]
 for index,item in enumerate(source):
  cap();rs=tuple(sorted(ray(x,d) for x in item['rays']));assert len(set(rs))==d
  signs=[]
  for j in range(d):
   values={1 if x[j]>0 else -1 for x in rs if x[j]};assert len(values)==1;signs.append(next(iter(values)))
  orth=tuple(signs);dd=det(rs);assert dd!=0;absolute=abs(normalized_det(rs));orientation=1 if dd>0 else -1
  if item['expected_det']is not None:assert frac(item['expected_det'])==absolute
  incidence=[]
  for i in range(d):
   face=rs[:i]+rs[i+1:];key=(orth,face);c=orientation if i%2==0 else -orientation
   groups.setdefault(key,[]).append((index,c));incidence.append((key,c))
  originals.append({'rays':[list(x) for x in rs],'weight':1,'det':pair(absolute)})
  checks.append((incidence,absolute))
  if index and index%256==0:r['partial']={'chain':label,'originals_checked':index+1};save(r)
 assert ans['originals']==originals and len(ans['facets'])==len(groups)
 ids={};facets_by_id={};signed={}
 for position,record in enumerate(ans['facets']):
  cap();assert set(record)=={'id','orthant','rays','coefficient','occurrences','signed_det'} and record['id']==position
  key=(tuple(record['orthant']),tuple(tuple(z) for z in record['rays']));assert key in groups and key not in ids
  occurrences=groups[key];coefficient=sum(c for _,c in occurrences);dd=normalized_det((key[0],)+key[1])
  assert record['occurrences']==len(occurrences) and record['coefficient']==coefficient and frac(record['signed_det'])==dd
  ids[key]=position;facets_by_id[position]=record;signed[key]=dd
 expected_terms=[]
 for key,index in sorted(ids.items(),key=lambda kv:kv[1]):
  coefficient=sum(c for _,c in groups[key]);sd=signed[key]
  if coefficient and sd:expected_terms.append({'facet_id':index,'rays':[list(x) for x in (key[0],)+key[1]],'weight':coefficient*(1 if sd>0 else -1),'det':pair(abs(sd))})
 assert ans['terms']==expected_terms
 for index,((incidence,absolute),row) in enumerate(zip(checks,ans['source_checks'],strict=True)):
  expansion=sum((c*signed[key] for key,c in incidence),F());assert expansion==absolute
  assert row=={'index':index,'det':pair(absolute),'expanded_det':pair(expansion),'facets':[[ids[k],c] for k,c in incidence]}
 assert ans['counts']==counters(len(source),len(groups),d)
 assert type(ans['observed_max_integer_bits'])is int and ans['observed_max_integer_bits']>0
 return {'source_simplices':len(source),'unique_facets':len(groups),'cancelled_facets':sum(sum(c for _,c in v)==0 for v in groups.values()),'remaining_facets':sum(sum(c for _,c in v)!=0 for v in groups.values()),'zero_anchor_terms':sum(sum(c for _,c in v)!=0 and signed[k]==0 for k,v in groups.items()),'integration_terms':len(expected_terms),'counts':ans['counts']}
def moments(items,d):
 result=[F() for _ in range(1+d+d*(d+1)//2)]
 for item in items:
  rows=[[F(x,sum(abs(z) for z in ray)) for x in ray] for ray in item['rays']];w=frac(item['det'])*item['weight']
  values=[F(1)]+[sum((row[j] for row in rows),F())/d for j in range(d)]
  for j in range(d):
   for k in range(j,d):values.append(sum((rows[a][j]*rows[b][k]*(2 if a==b else 1) for a in range(d) for b in range(d)),F())/(d*(d+1)))
  result=[a+w*b for a,b in zip(result,values,strict=True)]
 return list(map(pair,result))
def controls():
 e=[(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)];m=(1,1,0,0);f=(1,1,2,0)
 rows=[('orthant4',4,[e],(4,4,0)),('split4',4,[[e[0],m,e[2],e[3]],[m,e[1],e[2],e[3]]],(6,7,1)),('duplicate4',4,[e,e],(4,4,0)),('permuted4',4,[e[::-1]],(4,4,0)),('scaled4',4,[[tuple(3*x for x in row) for row in e]],(4,4,0)),('negative4',4,[[tuple(-x for x in row) for row in e]],(4,4,0)),('nonconforming4',4,[[e[0],m,e[2],e[3]],[e[1],m,f,e[3]],[e[1],f,e[2],e[3]]],(10,11,1)),('orthant3',3,[[(1,0,0),(0,1,0),(0,0,1)]],(3,3,0))]
 return [(name,d,[{'rays':r,'expected_det':None} for r in rr],counts) for name,d,rr,counts in rows]
def dyadic(items):
 lo=hi=av=0;scale=2**128
 for row in items:
  q=frac(row['det'])*row['weight']*scale;lo+=q.numerator//q.denominator;hi+=-((-q.numerator)//q.denominator);v=abs(q);av+=(v.numerator+v.denominator-1)//v.denominator
 return {'bits':128,'signed_lower_units':str(lo),'signed_upper_units':str(hi),'absolute_upper_units':str(av),'term_count':len(items)}
def read_integrals(path,items,summary,totals):
 total=F();absolute=F();rows=[]
 with Path(path).open() as stream:
  for index,(line,item) in enumerate(zip(stream,items,strict=True)):
   cap();row=json.loads(line);assert set(row)=={'index','weight','determinant_hex','kernel_integral_hex','value_hex','q_min_hex','q_max_hex'};assert row['index']==index and row['weight']==item['weight']
   numbers={k:float.fromhex(row[k]) for k in ('determinant_hex','kernel_integral_hex','value_hex','q_min_hex','q_max_hex')}
   assert all(isfinite(x) for x in numbers.values());assert numbers['determinant_hex']==float(frac(item['det']))
   assert numbers['kernel_integral_hex']>0 and .25-1e-12<=numbers['q_min_hex']<=numbers['q_max_hex']<=1+1e-12
   value=item['weight']*(numbers['determinant_hex']/(2*pi**2)*numbers['kernel_integral_hex']);assert value.hex()==row['value_hex']
   total+=F.from_float(value);absolute+=abs(F.from_float(value));rows.append(row)
 assert summary['calls']==len(items) and summary['nodes']==8000*len(items) and summary['rows_sha']==sha(path)
 assert summary['value']==float(total) and summary['value_hex']==float(total).hex() and summary['sum_abs']==float(absolute)
 bound=F(int(totals['absolute_upper_units']),2**128)*F(249,54*3**19)
 assert frac(summary['ideal_truncation_bound_upper'])==bound and summary['ideal_truncation_bound_float']==float(bound)
 assert summary['named_arithmetic_per_node']=={'matmul_mul':16,'matmul_add':12,'norm_square':4,'norm_add':3,'q_square':1,'reciprocal':1,'weighted_mul':1,'reduction_add_allowance':1}
 assert isfinite(summary['cpu_seconds']) and summary['cpu_seconds']>0
 return rows

def main(r):
 s=read(P/'scope.json');g=read(P/'gate.json');ci=read(P/'completed_inputs.json');rv=read(g['source_review'])
 own={str(P/n) for n in ('check.py','scope.json','README.md')};assert own<=set(rv['pins']) and own|{str(P/'completed_inputs.json'),g['source_review']}<=set(g['pins'])
 assert g['ready'] is True and g['source_review_publication'] and g['execution_publication'] and all(rv[k] is True for k in ('complete','pass_all','execution_authorized'))
 for mapping in (s['pins'],g['pins'],ci['pins'],rv['pins']):verify(mapping)
 a=Path(s['author']);out=a/'output';p=read(a/'protocol.json');ag=read(a/'gate.json');ar=read(ag['review_path'])
 mandatory={str(a/n) for n in ('primitive.py','fixtures.py','run.py','PROTOCOL.md','protocol.json','launch.sh','gate.json','run1.log')}|{str(out/n) for n in ('results.json','progress.jsonl','chain.json.gz','rule.npz','baseline_integrals.jsonl','boundary_integrals.jsonl')}|{str(out/(name+'.json.gz')) for name,_,_,_ in controls()}|{ag['review_path'],p['archive']}
 assert mandatory<=set(ci['pins']) and ci['raw_commit'] and ci['publication']
 assert all(ar[k] is True for k in ('complete','pass_all','execution_allowed'))
 for mapping in (p['pins'],ag['pins'],ar['pins']):verify(mapping)
 assert {str(a/n) for n in ('primitive.py','fixtures.py','run.py','PROTOCOL.md','protocol.json')}<=set(ar['pins'])
 assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,)*2
 d=read(out/'results.json');assert d['complete'] is True and d['pass_all'] is True and d['controls_passed'] is True
 assert d['runtime']['invoked']==p['invoked_python']
 assert d['runtime']['cpu_limit']==[110,110] and d['runtime']['as_limit']==[2**31]*2 and d['runtime']['resolved']==p['resolved_python']
 assert d['numpy']=={'version':p['numpy_version'],'path':p['numpy_path']}
 assert [d[k] for k in ('model_calls','geometry_enumerations','rng_calls')]==[0,0,0]
 assert isfinite(d['cpu_seconds']) and 0<=d['cpu_seconds']<=110 and type(d['max_rss_kib'])is int and d['max_rss_kib']>0
 expected_events=[('prerequisites','prerequisites_passed',{'protocol_sha':sha(a/'protocol.json'),'gate_sha':sha(a/'gate.json')})]
 assert len(d['controls'])==8
 for record,(name,dim,inputs,expected) in zip(d['controls'],controls(),strict=True):
  cap();ans=compressed(out/(name+'.json.gz'));counts=audit_chain(inputs,ans,dim,r,name)
  ma=moments(ans['originals'],dim);mb=moments(ans['terms'],dim);assert ma==mb and frac(ma[0])==(2 if name=='duplicate4' else 1)
  assert record=={'name':name,'d':dim,'archive_sha':sha(out/(name+'.json.gz')),'original_moments':ma,'reduced_moments':mb,'live':counts['remaining_facets'],'unique':counts['unique_facets'],'cancelled':counts['cancelled_facets']}
  assert (record['live'],record['unique'],record['cancelled'])==expected
  expected_events.extend([('controls','control_attempt',{'name':name}),('controls','control_result',record)]);r['controls'].append({'name':name,'passed':True});save(r)
 guard_names=['empty','zero_ray','duplicate_ray','mixed_orthant','float_coordinate','bad_saved_det']
 guard_msgs=['nonempty dimension 3 or 4 chain required','zero ray','duplicate ray','not a full dimensional single orthant','integer rays only','archived normalized determinant mismatch']
 guard_counts=[{}, {'gcd_calls':16,'integer_ray_divisions':12},{'gcd_calls':16,'integer_ray_divisions':16},{'gcd_calls':16,'integer_ray_divisions':16},{},{'gcd_calls':16,'integer_ray_divisions':16,'determinants':1,'integer_multiplications':96,'integer_accumulations':24}]
 for row,name,msg,ct in zip(d['guards'],guard_names,guard_msgs,guard_counts,strict=True):
  assert row=={'name':name,'rejected':True,'message':msg,'counts':ct}
  expected_events.extend([('controls','guard_attempt',{'name':name}),('controls','guard_result',row)])
 source=compressed(p['archive']);assert source['checkpoint_sha256']==p['checkpoint_sha256'] and len(source['correct_simplices'])==5058 and len(source['gate_simplices'])==3853
 inputs=[]
 for row in source['correct_simplices']:
  rr=row['integer_rays'];assert all(type(x)is str and str(int(x))==x for ray in rr for x in ray)
  inputs.append({'rays':[[int(x) for x in ray] for ray in rr],'expected_det':row['normalized_absolute_determinant']})
 assert d['input']=={'checkpoint_sha256':p['checkpoint_sha256'],'correct_simplices':5058,'gate_simplices':3853,'archive_sha':sha(p['archive'])}
 ans=compressed(out/'chain.json.gz');counts=audit_chain(inputs,ans,4,r,'H4T4');ot=dyadic(ans['originals']);rt=dyadic(ans['terms'])
 assert d['chain']==dict(counts,archive_sha=sha(out/'chain.json.gz'),original_totals=ot,reduced_totals=rt)
 assert max(int(ot['signed_lower_units']),int(rt['signed_lower_units']))<=min(int(ot['signed_upper_units']),int(rt['signed_upper_units']))
 assert d['rule']['order']==20 and d['rule']['nodes']==8000 and d['rule']['sha']==sha(out/'rule.npz')
 assert all(isfinite(d[k]) and d[k]>0 for k in ('archive_decode_conversion_cpu','dyadic_accounting_cpu'))
 assert isfinite(d['chain_cpu']) and d['chain_cpu']>0 and isfinite(d['rule']['setup_cpu']) and d['rule']['setup_cpu']>0
 expected_events.append(('saved_complex','archive_decode_attempt',{'path':p['archive']}))
 for row in ans['source_checks']:expected_events.extend([('saved_complex','simplex_attempt',{'index':row['index']}),('saved_complex','simplex_result',row)])
 expected_events.extend([('saved_complex','chain_result',d['chain']),('quadrature','rule_attempt',{'order':20}),('quadrature','rule_result',d['rule'])])
 assert set(d['integration'])=={'baseline','boundary'}
 for label,items,totals in [('baseline',ans['originals'],ot),('boundary',ans['terms'],rt)]:
  rows=read_integrals(out/(label+'_integrals.jsonl'),items,d['integration'][label],totals)
  for row in rows:expected_events.extend([('quadrature','mass_attempt',{'method':label,'index':row['index']}),('quadrature','mass_result',{'method':label,'index':row['index'],'value_hex':row['value_hex']})])
 b=d['integration']['baseline'];c=d['integration']['boundary']
 expected={'boundary_minus_baseline':c['value']-b['value'],'combined_ideal_bound':b['ideal_truncation_bound_float']+c['ideal_truncation_bound_float'],'node_ratio':c['nodes']/b['nodes'],'integration_cpu_ratio':c['cpu_seconds']/b['cpu_seconds'],'cold_boundary_components_cpu':d['archive_decode_conversion_cpu']+d['chain_cpu']+d['dyadic_accounting_cpu']+d['rule']['setup_cpu']+c['cpu_seconds'],'component_timer_excludes_imports_prerequisites_controls_and_some_serialization':True,'inherited_enumeration_not_included_in_incremental_timer':True,'floating_evaluation_certified':False}
 assert d['comparison']==expected;expected_events.append(('quadrature','complete',expected));last_cpu=0
 with (out/'progress.jsonl').open() as stream:
  for line,(phase,kind,data) in zip(stream,expected_events,strict=True):
   event=json.loads(line);assert set(event)=={'phase','kind','cpu','data'} and event['phase']==phase and event['kind']==kind and event['data']==data
   assert isfinite(event['cpu']) and last_cpu<=event['cpu']<=d['cpu_seconds'];last_cpu=event['cpu']
 assert len(expected_events)==34+4*5058+2*len(ans['terms']);cap()
 r.update(complete=True,pass_all=True,guards=6,simplices=5058,chain_summary=counts,events=len(expected_events),original_totals=ot,reduced_totals=rt,comparison=expected,baseline_value=b['value'],boundary_value=c['value'],named_baseline_node_categories=39*b['nodes'],named_boundary_node_categories=39*c['nodes'],author_cpu=d['cpu_seconds'],author_calls=0,model_geometry_mass_density_rng_calls=0,rule_array_decodes=0,peak_bits_independently_replayed=False,quadrature_recomputed=False,input_pins=ci['pins'],scope_sha256=sha(P/'scope.json'))
if __name__=='__main__':
 assert not (P/'results.json').exists() and not (P/'results.tmp').exists();r={'complete':False,'pass_all':False,'controls':[]};save(r)
 try:main(r)
 except BaseException:r.update(complete=False,pass_all=False,failure=traceback.format_exc());raise
 finally:save(r)
