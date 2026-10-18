"""Independent T231 saved review. Stdlib NPY decoding, binary-bit exact masses.
No NumPy/author imports, geometry, probability integration, models or RNG.
"""
from pathlib import Path
from fractions import Fraction as F
import ast,hashlib,json,math,os,resource,struct,sys,time,traceback,zipfile
HERE=Path(__file__).resolve().parent
OWNER=Path('/workspace/work/recover-small_h4_4/review/prefix_mass_coverage_review')
OUT=HERE/'output'
DEN=1<<1074
KEYS=[f'{m}_k{k}' for m in ['target','zoo16_4','zoo16_5'] for k in [2,3,4]]+['gpu_k3','gpu_k4','gpu_k5']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def same(a,b,where):assert a==b,(where,a,b)
def put(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def fraction(v):return F(int(v['numerator']),int(v['denominator']))
def checkrat(got,want,label):
 want=F(want);same(fraction(got),want,label);same(got['float'],float(want),label+' display')
def npy(z,name,payload=True):
 """Read a bounded NPY header; do not evaluate array element code."""
 info=z.getinfo(name+'.npy')
 with z.open(info) as f:
  prefix=f.read(8);assert prefix[:6]==b'\x93NUMPY'
  version=tuple(prefix[6:]);assert version in [(1,0),(2,0)]
  n=2 if version==(1,0) else 4;hlen=int.from_bytes(f.read(n),'little');assert 0<hlen<=65536
  header=ast.literal_eval(f.read(hlen).decode('latin1'))
  same(set(header),{'descr','fortran_order','shape'},'NPY fields')
  desc=header['descr'];shape=header['shape'];assert isinstance(desc,str) and isinstance(shape,tuple) and all(type(x)is int and x>=0 for x in shape)
  assert type(header['fortran_order'])is bool
  assert len(desc)>=3 and desc[0] in '<>|=' and desc[1] in 'iufbUS'
  size=int(desc[2:])*(4 if desc[1]=='U' else 1)
  count=math.prod(shape);same(info.file_size,8+n+hlen+count*size,'NPY member length')
  raw=f.read() if payload else None
  if payload:same(len(raw),count*size,'NPY payload length')
 if desc[1] in 'US':dtype=desc
 elif desc[1]=='b':same(size,1,'bool itemsize');dtype='bool'
 else:dtype={'i':'int','u':'uint','f':'float'}[desc[1]]+str(8*size)
 return {'shape':list(shape),'dtype':dtype,'desc':desc,'fortran':header['fortran_order'],'bytes':size},raw

def mass_bits(payload):
 units=[];bad=negative=zeros=0
 for (bits,) in struct.iter_unpack('<Q',payload):
  sign=bits>>63;e=(bits>>52)&2047;mant=bits&((1<<52)-1)
  if e==2047:bad+=1;continue
  if e==0 and mant==0:zeros+=1;units.append(0);continue
  if sign:negative+=1
  u=mant if e==0 else ((1<<52)|mant)<<(e-1)
  units.append(-u if sign else u)
 return units,{'nonfinite':bad,'negative':negative,'zero':zeros}

def integer_values(meta,payload):
 desc=meta['desc'];assert desc[1] in 'iu' and meta['bytes'] in [1,2,4,8]
 endian='big' if desc[0]=='>' else 'little';n=meta['bytes']
 return [int.from_bytes(payload[i:i+n],endian,signed=desc[1]=='i') for i in range(0,len(payload),n)]

def audit_one(spec,row):
 key=spec['key'];archive=OWNER/spec['archive'];report=read(OWNER/spec['report'])
 same(row['archive_sha256'],sha(archive),key+' archive');same(row['report_sha256'],sha(OWNER/spec['report']),key+' report')
 same(row['key'],key,'key');same(row['kind'],spec['kind'],'kind');same(row['k'],spec['k'],'k')
 with zipfile.ZipFile(archive) as z:
  keys=sorted(n[:-4] for n in z.namelist());assert all(n.endswith('.npy') for n in z.namelist())
  same(keys,['event','gate','mass'] if spec['kind']=='width' else ['complete','mass','patterns'],key+' members')
  metas={};data={}
  for n in keys:metas[n],data[n]=npy(z,n,payload=n!='event')
 same(row['keys'],keys,key+' keys')
 same(row['arrays'],{n:{k:meta[k] for k in ['shape','dtype']} for n,meta in metas.items()},key+' array metadata')
 mm=metas['mass'];assert mm['desc']=='<f8' and len(mm['shape'])==1 and mm['shape'][0]>0
 count=mm['shape'][0];same(row['mass_count'],count,key+' count')
 units,adverse=mass_bits(data.pop('mass'))
 for k,v in adverse.items():same(row[k],v,key+'/'+k)
 if spec['kind']=='width':
  report=report[key];gm=metas['gate'];same(gm['shape'],[count],key+' gate shape');same(metas['event']['shape'],[count],key+' event shape');assert metas['event']['desc'][1] in 'US'
  ids=integer_values(gm,data.pop('gate'));same(min(ids),0,key+' minID');regions=max(ids)+1;assert regions<=10**7
  dense=set(ids)==set(range(regions));same(row['dense_gate_ids'],dense,key+' dense');valid=dense
  same(row['complete'],None,key+' completeness absent')
  expected={d:report['N'][f'{float(d):.0e}'] for d in spec['deltas']}
 else:
  pm=metas['patterns'];same(pm['desc'],'|u1',key+' pattern dtype');same(pm['shape'],[count,2*spec['k']],key+' pattern shape');raw=data.pop('patterns');width=2*spec['k']
  if pm['fortran']:histories={bytes(raw[j*count+i] for j in range(width)) for i in range(count)}
  else:histories={raw[i:i+width] for i in range(0,len(raw),width)}
  unique=len(histories);del histories,raw;same(row['unique_patterns'],unique,key+' distinct byte histories');valid=unique==count;regions=count
  same(metas['complete']['shape'],[],'complete scalar');same(metas['complete']['dtype'],'bool','complete dtype')
  same(row['complete'],bool(data['complete'][0]),key+' complete');same(row['reported_complete'],report['complete'],key+' reported complete')
  for field in ['subtrees','subtrees_done']:same(row[field],report[field],key+'/'+field)
  expected={d:report['N_delta'][f'{float(d):g}'] for d in spec['deltas']}
 same(row['regions'],regions,key+' regions');same(row['reported_regions'],report['regions'],key+' reported regions');same(row['history_group_schema_valid'],valid,key+' schema validity')
 schema=read(OWNER/'saved/output'/f'{key}_schema.json')
 for field in ['key','archive_sha256','report_sha256','kind','k','keys','arrays','mass_count','nonfinite','negative','zero','regions','reported_regions','complete','history_group_schema_valid']:
  same(schema[field],row[field],key+' durable schema/'+field)
 result={'key':key,'mass_count':count,'regions':regions,'complete_flag':row['complete'],'adverse':adverse,'history_group_schema_valid':valid}
 if adverse['nonfinite'] or adverse['negative'] or not valid:
  same(row['status'],'unresolved',key+' invalid status');assert 'arithmetic_skipped' in row and 'exact' not in row;result['valid']=False;return result
 if sum(units)<=0:
  same(row['status'],'unresolved',key+' zero status');same(row['exact'],{'valid_nonnegative_total':False},key+' zero total');result['valid']=False;return result
 if spec['kind']=='width':
  grouped=[0]*regions
  for g,u in zip(ids,units,strict=True):grouped[g]+=u
  units=grouped
 w=sorted(units);del units;total=sum(w);square=sum(u*u for u in w);exact=row['exact'];same(exact['valid_nonnegative_total'],True,key+' valid')
 for field,value in [('total',F(total,DEN)),('total_minus_one',F(total,DEN)-1),('minimum',F(w[0],DEN)),('maximum',F(w[-1],DEN)),('inverse_simpson',F(total*total,square))]:checkrat(exact[field],value,key+'/'+field)
 same(exact['zero_count'],sum(x==0 for x in w),key+' grouped zeros');same(row['status'],'complete',key+' status')
 requests=[]
 for d in spec['deltas']:
  requests.extend([(F(d),d,'decimal_delta'),(1-F.from_float(1-float(d)),d,'rounded_one_minus_binary_delta')])
 requests.sort();j=0;omitted=0;counts={};minimality=[]
 for eps,d,kind in requests:
  # Maximal discarded ascending suffix, independent of author's descending prefix.
  while j<len(w) and (omitted+w[j])*eps.denominator<=eps.numerator*total:omitted+=w[j];j+=1
  n=len(w)-j;assert n>0
  excess=eps-F(omitted,total);shortfall=F(w[j],total)-excess;assert excess>=0 and shortfall>0
  got=exact['coverage'][d][kind];same(got['N'],n,key+'/'+d+'/'+kind);same(got['reported_N'],expected[d],key+' reportedN');same(got['matches_reported'],n==expected[d],key+' match flag')
  checkrat(got['selected_excess_relative_total'],excess,key+' excess');checkrat(got['previous_shortfall_relative_total'],shortfall,key+' shortfall')
  if kind=='decimal_delta':counts[d]=n
  minimality.append({'delta':d,'kind':kind,'N':n,'omitted_units':str(omitted),'boundary_units':str(w[j]),'excess':str(excess),'shortfall':str(shortfall)})
 same(set(exact['coverage']),set(spec['deltas']),key+' threshold coverage')
 for d in spec['deltas']:
  same(set(exact['coverage'][d]),{'decimal_delta','rounded_one_minus_binary_delta'},key+' threshold conventions')
  n=row['source_arithmetic_counts'][d];assert type(n)is int and 1<=n<=regions+1
  same(row['source_arithmetic_matches'][d],n==expected[d],key+' source count comparison')
 for field in ['source_sorted_sum','source_sum_minus_one','reported_sum_minus_one','reported_sum_discrepancy']:assert math.isfinite(row[field])
 same(row['source_sum_minus_one'],row['source_sorted_sum']-1,key+' source sum display')
 same(row['reported_sum_minus_one'],report.get('mass_sum_minus_1',report.get('leafsum_minus_1')),key+' reported sum')
 same(row['reported_sum_discrepancy'],row['source_sum_minus_one']-row['reported_sum_minus_one'],key+' source sum difference')
 if spec['kind']=='gpu':
  same(row['reported_inverse_simpson'],report['mass_eff_regions'],key+' reported Simpson');same(row['inverse_simpson_discrepancy'],row['source_inverse_simpson']-row['reported_inverse_simpson'],key+' source Simpson difference')
 result.update(valid=True,total=str(F(total,DEN)),inverse_simpson=str(F(total*total,square)),counts=counts,minimality=minimality)
 return result

def main():
 assert not OUT.exists();OUT.mkdir();sys.set_int_max_str_digits(0)
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));start=time.process_time()
 result={'complete':False,'pass_all':False,'rows':[],'author_calls':0,'model_calls':0,'rng_calls':0,'probability_or_geometry_calls':0,'current':None}
 def save():result.update(cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);put(OUT/'results.json',result)
 def event(stage,**kw):
  with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'stage':stage,**kw})+'\n');f.flush();os.fsync(f.fileno())
 try:
  own=read(HERE/'protocol.json');gate=read(HERE/'review_gate.json');inputs=read(HERE/'input_pins.json')
  assert gate['complete'] is True and gate['pass_all'] is True
  mandatory={str(HERE/n) for n in ['check.py','protocol.json','input_pins.json']}
  assert mandatory<=set(gate['pins'])
  reviewer=gate['source_review'];assert reviewer['commit'] and reviewer['path'] in gate['pins']
  same(gate['pins'][reviewer['path']],reviewer['sha256'],'reviewer pin')
  review=read(reviewer['path']);assert review['complete'] is True and review['pass_all'] is True
  assert {str(HERE/'check.py'),str(HERE/'protocol.json')}<=set(review['pins'])
  for path,h in review['pins'].items():same(sha(path),h,'source reviewer dependency')
  required_inputs={str(OWNER/'saved'/n) for n in ['output/results.json','output/progress.jsonl','run1.log','gate.json']}
  required_inputs.update(str(OWNER/'saved/output'/f'{key}{suffix}.json') for key in KEYS for suffix in ['', '_schema'])
  assert required_inputs<=set(inputs['pins']), 'completed owner evidence coverage'
  for mapping in [own['pins'],gate['pins'],inputs['pins']]:
   assert mapping
   for path,h in mapping.items():same(sha(path),h,path)
  same(gate['pins'][str(HERE/'protocol.json')],sha(HERE/'protocol.json'),'own protocol gate');same(gate['pins'][str(HERE/'check.py')],sha(HERE/'check.py'),'own checker gate')
  same(inputs['checker_sha256'],sha(HERE/'check.py'),'input checker pin')
  p=read(OWNER/'saved/protocol.json');same([s['key'] for s in p['datasets']],KEYS,'fixed12')
  for rec in read(OWNER/'manifest.json')['records']:same(sha(OWNER/rec['snapshot']),rec['sha256'],rec['snapshot'])
  data=read(OWNER/'saved/output/results.json');same(data['status'],'complete','author completed');same([r['key'] for r in data['rows']],KEYS,'all12 outputs')
  same(data['protocol_sha256'],sha(OWNER/'saved/protocol.json'),'author protocol');same(data['manifest_sha256'],sha(OWNER/'manifest.json'),'author manifest')
  for path,h in data['pins'].items():same(sha(path),h,'author result dependency')
  events=[json.loads(x) for x in (OWNER/'saved/output/progress.jsonl').read_text().splitlines()];expected=[('start',None)]+[(stage,key) for key in KEYS for stage in ['dataset_start','dataset_complete']]+[('complete',None)]
  same([(e['event'],e.get('key')) for e in events],expected,'all26 events');assert all(math.isfinite(e['cpu']) and e['cpu']>=0 for e in events)
  assert math.isfinite(data['cpu']) and 0<=events[-2]['cpu']<=data['cpu']<=events[-1]['cpu']
  save();event('started')
  for spec,row in zip(p['datasets'],data['rows'],strict=True):
   assert time.process_time()-start<17;result['current']=spec['key'];save();event('attempt',key=spec['key'])
   same(read(OWNER/'saved/output'/f"{spec['key']}.json"),row,'durable complete row');assert math.isfinite(row['cpu']) and row['cpu']>=0
   got=audit_one(spec,row);result['rows'].append(got);save();event('checked',key=spec['key'])
  by={r['key']:r for r in result['rows']};growth=[]
  for family,depths in [('target',[2,3,4]),('zoo16_4',[2,3,4]),('zoo16_5',[2,3,4]),('gpu',[3,4,5])]:
   for lo,hi in zip(depths,depths[1:]):growth.append((family,lo,hi))
  same(len(data['growth']),8,'eight growth records')
  for (family,lo,hi),g in zip(growth,data['growth'],strict=True):
   same((g['family'],g['k_from'],g['k_to']),(family,lo,hi),'growth identity');a,b=by[f'{family}_k{lo}'],by[f'{family}_k{hi}']
   if not(a['valid'] and b['valid']):same(g['status'],'unresolved','invalid growth');continue
   checkrat(g['regions'],F(b['regions'],a['regions']),'region growth');same(set(g['coverage']),set(a['counts']),'growth deltas')
   for d,n in a['counts'].items():checkrat(g['coverage'][d],F(b['counts'][d],n),'coverage growth')
  same(data['report_count_discrepancies'],sum(not x for r in data['rows'] for x in r.get('source_arithmetic_matches',{}).values()),'source count mismatch tally')
  same(data['adverse_mass_datasets'],[r['key'] for r in data['rows'] if r['nonfinite'] or r['negative']],'adverse tally')
  assert time.process_time()-start<17
  result.update(complete=True,pass_all=True,current=None,source_float_scope='NumPy reporting-kernel execution is not independently repeated. Saved report bindings/displays/boolean mismatch contractions checked; exact binary-mass oracle independently reconstructed.',author_cpu_seconds=data['cpu'],input_pin_sha256=sha(HERE/'input_pins.json'));save();event('complete')
 except BaseException as ex:result.update(failure=repr(ex),traceback=traceback.format_exc());save();event('failed',error=repr(ex));raise
if __name__=='__main__':main()
